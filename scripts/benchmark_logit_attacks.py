#!/usr/bin/env python3
"""
Benchmark latency/throughput impact of logit-sampling attacks.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oml.attack.logit_sampling_attacks import LogitSamplinAttackModel  # noqa: E402
from oml.attack.lookahead import LookaheadAttackedModel  # noqa: E402
from oml.attack.perplexity_filtering import PerplexityFilteringAttackModel  # noqa: E402


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@dataclass
class AttackSpec:
    label: str
    type: str = "baseline"
    processor_name: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    attack_label: str
    attack_type: str
    processor_name: Optional[str]
    run_idx: int
    batch_size: int
    latency_samples: List[Dict[str, float]]
    max_memory_mb: Optional[float]
    attack_kwargs: Dict[str, Any]


def resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    key = name.lower()
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{name}'. Expected one of {list(DTYPE_MAP.keys())}.")
    return DTYPE_MAP[key]


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_prompts(prompts_cfg: DictConfig) -> List[str]:
    values = OmegaConf.to_container(prompts_cfg.get("values", []), resolve=True) or []
    prompt_list: List[str]
    if prompts_cfg.get("file"):
        path = Path(to_absolute_path(prompts_cfg.file))
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        prompt_list = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif values:
        prompt_list = [str(v) for v in values]
    else:
        prompt_list = [prompts_cfg.text]

    if not prompt_list:
        raise ValueError("No prompts provided.")

    batch_size = prompts_cfg.batch_size
    if len(prompt_list) == 1 and batch_size > 1:
        prompt_list = prompt_list * batch_size
    elif len(prompt_list) < batch_size:
        raise ValueError(
            f"Need at least {batch_size} prompts, but only {len(prompt_list)} provided."
        )
    elif len(prompt_list) > batch_size:
        prompt_list = prompt_list[:batch_size]

    return prompt_list


def configure_tokenizer_and_model(
    tokenizer,
    model,
    padding_side: str = "right",
) -> None:
    tokenizer.padding_side = padding_side
    added_special = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        added_special = True
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token = tokenizer.pad_token
    if added_special:
        model.resize_token_embeddings(len(tokenizer))


def encode_batch(
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    encodings = tokenizer(
        list(prompts),
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return {k: v.to(device) for k, v in encodings.items()}


def get_attack_type(value: Optional[str]) -> str:
    if value is None:
        return "baseline"
    value = value.lower()
    if value in {"none", "baseline"}:
        return "baseline"
    if value in {"logit"}:
        return "logit"
    if value in {"lookahead"}:
        return "lookahead"
    if value in {"perplexity", "perplexity_filtering"}:
        return "perplexity"
    return value


def build_attack_model(
    base_model,
    base_tokenizer,
    spec: AttackSpec,
    device: torch.device,
):
    attack_type = get_attack_type(spec.type)
    if attack_type == "baseline":
        return base_model, base_tokenizer
    if attack_type == "logit":
        if not spec.processor_name:
            raise ValueError(f"Attack '{spec.label}' missing processor_name.")
        kwargs = dict(spec.kwargs or {})
        attacked = LogitSamplinAttackModel(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            device=str(device),
            logit_sampling_attack_name=spec.processor_name,
            logit_sampling_attack_kwargs=kwargs,
        )
        attacked.base_model.to(device)
        return attacked, base_tokenizer
    if attack_type == "lookahead":
        kwargs = dict(spec.kwargs or {})
        attacked = LookaheadAttackedModel(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            device=str(device),
            **kwargs,
        )
        return attacked, base_tokenizer
    if attack_type == "perplexity":
        kwargs = dict(spec.kwargs or {})
        attacked = PerplexityFilteringAttackModel(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            **kwargs,
        )
        return attacked, base_tokenizer
    raise ValueError(f"Unsupported attack type '{spec.type}' for '{spec.label}'.")


def run_single_benchmark(
    model,
    tokenizer,
    batch_inputs: Dict[str, torch.Tensor],
    attack_spec: AttackSpec,
    generation_kwargs: Dict[str, Any],
    token_steps: Sequence[int],
    device: torch.device,
) -> Tuple[BenchmarkRun, List[str]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    latency_samples: List[Dict[str, float]] = []
    outputs = None

    for gen_steps in token_steps:
        gen_kwargs = dict(generation_kwargs)
        gen_kwargs["max_new_tokens"] = gen_steps
        gen_kwargs["input_ids"] = batch_inputs["input_ids"]
        gen_kwargs["attention_mask"] = batch_inputs["attention_mask"]
        gen_kwargs["return_dict_in_generate"] = True
        gen_kwargs["output_scores"] = True
        start_time = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latency_samples.append({"max_new_tokens": gen_steps, "latency_ms": elapsed_ms})

    if outputs is None:
        decoded = []
    else:
        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

    # Clean up to avoid memory leak before next run
    del outputs

    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        max_mem = None
    batch_size = batch_inputs["input_ids"].shape[0]

    return (
        BenchmarkRun(
            attack_label=attack_spec.label,
            attack_type=attack_spec.type,
            processor_name=attack_spec.processor_name
            if attack_spec.processor_name
            else (
                "LookaheadAttackedModel"
                if get_attack_type(attack_spec.type) == "lookahead"
                else (
                    "PerplexityFilteringAttackModel"
                    if get_attack_type(attack_spec.type) == "perplexity"
                    else None
                )
            ),
            run_idx=0,
            batch_size=batch_size,
            latency_samples=latency_samples,
            max_memory_mb=max_mem,
            attack_kwargs=attack_spec.kwargs or {},
        ),
        decoded,
    )


def summarize_runs(runs: List[BenchmarkRun]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[BenchmarkRun]] = defaultdict(list)
    for run in runs:
        grouped[run.attack_label].append(run)

    summaries: List[Dict[str, Any]] = []
    for label, items in grouped.items():
        latency_buckets: Dict[int, List[float]] = defaultdict(list)
        for run in items:
            for sample in run.latency_samples:
                latency_buckets[int(sample["max_new_tokens"])].append(sample["latency_ms"])

        latency_curve = []
        for max_tokens in sorted(latency_buckets.keys()):
            samples = latency_buckets[max_tokens]
            avg_latency = mean(samples)
            std_latency = stdev(samples) if len(samples) > 1 else None
            latency_curve.append(
                {
                    "max_new_tokens": max_tokens,
                    "avg_latency_ms": avg_latency,
                    "std_latency_ms": std_latency,
                    "num_samples": len(samples),
                }
            )

        mem_vals = [r.max_memory_mb for r in items if r.max_memory_mb is not None]
        avg_mem = mean(mem_vals) if mem_vals else None

        summaries.append(
            {
                "attack_label": label,
                "attack_type": items[0].attack_type,
                "processor_name": items[0].processor_name,
                "num_runs": len(items),
                "latency_curve": latency_curve,
                "avg_max_memory_mb": avg_mem,
                "attack_kwargs": items[0].attack_kwargs,
            }
        )
    return summaries


def export_results(
    runs: List[BenchmarkRun],
    summaries: List[Dict[str, Any]],
    output_dir: Path,
    save_csv: bool,
    save_json: bool,
    config_snapshot: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if runs and save_csv:
        csv_path = output_dir / "runs.csv"
        fieldnames = [
            "attack_label",
            "attack_type",
            "processor_name",
            "run_idx",
            "batch_size",
            "max_new_tokens",
            "latency_ms",
            "max_memory_mb",
            "attack_kwargs",
        ]
        with csv_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for run in runs:
                for sample in run.latency_samples:
                    writer.writerow(
                        {
                            "attack_label": run.attack_label,
                            "attack_type": run.attack_type,
                            "processor_name": run.processor_name,
                            "run_idx": run.run_idx,
                            "batch_size": run.batch_size,
                            "max_new_tokens": sample["max_new_tokens"],
                            "latency_ms": sample["latency_ms"],
                            "max_memory_mb": run.max_memory_mb,
                            "attack_kwargs": json.dumps(run.attack_kwargs),
                        }
                    )

    if save_json:
        json_path = output_dir / "results.json"
        payload = {
            "config": config_snapshot,
            "runs": [asdict(run) for run in runs],
            "summaries": summaries,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        json_path.write_text(json.dumps(payload, indent=2))


def format_float(value: Optional[float], suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


@hydra.main(config_path="../configs", config_name="logit_attack_benchmark", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.benchmark.seed)

    device = torch.device(cfg.model.device)
    dtype = resolve_dtype(cfg.model.get("dtype"))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_name,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=dtype,
        trust_remote_code=cfg.model.trust_remote_code,
        device_map=cfg.model.device_map,
    )
    if cfg.model.device_map is None:
        base_model.to(device)
    base_model.eval()

    configure_tokenizer_and_model(tokenizer, base_model, cfg.prompts.padding_side)

    prompts = prepare_prompts(cfg.prompts)
    batch_inputs = encode_batch(tokenizer, prompts, device)

    if cfg.benchmark.get("token_steps"):
        token_steps = sorted({int(v) for v in cfg.benchmark.token_steps})
    else:
        token_steps = list(range(1, cfg.generation.max_new_tokens + 1))

    max_tokens_allowed = cfg.generation.max_new_tokens
    if any(step <= 0 for step in token_steps):
        raise ValueError("benchmark.token_steps must contain positive integers.")
    if any(step > max_tokens_allowed for step in token_steps):
        raise ValueError(
            f"benchmark.token_steps may not exceed generation.max_new_tokens ({max_tokens_allowed})."
        )

    generation_cfg = {
        "max_new_tokens": cfg.generation.max_new_tokens,
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
        "do_sample": cfg.generation.do_sample,
        "num_beams": cfg.generation.num_beams,
        "use_cache": cfg.generation.use_cache,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    attack_specs: List[AttackSpec] = []
    for raw_spec in cfg.attacks:
        spec_dict = OmegaConf.to_container(raw_spec, resolve=True)
        attack_type = get_attack_type(spec_dict.get("type"))
        processor_name = spec_dict.get("processor_name")
        if attack_type == "lookahead" and processor_name is None:
            processor_name = "LookaheadAttackedModel"
        if attack_type == "perplexity" and processor_name is None:
            processor_name = "PerplexityFilteringAttackModel"
        attack_specs.append(
            AttackSpec(
                label=spec_dict.get("label") or processor_name or "unnamed",
                type=attack_type,
                processor_name=processor_name,
                kwargs=spec_dict.get("kwargs") or {},
            )
        )

    runs: List[BenchmarkRun] = []
    output_dir = Path.cwd()
    print(f"Saving benchmark artifacts under {output_dir}")

    for attack in attack_specs:
        print(f"\n=== Attack: {attack.label} ({attack.type}) ===")
        attack_model, attack_tokenizer = build_attack_model(
            base_model, tokenizer, attack, device
        )
        attack_batch_inputs = batch_inputs
        if attack_tokenizer is not tokenizer:
            attack_batch_inputs = encode_batch(attack_tokenizer, prompts, device)

        for warmup_idx in range(cfg.benchmark.warmup_runs):
            print(f"  Warmup run {warmup_idx + 1}/{cfg.benchmark.warmup_runs}...", end="", flush=True)
            run_single_benchmark(
                attack_model,
                attack_tokenizer,
                attack_batch_inputs,
                attack,
                generation_kwargs=generation_cfg,
                token_steps=token_steps,
                device=device,
            )
            print(" done.")

        for run_idx in range(cfg.benchmark.runs):
            print(f"  Measured run {run_idx + 1}/{cfg.benchmark.runs}...")
            run_result, decoded = run_single_benchmark(
                attack_model,
                attack_tokenizer,
                attack_batch_inputs,
                attack,
                generation_kwargs=generation_cfg,
                token_steps=token_steps,
                device=device,
            )
            run_result.run_idx = run_idx
            runs.append(run_result)

            if cfg.benchmark.print_decoded > 0 and run_idx == 0:
                sample_outputs = decoded[: cfg.benchmark.print_decoded]
                for idx, text in enumerate(sample_outputs):
                    print(f"    Sample {idx + 1}: {text}")

            curve_str = ", ".join(
                f"{sample['max_new_tokens']}→{sample['latency_ms']:.1f}ms"
                for sample in run_result.latency_samples
            )
            print(f"    latency curve: {curve_str}")

    summaries = summarize_runs(runs)
    config_snapshot = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "prompts": OmegaConf.to_container(cfg.prompts, resolve=True),
        "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        "benchmark": OmegaConf.to_container(cfg.benchmark, resolve=True),
        "attacks": OmegaConf.to_container(cfg.attacks, resolve=True),
    }
    export_results(
        runs,
        summaries,
        output_dir,
        save_csv=cfg.benchmark.save_csv,
        save_json=cfg.benchmark.save_json,
        config_snapshot=config_snapshot,
    )

    print("\n=== Summary ===")
    for summary in summaries:
        curve = ", ".join(
            f"{point['max_new_tokens']}→{format_float(point['avg_latency_ms'], ' ms', 1)}"
            for point in summary["latency_curve"]
        )
        print(f"{summary['attack_label']:<20}latency: {curve}")


if __name__ == "__main__":
    main()

