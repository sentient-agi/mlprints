#!/usr/bin/env python
"""
Run utility + fingerprint evaluation for a saved language-model checkpoint on a *single* GPU.

This version re-uses the existing evaluation helpers defined in
`eval_utility.py` (for benchmark performance) and `check_fingerprints.py`
(for fingerprint verification) instead of the earlier custom / vLLM code-path.
It is designed to be launched asynchronously by the training callback
(`AsyncEvalLauncherCallback`) and therefore keeps the same CLI interface that
was used previously.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Local imports – fall back to PYTHONPATH if executed via subprocess ----------
# ---------------------------------------------------------------------------
if not os.environ.get("PYTHONPATH"):
    os.environ["PYTHONPATH"] = os.getcwd()

import lm_eval                                   # type: ignore

import old_files.eval_utility as eval_utility                              # Benchmark helper (utility)
from old_files.check_fingerprints import eval_backdoor_acc # Verification helper (V)
from old_files.generate_finetuning_data import (
    get_fingerprint_ds,
)

###############################################################################
# Helper functions                                                             #
###############################################################################

def _flatten_lm_eval_results(results: dict) -> dict:
    """Flatten the nested lm-eval results dict to a single level."""
    flat: dict[str, float] = {}
    for task_name, task_res in results.get("results", {}).items():
        for metric_name, value in task_res.items():
            if metric_name.endswith("_stderr"):
                continue  # skip standard-error entries
            try:
                flat[f"U/{task_name}/{metric_name}"] = float(value)
            except (ValueError, TypeError):
                pass
    return flat


def _compute_openllm_average(results: dict, use_tiny: bool) -> Optional[float]:
    """Return the macro-average accuracy across OpenLLM benchmark tasks."""
    datasets = eval_utility.ALL_DATASETS_TINY if use_tiny else eval_utility.ALL_DATASETS
    total, n = 0.0, 0
    for ds in datasets.values():
        for task in ds["tasks"]:
            res = results.get("results", {}).get(task, {})
            for metric in ds["metric"]:
                for key in (metric, f"{metric},none"):
                    if key in res:
                        total += res[key]
                        n += 1
                        break
    if n == 0:
        return None
    return total / n


###############################################################################
# Main evaluation routine                                                      #
###############################################################################

def run_eval(
    *,
    model_path: str,
    result_path: str,
    tasks: List[str],
    batch_size: int,
    limit: Optional[int],
    num_fewshot: int,
    # Fingerprint-specific
    num_fingerprints: int,
    max_key_len: int,
    max_resp_len: int,
    fingerprints_file_path: Optional[str],
    fingerprint_strategy: str,
    use_tiny_benchmarks: bool = False,
) -> None:
    """Evaluate <model_path> for utility + verification and write JSONL."""

    logging.info("Loading model from %s", model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    # -----------------------------------------------------------------------
    # 1) Utility – lm-evaluation-harness                                    
    # -----------------------------------------------------------------------
    task_str = ",".join(tasks) if tasks else ("tinyBenchmarks" if use_tiny_benchmarks else "openllm")
    logging.info("Running utility evaluation on tasks: %s", task_str)

    lm_args = {
        "model": "hf",
        "model_args": f"pretrained={model_path},local_files_only=True,trust_remote_code=True,dtype=bfloat16",
        "tasks": task_str,
        "batch_size": batch_size,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "apply_chat_template": True,
    }

    util_results = lm_eval.simple_evaluate(**{k: v for k, v in lm_args.items() if v is not None})
    flat_util = _flatten_lm_eval_results(util_results)

    # Macro-average for OpenLLM if applicable
    macro_acc = _compute_openllm_average(util_results, use_tiny_benchmarks)
    if macro_acc is not None:
        flat_util["U/openllm_macro_acc"] = macro_acc

    # -----------------------------------------------------------------------
    # 2) Verification – fingerprint retention                                
    # -----------------------------------------------------------------------
    logging.info("Preparing fingerprint dataset (n=%d) …", num_fingerprints)
    fp_dataset, _ = get_fingerprint_ds(
        tokenizer,
        num_fingerprints=num_fingerprints,
        key_length=max_key_len,
        response_length=max_resp_len,
        deterministic_length=True,
        strategy=fingerprint_strategy,
        cache_path=fingerprints_file_path,
        num_responses_per_fingerprint=1,
        get_eval_set=True,
        seed=42,
    )

    logging.info("Running fingerprint verification …")
    acc_vec, frac_vec = eval_backdoor_acc(
        model,
        tokenizer,
        fp_dataset["train"],
        prompt_templates=["{}"],
        temperature=0.0,
        verbose=False,
        output_file_path=None,
        use_chat_template=False,
    )

    fp_metrics = {
        "fingerprint/acc_mean": float(acc_vec.mean()),
        "fingerprint/frac_acc_mean": float(frac_vec.mean()),
    }

    # -----------------------------------------------------------------------
    # Combine + persist                                                      
    # -----------------------------------------------------------------------
    metrics = {
        **flat_util,
        **fp_metrics,
        "model_path": model_path,
        "timestamp": datetime.utcnow().isoformat(),
    }

    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "a", encoding="utf-8") as fp:
        json.dump(metrics, fp)
        fp.write("\n")
    logging.info("Evaluation finished. Results appended to %s", result_path)


###############################################################################
# CLI                                                                         #
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the saved checkpoint")
    parser.add_argument("--result_path", default=None, help="File to append metrics to (JSONL)")
    parser.add_argument("--tasks", default="openllm", help="Comma-separated list of lm-eval tasks or 'openllm'/'tinyBenchmarks'")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for lm-eval")
    parser.add_argument("--limit", type=int, default=None, help="Limit docs per task (int or fraction ≤1.0)")
    parser.add_argument("--num_fewshot", type=int, default=0)
    # Fingerprint args -------------------------------------------------------
    parser.add_argument("--num_fingerprints", type=int, default=128)
    parser.add_argument("--max_key_len", type=int, default=16)
    parser.add_argument("--max_resp_len", type=int, default=1)
    parser.add_argument("--fingerprints_file_path", type=str, default=None)
    parser.add_argument("--fingerprint_strategy", type=str, default="english")
    # Tiny benchmark flag -----------------------------------------------------
    parser.add_argument("--tiny", action="store_true", help="Use tinyBenchmarks instead of openllm")

    args = parser.parse_args()

    result_path = args.result_path or os.path.join(args.model_path, "eval_metrics.jsonl")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    run_eval(
        model_path=args.model_path,
        result_path=result_path,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        num_fingerprints=args.num_fingerprints,
        max_key_len=args.max_key_len,
        max_resp_len=args.max_resp_len,
        fingerprints_file_path=args.fingerprints_file_path,
        fingerprint_strategy=args.fingerprint_strategy,
        use_tiny_benchmarks=args.tiny,
    ) 