# lm_eval_utils.py
"""
Light-weight wrapper around `lm_eval` for quick utility benchmarking.

This module exposes a single public helper ``evaluate_model`` that is able to
run the (tiny) OpenLLM benchmarking suite on a local or HF-Hub model path and
optionally log the results to Weights & Biases.

The goal is **not** to re-implement a full benchmarking abstraction – that will
live in ``engine.utility.benchmarks``. Instead, this file serves as a thin
compatibility bridge so that existing training / verification scripts that used
``eval_utility.py`` continue to work after the refactor, while also providing a
clean importable API for other internal modules.

Example
-------
>>> from engine.utility.lm_eval_utils import evaluate_model
>>> results = evaluate_model("/path/to/final_model")
>>> print(results["total_accuracy"])
"""
from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover – optional dependency
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False

import datasets  # noqa: F401 – must be imported *before* lm_eval to enable remote code

# datasets requires this env variable to allow custom code when running under
# restricted environments (e.g. some CI). The original eval_utility explicitly
# sets it so we replicate that behaviour here.
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True  # type: ignore

import lm_eval  # type: ignore

__all__ = [
    "evaluate_model",
    "load_evaluation",
]

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# NOTE: Keep the structure identical to the original script so that we can
# compute the aggregated accuracy in the exact same way.
ALL_DATASETS: Dict[str, Dict[str, object]] = {
    "ARC": {"n_shot": 25, "tasks": ["arc_challenge"], "metric": ["acc_norm"]},
    "HellaSwag": {"n_shot": 10, "tasks": ["hellaswag"], "metric": ["acc_norm"]},
    "TruthfulQA": {"n_shot": 0, "tasks": ["truthfulqa_mc2"], "metric": ["acc"]},
    "MMLU": {"n_shot": 5, "tasks": ["mmlu"], "metric": ["acc"]},
    "Winogrande": {"n_shot": 5, "tasks": ["winogrande"], "metric": ["acc"]},
    "GSM8k": {
        "n_shot": 5,
        "tasks": ["gsm8k"],
        "metric": [
            "exact_match,strict-match",
            "exact_match,flexible-extract",
        ],
    },
}

ALL_DATASETS_TINY: Dict[str, Dict[str, object]] = {
    "tinyARC": {"n_shot": 25, "tasks": ["tinyArc"], "metric": ["acc_norm"]},
    "tinyHellaswag": {"n_shot": 10, "tasks": ["tinyHellaswag"], "metric": ["acc_norm"]},
    "tinyTruthfulQA": {"n_shot": 0, "tasks": ["tinyTruthfulQA"], "metric": ["acc"]},
    "tinyMMLU": {"n_shot": 5, "tasks": ["tinyMMLU"], "metric": ["acc_norm"]},
    "tinyWinogrande": {"n_shot": 5, "tasks": ["tinyWinogrande"], "metric": ["acc_norm"]},
    "tinyGSM8k": {
        "n_shot": 5,
        "tasks": ["tinyGSM8k"],
        "metric": [
            "exact_match,strict-match",
            "exact_match,flexible-extract",
        ],
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    *,
    wandb_run_name: str | None = None,
    use_tiny_benchmarks: bool = False,
    eval_batch_size: int = 6,
    apply_chat_template: bool | None = None,
    delete_model: bool = False,
) -> Dict[str, float]:
    """Run lm-eval benchmark suite and return aggregated accuracy.

    Parameters
    ----------
    model_path:
        Local path («…/final_model") or HF Hub repo id.
    wandb_run_name:
        If provided and ``wandb`` is installed, results are logged there.
    use_tiny_benchmarks:
        Whether to run the much faster *tiny* benchmark variant.
    eval_batch_size:
        Forward batch size.
    apply_chat_template:
        If ``True`` the HF chat template is applied prior evaluation. If
        ``None`` we will try to infer the value from the model's
        ``fingerprinting_config.json`` (if present) and fall back to ``False``.
    delete_model:
        Delete the on-disk checkpoint after evaluation (useful inside sweeps).

    Returns
    -------
    Dict[str, float]
        Mapping of task-level accuracies plus a top-level key
        ``"total_accuracy"``.
    """

    model_path = str(model_path)
    config_path = model_path.replace("final_model", "fingerprinting_config.json")

    # Try load config to inherit metadata such as "use_chat_template".
    config: Dict[str, object] = {"model_path": model_path}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config.update(json.load(f))  # type: ignore[arg-type]
        except json.JSONDecodeError:
            print(f"[lm_eval_utils] Warning – could not decode {config_path}, continuing…")

    if apply_chat_template is None:
        apply_chat_template = bool(config.get("use_chat_template", False))

    # Initialise WandB early so that the run does not get marked as crashed if
    # we exit via Ctrl-C or exception.
    if wandb_run_name and _WANDB_AVAILABLE:
        wandb.init(project=wandb_run_name, config=config, name=wandb_run_name)

    torch.cuda.empty_cache()

    task_suite = "tinyBenchmarks" if use_tiny_benchmarks else "openllm"

    ds_results = lm_eval.simple_evaluate(
        model="hf",
        model_args=(
            f"pretrained={model_path},local_files_only=True,"
            "trust_remote_code=True,dtype=bfloat16"
        ),
        tasks=task_suite,
        batch_size=eval_batch_size,
        apply_chat_template=apply_chat_template,
    )

    # Persist raw results next to the model so that later scripts can load them
    suffix = "eval_results_tiny" if use_tiny_benchmarks else "eval_results"
    out_file = model_path.replace("final_model", suffix) + ".json"
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(ds_results["results"], f, indent=2)
    except (FileNotFoundError, PermissionError) as exc:
        print(f"[lm_eval_utils] Warning – could not save evaluation results: {exc}")

    total_accuracy = _aggregate_accuracy(ds_results["results"], use_tiny_benchmarks)

    if wandb_run_name and _WANDB_AVAILABLE:
        wandb.log({
            (
                "eval/OpenLLMTinyLeaderboard"
                if use_tiny_benchmarks
                else "eval/OpenLLMLeaderboard"
            ): total_accuracy
        })

    print("=" * 40)
    print(f"Total accuracy: {total_accuracy:.4f}")
    print("=" * 40)

    if delete_model:
        try:
            print(f"Deleting model artefacts at {model_path}…")
            # Use pathlib for robustness; must ensure we don't accidentally delete "./".
            model_root = Path(model_path).expanduser().resolve()
            if model_root.exists() and "final_model" in model_root.name:
                import shutil
                shutil.rmtree(model_root)
        except Exception as exc:  # pragma: no cover – best-effort clean-up
            print(f"[lm_eval_utils] Warning –   failed to delete model: {exc}")

    # Return fine-grained metrics as well so that callers can dig deeper.
    return {"total_accuracy": total_accuracy, **ds_results["results"]}


def load_evaluation(model_path: str, *, use_tiny_benchmarks: bool = False) -> Tuple[float, Dict[str, float]]:
    """Load a previously saved evaluation JSON and recompute aggregated score."""
    suffix = "eval_results_tiny" if use_tiny_benchmarks else "eval_results"
    json_path = model_path.replace("final_model", suffix) + ".json"

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Evaluation file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        results: Dict[str, float] = json.load(f)

    total_accuracy = _aggregate_accuracy(results, use_tiny_benchmarks)
    print("=" * 40)
    print(f"Total accuracy: {total_accuracy:.4f}")
    print("=" * 40)
    return total_accuracy, results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _aggregate_accuracy(results: Dict[str, Dict[str, float]], use_tiny: bool) -> float:
    """Compute macro-average accuracy over the dataset definitions above."""
    datasets_cfg = ALL_DATASETS_TINY if use_tiny else ALL_DATASETS

    total_acc: float = 0.0
    total_tasks: int = 0

    for ds in datasets_cfg.values():
        for task in ds["tasks"]:  # type: ignore[index]
            task_res: Dict[str, float] = results.get(task, {})  # default empty dict
            for metric in ds["metric"]:  # type: ignore[index]
                # The lm-eval harness sometimes stores variants such as "metric,none"
                val: Optional[float] = None
                if metric in task_res:
                    val = task_res[metric]
                else:
                    alt_key = f"{metric},none"
                    val = task_res.get(alt_key)

                if val is not None:
                    total_acc += val
                    total_tasks += 1
                else:
                    print(f"[lm_eval_utils] Metric {metric} missing for task {task}")

    return total_acc / total_tasks if total_tasks else 0.0 