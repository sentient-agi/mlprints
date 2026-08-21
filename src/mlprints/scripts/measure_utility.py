"""Measure model utility with LightEval from a YAML config."""

import argparse
from pathlib import Path
from typing import Any

from mlprints.common.utils import load_yaml, normalize_str_to_path, save_yaml, set_seeds
from mlprints.measure.utility import (
    UtilityResult,
    evaluate_model,
    serialize_utility_results,
)
from mlprints.scripts.utils import get_experiment_dir, load_model_and_tokenizer


def measure_from_config(
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[UtilityResult, str]:
    model_config = dict(config.get("model", {}))
    evaluation = dict(config.get("evaluation", {}))
    tasks = evaluation.pop("tasks", None)
    if not tasks:
        raise ValueError("utility config requires evaluation.tasks")

    batch_size = evaluation.pop("batch_size", None)
    generation_params = evaluation.pop("generation_params", {})
    benchmark_name = evaluation.pop("name", tasks)
    loaded = load_model_and_tokenizer(model_config, role="utility")

    return evaluate_model(
        model=loaded["model"],
        tokenizer=loaded["tokenizer"],
        eval_benchmark_name=benchmark_name,
        tasks=tasks,
        batch_size=batch_size,
        benchmark_config=evaluation,
        generation_params=generation_params,
        tracker_output_dir=output_dir,
    )


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config_path", help="Utility evaluation YAML config")
    parser.add_argument("--experiments-dir", help="Override MLPRINTS_EXPERIMENTS_DIR")
    parser.add_argument("--experiment-name", help="Use a specific experiment directory name")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)
    args = parser.parse_args(argv)

    config_path = normalize_str_to_path(args.config_path)
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("utility config must be a mapping")

    output_dir = get_experiment_dir(args.experiments_dir, args.experiment_name)
    set_seeds(config.get("seed", 42))
    save_yaml(output_dir / "config.yaml", config)
    results, _resolved_tasks = measure_from_config(config, output_dir)
    save_yaml(
        output_dir / "utility.yaml",
        serialize_utility_results(results),
    )
    print(f"Utility results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
