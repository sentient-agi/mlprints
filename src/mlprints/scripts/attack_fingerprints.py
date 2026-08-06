"""Prepare a model attack from a YAML config."""

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from mlprints.attack import AttackModel
from mlprints.common.attacks import ATTACK_ALGOS, check_attack_algo
from mlprints.common.utils import (
    get_timestamp_uuid,
    load_implementation,
    load_yaml,
    normalize_str_to_path,
    save_yaml,
    set_seeds,
)
from mlprints.scripts.utils import (
    get_experiment_dir,
    looks_like_hf_model_id,
    resolve_checkpoint_path,
)


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config_path",
        help="Path to the config YAML",
    )
    parser.add_argument("--implementation", help="Local attack Python file")
    parser.add_argument(
        "--model-checkpoint",
        required=True,
        help="Hugging Face model ID, checkpoint, or training run to attack",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint to select from a training run: latest, final, step, or name",
    )
    parser.add_argument(
        "--experiments-dir",
        help="Override MLPRINTS_EXPERIMENTS_DIR; requires --experiment-name",
    )
    parser.add_argument(
        "--experiment-name",
        help="Experiment directory name; requires --experiments-dir",
    )


def _get_experiment_dir(
    model_checkpoint: str,
    experiments_dir: str | None,
    experiment_name: str | None,
) -> Path:
    if (experiments_dir is None) != (experiment_name is None):
        raise ValueError(
            "--experiments-dir and --experiment-name must be provided together"
        )
    if experiments_dir is not None:
        return get_experiment_dir(experiments_dir, experiment_name)

    checkpoint_path = Path(model_checkpoint)
    parts = checkpoint_path.parts
    if {"fingerprints", "trained", "checkpoints"}.issubset(parts):
        fingerprints_index = parts.index("fingerprints")
        if fingerprints_index:
            experiment_dir = Path(*parts[:fingerprints_index])
            print(f"Auto-detected experiment directory: {experiment_dir}")
            return experiment_dir

    raise ValueError(
        "could not infer the experiment directory from --model-checkpoint; "
        "provide --experiments-dir and --experiment-name"
    )


def _get_attack_dir(experiment_dir: Path, algo_name: str) -> Path:
    attack_dir = experiment_dir / "attacks" / algo_name / get_timestamp_uuid()
    attack_dir.mkdir(parents=True, exist_ok=False)
    return attack_dir


def prepare_attack(
    model_checkpoint: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    algo_name = config["algo"]["name"]
    check_attack_algo(algo_name)
    return ATTACK_ALGOS[algo_name]["prepare"](
        model_checkpoint,
        **config["algo"]["params"],
    )


def _register_implementation(path: str, algo_name: str) -> Path:
    implementation_path = normalize_str_to_path(path)
    module = load_implementation(implementation_path)
    attack_classes = [
        value
        for value in vars(module).values()
        if (
            isinstance(value, type)
            and value.__module__ == module.__name__
            and issubclass(value, AttackModel)
        )
    ]
    if len(attack_classes) != 1:
        raise ValueError(
            "attack implementations must define exactly one AttackModel subclass"
        )

    ATTACK_ALGOS[algo_name] = {
        "prepare": getattr(module, algo_name),
        "class": attack_classes[0],
    }
    return implementation_path


def _resolve_model_checkpoint(path_or_model_id: str, checkpoint: str) -> str:
    if looks_like_hf_model_id(path_or_model_id):
        return path_or_model_id

    model_path = normalize_str_to_path(path_or_model_id)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return resolve_checkpoint_path(str(model_path), checkpoint=checkpoint)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)
    args = parser.parse_args(argv)

    config_path = normalize_str_to_path(args.config_path)
    config = load_yaml(config_path)
    if not isinstance(config, dict) or not isinstance(config.get("algo"), dict):
        raise ValueError("attack config must contain an algo mapping")

    algo_name = config["algo"].get("name")
    if not algo_name:
        raise ValueError("attack config must contain algo.name")

    implementation_path: Path | None = None
    if args.implementation:
        implementation_path = _register_implementation(
            args.implementation,
            algo_name,
        )

    model_checkpoint = _resolve_model_checkpoint(
        args.model_checkpoint,
        args.checkpoint,
    )
    experiment_dir = _get_experiment_dir(
        model_checkpoint,
        args.experiments_dir,
        args.experiment_name,
    )
    set_seeds(config["seed"])

    attack_dir = _get_attack_dir(experiment_dir, algo_name)
    if implementation_path is not None:
        shutil.copy2(implementation_path, attack_dir / "implementation.py")

    attack_config, metadata = prepare_attack(model_checkpoint, config)
    save_yaml(attack_dir / "config.yaml", config)
    save_yaml(attack_dir / "metadata.yaml", metadata)
    save_yaml(attack_dir / "attack.yaml", attack_config)
    print(f"Prepared {algo_name} attack for: {model_checkpoint}")
    print(f"Attack saved to: {attack_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())