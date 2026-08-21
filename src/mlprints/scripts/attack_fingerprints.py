"""Prepare a model attack from a YAML config."""

import argparse
import os
import shutil
from pathlib import Path

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
    parser.add_argument(
        "--implementation",
        help="Local attack Python file",
    )
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
        implementation_path = normalize_str_to_path(args.implementation)
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
                "attack implementations must define exactly one "
                "AttackModel subclass"
            )
        ATTACK_ALGOS[algo_name] = {
            "prepare": getattr(module, algo_name),
            "class": attack_classes[0],
        }

    if looks_like_hf_model_id(args.model_checkpoint):
        model_checkpoint = args.model_checkpoint
    else:
        model_path = normalize_str_to_path(args.model_checkpoint)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        model_checkpoint = resolve_checkpoint_path(
            str(model_path),
            checkpoint=args.checkpoint,
        )

    if (args.experiments_dir is None) != (args.experiment_name is None):
        raise ValueError(
            "--experiments-dir and --experiment-name must be provided together"
        )
    if args.experiments_dir is not None:
        experiment_dir = get_experiment_dir(
            args.experiments_dir,
            args.experiment_name,
        )
    else:
        checkpoint_path = Path(model_checkpoint)
        parts = checkpoint_path.parts
        if {"fingerprints", "trained", "checkpoints"}.issubset(parts):
            fingerprints_index = parts.index("fingerprints")
        else:
            fingerprints_index = 0
        if not fingerprints_index:
            raise ValueError(
                "could not infer the experiment directory from "
                "--model-checkpoint; provide --experiments-dir and "
                "--experiment-name"
            )
        experiment_dir = Path(*parts[:fingerprints_index])
        print(f"Auto-detected experiment directory: {experiment_dir}")
    set_seeds(config["seed"])

    attack_dir = (
        experiment_dir
        / "attacks"
        / algo_name
        / get_timestamp_uuid()
    )
    attack_dir.mkdir(parents=True, exist_ok=False)
    if implementation_path is not None:
        shutil.copy2(implementation_path, attack_dir / "implementation.py")

    check_attack_algo(algo_name)
    attack_config, metadata = ATTACK_ALGOS[algo_name]["prepare"](
        model_checkpoint,
        **config["algo"]["params"],
    )
    save_yaml(attack_dir / "config.yaml", config)
    save_yaml(attack_dir / "metadata.yaml", metadata)
    save_yaml(attack_dir / "attack.yaml", attack_config)
    print(f"Prepared {algo_name} attack for: {model_checkpoint}")
    print(f"Attack saved to: {attack_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())