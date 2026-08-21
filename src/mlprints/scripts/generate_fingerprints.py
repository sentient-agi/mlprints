"""Generate and optionally train fingerprints from a YAML config."""

import argparse
import copy
import os
from pathlib import Path
from typing import Any

from mlprints.common.utils import (
    get_timestamp_uuid,
    load_implementation,
    load_yaml,
    normalize_str_to_path,
    save_yaml,
    set_seeds,
)
from mlprints.common.fingerprints import (
    FINGERPRINT_ALGOS,
    check_fingerprint_algo,
    check_fingerprint_train,
)
from mlprints.loading import load_model_and_tokenizer
from mlprints.scripts.utils import (
    find_existing_fingerprint_dir,
    find_existing_trained_dir,
    get_checkpoints_dir,
    get_experiment_dir,
    get_trained_dir,
    iter_grid_configs,
)


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config_path",
        help="Path to the config YAML",
    )
    parser.add_argument("--implementation", help="Local fingerprint Python file")
    parser.add_argument(
        "--model",
        help="Target model ID or path; overrides the configured target model",
    )
    parser.add_argument(
        "--tokenizer",
        help="Target tokenizer ID or path; overrides the configured target tokenizer",
    )

    parser.add_argument(
        "--experiments-dir",
        help="Optional override path for the output experiments directory. If not set, "
             "falls back to using the MLPRINTS_EXPERIMENTS_DIR environment variable",
    )

    parser.add_argument(
        "--experiment-name",
        help="Optional override name for the experiment's directory. If not set, "
            "falls back to creating a *new* hash timestamped directory",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable automatically training the fingerprints, based on the config "
            "file (if supported by the chosen fingerprint type)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse matching fingerprint and training runs",
    )


def target_model_overrides(
    config: dict[str, Any],
    *,
    model_id: str | None = None,
    tokenizer_id: str | None = None,
) -> dict[str, Any]:
    """Return a config with target model overrides applied to every stage."""
    if model_id is None and tokenizer_id is None:
        return config

    updated = copy.deepcopy(config)
    for section in (
        updated["algo"].get("params"),
        updated["algo"].get("training"),
    ):
        if section is None:
            continue
        models = section.get("models")
        if not models:
            continue
        target_role = "target" if "target" in models else next(iter(models))
        target = models[target_role]
        if model_id is not None:
            target["model_id"] = model_id
        if tokenizer_id is not None:
            target["tokenizer_id"] = tokenizer_id
    return updated


def _load_model_kwargs(
    model_configs: dict[str, dict[str, Any]],
    *,
    is_train: bool = False,
) -> dict[str, Any]:
    model_configs = dict(model_configs)
    if not model_configs:
        raise ValueError("models must contain at least one model configuration")
    if "target" not in model_configs:
        first_role = next(iter(model_configs))
        target_config = model_configs.pop(first_role)
        model_configs = {"target": target_config, **model_configs}

    kwargs = {}
    for role, model_config in model_configs.items():
        loaded = load_model_and_tokenizer(
            model_config,
            role=role,
            is_train=is_train and role == "target",
        )
        kwargs[f"{role}_model"] = loaded["model"]
        kwargs[f"{role}_tokenizer"] = loaded["tokenizer"]
    return kwargs


def generate_fingerprints(
    config: dict[str, Any],
    *,
    model_id: str | None = None,
    tokenizer_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = target_model_overrides(
        config,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
    )
    algo_config = config["algo"]
    params = dict(algo_config["params"])
    model_configs = dict(params.pop("models"))
    generate_function = FINGERPRINT_ALGOS[algo_config["name"]]["generate"]
    return generate_function(
        **_load_model_kwargs(model_configs),
        **params,
    )


def train_fingerprints(
    config: dict[str, Any],
    fingerprints_dir: Path,
    fingerprints: list[dict[str, Any]],
    *,
    index: int,
    total: int,
    skip_existing: bool = False,
    model_id: str | None = None,
    tokenizer_id: str | None = None,
) -> Path:
    config = target_model_overrides(
        config,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
    )
    algo_name = config["algo"]["name"]
    if skip_existing:
        existing = find_existing_trained_dir(fingerprints_dir, config)
        if existing is not None:
            print(f"[{index}/{total}] Reusing trained run: {existing}")
            return existing

    set_seeds(config["seed"])
    trained_dir = get_trained_dir(fingerprints_dir)
    checkpoints_dir = get_checkpoints_dir(trained_dir)
    print(f"[{index}/{total}] Training {algo_name} fingerprints...")

    algo_config = config["algo"]
    training_params = dict(algo_config["training"])
    model_configs = training_params.pop("models", None)
    if model_configs is None:
        model_configs = algo_config["params"]["models"]
    train_function = FINGERPRINT_ALGOS[algo_name]["train"]
    metadata = train_function(
        **_load_model_kwargs(dict(model_configs), is_train=True),
        checkpoints_dir=checkpoints_dir,
        fingerprints=fingerprints,
        **training_params,
    )
    save_yaml(trained_dir / "config.yaml", config)
    save_yaml(trained_dir / "metadata.yaml", metadata)
    print(f"Training complete: {checkpoints_dir}")
    return trained_dir


def main(argv: list | None = None) -> int:
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)
    args = parser.parse_args(argv)

    config_path = normalize_str_to_path(args.config_path)
    config = load_yaml(config_path)
    config = target_model_overrides(
        config,
        model_id=args.model,
        tokenizer_id=args.tokenizer,
    )
    if args.implementation:
        module = load_implementation(args.implementation)
        name = config["algo"]["name"]
        FINGERPRINT_ALGOS[name] = {
            "generate": getattr(module, name),
            "train": getattr(module, f"train_{name}", None),
        }

    experiment_dir = get_experiment_dir(args.experiments_dir, args.experiment_name)
    algo_name = config["algo"]["name"]
    check_fingerprint_algo(algo_name)
    if args.train:
        check_fingerprint_train(algo_name)

    for index, total, generation_config, _ in iter_grid_configs(
        config,
        include_prefixes=[("algo", "params")],
    ):
        existing = (
            find_existing_fingerprint_dir(
                experiment_dir,
                algo_name,
                generation_config,
            )
            if args.skip_existing
            else None
        )
        if existing is not None:
            fingerprints_dir = existing
            fingerprints = load_yaml(fingerprints_dir / "fingerprints.yaml")
            print(f"[{index}/{total}] Reusing fingerprints: {fingerprints_dir}")
        else:
            set_seeds(generation_config["seed"])
            fingerprints_dir = (
                experiment_dir
                / "fingerprints"
                / algo_name
                / get_timestamp_uuid()
            )
            fingerprints_dir.mkdir(parents=True, exist_ok=False)
            save_yaml(fingerprints_dir / "config.yaml", generation_config)
            print(f"[{index}/{total}] Generating {algo_name} fingerprints...")
            fingerprints, metadata = generate_fingerprints(generation_config)
            save_yaml(fingerprints_dir / "fingerprints.yaml", fingerprints)
            save_yaml(fingerprints_dir / "metadata.yaml", metadata)
            print(f"Generated {len(fingerprints)} fingerprints")

        if args.train:
            for train_index, train_total, training_config, _ in iter_grid_configs(
                generation_config,
                include_prefixes=[("algo", "training")],
            ):
                train_fingerprints(
                    training_config,
                    fingerprints_dir,
                    fingerprints,
                    index=train_index,
                    total=train_total,
                    skip_existing=args.skip_existing,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())