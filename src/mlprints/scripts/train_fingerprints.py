"""Train previously generated fingerprints from a YAML config."""

import argparse
import os

from mlprints.common.fingerprints import FINGERPRINT_ALGOS, check_fingerprint_train
from mlprints.common.utils import (
    load_implementation,
    load_yaml,
    normalize_str_to_path,
)
from mlprints.scripts.generate_fingerprints import train_fingerprints
from mlprints.scripts.utils import iter_grid_configs


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config_path",
        help="Path to the config YAML",
    )
    parser.add_argument("--implementation", help="Local fingerprint Python file")
    parser.add_argument(
        "--fingerprints-dir",
        required=True,
        help="Path to the saved fingerprints directory",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse matching training runs",
    )


def main(argv: list | None = None) -> int:
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)
    args = parser.parse_args(argv)

    config_path, fingerprints_dir = normalize_str_to_path(
        args.config_path,
        args.fingerprints_dir,
    )
    config = load_yaml(config_path)
    if args.implementation:
        module = load_implementation(args.implementation)
        name = config["algo"]["name"]
        FINGERPRINT_ALGOS[name] = {
            "generate": getattr(module, name),
            "train": getattr(module, f"train_{name}", None),
        }

    fingerprints_path = fingerprints_dir / "fingerprints.yaml"
    if not fingerprints_path.is_file():
        raise FileNotFoundError(f"Fingerprints file not found: {fingerprints_path}")

    fingerprints = load_yaml(fingerprints_path)
    print(f"Loaded {len(fingerprints)} fingerprints from: {fingerprints_path}")

    check_fingerprint_train(config["algo"]["name"])
    for index, total, training_config, _ in iter_grid_configs(
        config,
        include_prefixes=[("algo", "training")],
    ):
        train_fingerprints(
            training_config,
            fingerprints_dir,
            fingerprints,
            index=index,
            total=total,
            skip_existing=args.skip_existing,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())