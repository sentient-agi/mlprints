"""
Script to train fingerprints from YAML configs.

Utilizes the function train_fingerprints from the generate_fingerprints.py script.
"""

import argparse
import os
from typing import Optional
import yaml

from oml.common.utils import (
    normalize_str_to_path,
    set_seeds,
    save_yaml
)
from oml.common.fingerprints import check_fingerprint_train
from utils import get_trained_dir, get_checkpoints_dir
from generate_fingerprints import train_fingerprints


def _add_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the command-line arguments to the parser.
    """
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the config YAML",
    )
    parser.add_argument(
        "--fingerprints-dir",
        type=str,
        required=True,
        help="Path to the saved fingerprints directory",
    )


def main(argv: Optional[list] = None) -> int:
    os.environ["NCCL_DEBUG"] = "WARN"  # suppresses NCCL INFO verbosity

    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)
    args = parser.parse_args(argv)

    config_path, fingerprints_dir = normalize_str_to_path(args.config_path, args.fingerprints_dir)
    with open(config_path, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    fingerprints_path = fingerprints_dir / "fingerprints.yaml"
    if not fingerprints_path.exists():
        raise FileNotFoundError(f"Fingerprints file not found: {fingerprints_path}")
    
    with open(fingerprints_path, "r", encoding="utf8") as file:
        fingerprints = yaml.safe_load(file)
    
    print(f"Loaded {len(fingerprints)} fingerprints from: {fingerprints_path}")

    set_seeds(config["seed"])

    algo_name = config["algo"]["name"]
    check_fingerprint_train(algo_name)

    trained_dir = get_trained_dir(fingerprints_dir)
    checkpoints_dir = get_checkpoints_dir(trained_dir)

    print(f"\nTraining {algo_name} fingerprints...")
    training_metadata = train_fingerprints(config, checkpoints_dir, fingerprints)
    
    save_yaml(trained_dir / "config.yaml", config)
    save_yaml(trained_dir / "metadata.yaml", training_metadata)
    
    print(f"Training complete!\nCheckpoints saved to: {checkpoints_dir}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())