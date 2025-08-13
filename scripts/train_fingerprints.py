"""
scripts/generate_fingerprints.py

Train fingerprints from input config YAML.
"""

import argparse
import torch
import yaml
import os
from oml.fingerprint.perinucleus import train_perinucleus


def add_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the command-line arguments to the parser.
    """
    parser.add_argument(
        "config",
        type=os.path.abspath,
        help="Path to the config YAML",
    )
    parser.add_argument(
        "fingerprints", type=os.path.abspath, help="Path to the saved fingerprints"
    )
    return


def set_seeds(seed: int = 42):
    """
    Set seeds for torch.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_fingerprints(fingerprints: list[dict], config: dict):
    """
    Train fingerprints into a given model given a set of fingerprints and a
    training config.
    """

    assert config["algo"]["name"] == "perinucleus", "Only perinucleus is implemented!"

    train_perinucleus(fingerprints, **config["algo"]["training"])


if __name__ == "__main__":
    # get the config
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    fingerprints_path = args.fingerprints
    with open(fingerprints_path, "r", encoding="utf8") as file:
        fingerprints = yaml.safe_load(file)

    # generate and save fingerprints
    train_fingerprints(fingerprints, config)
