"""
   scripts/generate_fingerprints.py

   Generate fingerprints from input config YAML.
"""
import torch
import os
from datetime import datetime
import yaml
import argparse
from oml.fingerprint.rofl import rofl
from oml.fingerprint.perinucleus import perinucleus, train_perinucleus


def add_args(parser: argparse.ArgumentParser) -> None:
    """
        Add the command-line arguments to the parser.
    """
    parser.add_argument(
        "config",
        type=os.path.abspath,
        help="Path to the config YAML",
    )
    return


def set_seeds(seed: int=42):
    """
        Set seeds for torch.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_save_dir(config: dict):
    """
        Get timestamped output directory, named with algo.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(
        "experiments", config["algo"]["name"], f"{timestamp}"
    )
    return save_dir


def generate_fingerprints(config: dict):
    """
        Generate fingerprints (and related meta information dictionary)
        according to the config dictionary.
    """

    assert config["algo"]["name"] == "rofl" or config["algo"]["name"] == "perinucleus", "Only rofl and perinucleus are implemented!"
    set_seeds(config["seed"])

    if config["algo"]["name"] == "rofl":
        fps, metas = rofl(**config["algo"]["params"])
        return fps, metas
    elif config["algo"]["name"] == "perinucleus":
        fps = perinucleus(**config["algo"]["params"])
        print(fps)
        return fps, []


if __name__ == "__main__":

    # get the config
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)

    config_path = parser.parse_args().config
    with open(config_path, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)
    
    # prep the output dir
    save_dir = get_save_dir(config)
    os.makedirs(save_dir)

    config_dump_path = os.path.join(save_dir, "config.yaml")
    with open(config_dump_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    # generate and save fingerprints
    fingerprints, metas = generate_fingerprints(config)

    fp_dump_path = os.path.join(save_dir, "fingerprints.yaml")
    with open(fp_dump_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(fingerprints, f, sort_keys=False, allow_unicode=True)
    
    meta_dump_path = os.path.join(save_dir, "metas.yaml")
    with open(meta_dump_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metas, f, sort_keys=False, allow_unicode=True)