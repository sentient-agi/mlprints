"""
scripts/attack_fingerprints.py

Prepare fingerprint attacks from the input config YAML.
"""

import argparse
import os
from pathlib import Path
from typing import Optional
import yaml

from oml.common.utils import normalize_str_to_path, get_timestamp_uuid, set_seeds, save_yaml
from oml.common.attacks import ATTACK_ALGOS, check_attack_algo


def add_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the command-line arguments to the parser.
    """
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the config YAML",
    )
    
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint directory to attack",
    )

    parser.add_argument(
        "--experiments-dir",
        type=str,
        required=False,
        help="Experiments directory path. Must be used together with --experiment-name. "
             "If neither are provided, auto-detects from model checkpoint path.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        required=False,
        default=None,
        help="Experiment name. Must be used together with --experiments-dir. "
            "If neither are provided, auto-detects from model checkpoint path.",
    )


def get_experiment_dir(model_checkpoint: Optional[str] = None,
                       experiments_dir: Optional[str] = None,
                       experiment_name: Optional[str] = None) -> Path:
    """
    Get the experiment directory, either by auto-detecting from model checkpoint path
    or using provided experiments_dir and experiment_name.
    
    Args:
        model_checkpoint: Path to the model checkpoint
        experiments_dir: Optional experiments directory (must be used with experiment_name)
        experiment_name: Optional experiment name (must be used with experiments_dir)
    
    Returns:
        Path to the experiment directory
    """
    # Check that either both or neither experiments_dir and experiment_name are provided
    if (experiments_dir is None) != (experiment_name is None):
        raise ValueError(
            "You must specify either BOTH --experiments-dir and --experiment-name, "
            "or NEITHER (to auto-detect from model checkpoint path)"
        )
    
    if experiments_dir is not None and experiment_name is not None:
        experiments_dir = normalize_str_to_path(experiments_dir)
        experiment_dir = experiments_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir
    
    # Neither provided - auto-detect from model checkpoint path
    # Expected structure: experiment/fingerprints/algo/timestamp/trained/timestamp/checkpoints
    # We need to go up 6 levels from checkpoints to get to experiment root
    
    # First check if this looks like a checkpoint path
    if model_checkpoint is None:
        raise ValueError(
            "No model checkpoint specified. Please use --model-checkpoint to specify the model checkpoint path. "
            "Otherwise, use --experiments-dir and --experiment-name to specify the experiment directory."
        )
    
    path_parts = Path(model_checkpoint).parts
    
    # Look for the path structure
    if len(set(['checkpoints', 'trained', 'fingerprints']) & set(path_parts)) == 3:
        try:
            # Find the index of 'fingerprints'
            fingerprints_idx = path_parts.index('fingerprints')
            
            # Check if fingerprints has at least one parent
            if fingerprints_idx >= 1:
                # Get everything up to (but not including) 'fingerprints'
                experiment_dir = Path(*path_parts[:fingerprints_idx])
                print(f"Auto-detected experiment directory: {experiment_dir}")
                return experiment_dir
        except IndexError:
            raise ValueError(
                "Could not auto-detect experiment directory from model checkpoint path.\n"
                "The model checkpoint should be in the structure: "
                "experiment/fingerprints/algo/timestamp/trained/timestamp/checkpoints\n"
                "Please specify --experiments-dir and --experiment-name explicitly."
            )


def get_attack_dir(experiment_dir: Path, config: dict) -> Path:
    """
    Create and return a hash timestamped attack directory.
    """
    algo_name = config["algo"]["name"]
    
    check_attack_algo(algo_name)
    
    # Use algorithm name directly as directory name
    attack_dir = experiment_dir / "attacks" / algo_name / get_timestamp_uuid()
    attack_dir.mkdir(parents=True, exist_ok=False)  # has to be unique
    
    return attack_dir


def prepare_attack(model_checkpoint: str, config: dict) -> tuple:
    """
    Prepare fingerprint attack according to the config dictionary.
    
    Returns:
        tuple: (attack_config, metadata) where:
            - attack_config contains the configuration to recreate the attack
            - metadata contains attack metadata
    """
    algo_name = config["algo"]["name"]

    check_attack_algo(algo_name)
    
    attack_config, metadata = ATTACK_ALGOS[algo_name]["prepare"](model_checkpoint, **config["algo"]["params"])

    return attack_config, metadata





def main():
    os.environ["NCCL_DEBUG"] = "WARN" # supresses NCCL INFO verbosity

    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    args = parser.parse_args()

    config_path = normalize_str_to_path(args.config_path)
    with open(config_path, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    # Handle model checkpoint - could be HuggingFace ID or local path
    model_checkpoint_str = args.model_checkpoint
    
    # Check if it's a HuggingFace model ID (contains forward slash but not a file path)
    if "/" in model_checkpoint_str and not model_checkpoint_str.startswith("/") and not model_checkpoint_str.startswith("./"):
        # It's a HuggingFace model ID, use as-is
        print(f"Using HuggingFace model: {model_checkpoint_str}")
    else:
        # It's a local path, verify it exists
        model_checkpoint = normalize_str_to_path(model_checkpoint_str)
        if not model_checkpoint.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")
        model_checkpoint_str = str(model_checkpoint)

    experiment_dir = get_experiment_dir(model_checkpoint_str, args.experiments_dir, args.experiment_name)

    set_seeds(config["seed"])

    attack_dir = get_attack_dir(experiment_dir, config)
    
    # Prepare attack - returns configuration and metadata
    attack_config, metadata = prepare_attack(model_checkpoint_str, config)
    
    # Save the three required files
    save_yaml(attack_dir / "config.yaml", config)  # Original input configuration
    save_yaml(attack_dir / "metadata.yaml", metadata)  # Attack metadata
    save_yaml(attack_dir / "attack.yaml", attack_config)  # Configuration to recreate the attacked model

    print(f"\nAttack preparation complete!")
    print(f"Model attacked: {model_checkpoint_str}")
    print(f"Attack directory: {attack_dir}")


if __name__ == "__main__":
    raise SystemExit(main())