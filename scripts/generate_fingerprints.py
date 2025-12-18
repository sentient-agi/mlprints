"""
Script to generate (and optionally train) fingerprints from YAML configs.
"""

import argparse
import inspect
import os
from pathlib import Path
from typing import Optional, Any
import yaml

from utils import get_experiment_dir, get_trained_dir, get_checkpoints_dir, load_model_and_tokenizer
from oml.common.utils import (
    normalize_str_to_path, 
    get_timestamp_uuid, 
    set_seeds, 
    save_yaml
)
from oml.common.fingerprints import (
    FINGERPRINT_ALGOS,
    check_fingerprint_algo,
    check_fingerprint_train
)


def _add_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds command-line arguments to the argument parser.
    """
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the config YAML",
    )

    parser.add_argument(
        "--experiments-dir",
        type=str,
        required=False,
        help="Optional override path for the output experiments directory. If not set, "
             "falls back to using the MLPRINTS_EXPERIMENTS_DIR environment variable",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        required=False,
        default=None,
        help="Optional override name for the experiment's directory. If not set, "
            "falls back to creating a *new* hash timestamped directory",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable automatically training the fingerprints, based on the config "
            "file (if supported by the chosen fingerprint type)",
    )


def _get_fingerprints_dir(experiment_dir: Path, algo_name: str) -> Path:
    """
    Creates and returns the path to a new unique hash timestamped fingerprints directory
    given an experiment's path and a fingerprinting algorithm's name.
    """
    fingerprints_dir = experiment_dir / "fingerprints" / algo_name / get_timestamp_uuid()
    fingerprints_dir.mkdir(parents=True, exist_ok=False)

    return fingerprints_dir

def _route_extra_models(
    models_configs: dict[str, dict[str, Any]],
    func: Any,
    algo_name: str
) -> dict[str, Any]:
    """
    Map model configs to a function's kwargs and loads them.

    - (1) If the function explicitly accepts `{role}_model` / `{role}_tokenizer`,
      pass those explicitly.
    - (2) Any remaining roles are passed as a list into the first plural parameter
      containing "models" (e.g. `models`, `extra_models`, ...). If a plural
      "tokenizers" parameter exists, pass those too.
    """
    DEFAULT_MODEL_PARAM_NAME = "{role}_model"
    DEFAULT_TOKENIZER_PARAM_NAME = "{role}_tokenizer"
    DEFAULT_PLURAL_MODEL_PARAM_KEYWORD = "models" # e.g. `models`, `extra_models`, ...
    DEFAULT_PLURAL_TOKENIZER_PARAM_KEYWORD = "tokenizers" # e.g. `tokenizers`, `extra_tokenizers`, ...
    kwargs_mapped = {} # to be returned at the end
    func_signature = inspect.signature(func)
    accepted_kwargs = set(func_signature.parameters.keys())
    loaded_models_tokenizers = {
        key: load_model_and_tokenizer(cfg, role=key)
        for key, cfg in models_configs.items()
    }

    # 1) Attempt explicit per-role mapping
    for role in list(loaded_models_tokenizers.keys()):
        loaded = loaded_models_tokenizers[role]
        used_explicit = False
        model_param_name = DEFAULT_MODEL_PARAM_NAME.format(role=role)
        tokenizer_param_name = DEFAULT_TOKENIZER_PARAM_NAME.format(role=role)

        if model_param_name in accepted_kwargs:
            kwargs_mapped[model_param_name] = loaded["model"]
            used_explicit = True
        if tokenizer_param_name in accepted_kwargs:
            kwargs_mapped[tokenizer_param_name] = loaded["tokenizer"]
            used_explicit = True

        if used_explicit:
            loaded_models_tokenizers.pop(role)

    # 2) Attempt assigning remaining roles to plural params
    models_param_name = None
    tokenizers_param_name = None
    
    for param_name in func_signature.parameters.keys():
        param_lower = param_name.lower()
        if param_lower.endswith(
            (DEFAULT_MODEL_PARAM_NAME.format(role=""),
            DEFAULT_TOKENIZER_PARAM_NAME.format(role=""))
        ):
            continue
            
        if DEFAULT_PLURAL_MODEL_PARAM_KEYWORD in param_lower:
            if models_param_name is not None:
                raise ValueError(
                    f"Multiple 'models' parameters found for {algo_name} ({func.__name__}): "
                    f"{models_param_name}, {param_name}. Provide only one."
                )
            models_param_name = param_name
            
        if DEFAULT_PLURAL_TOKENIZER_PARAM_KEYWORD in param_lower:
            if tokenizers_param_name is not None:
                raise ValueError(
                    f"Multiple 'tokenizers' parameters found for {algo_name} ({func.__name__}): "
                    f"{tokenizers_param_name}, {param_name}. Provide only one."
                )
            tokenizers_param_name = param_name

    if loaded_models_tokenizers:
        if models_param_name is None:
            raise ValueError(
                f"Unmapped model configs {list(loaded_models_tokenizers.keys())} "
                f"for {algo_name} ({func.__name__}). "
                "Provide *_model params or a plural models parameter."
            )

        kwargs_mapped[models_param_name] = [loaded["model"] for loaded in loaded_models_tokenizers.values()]
        if tokenizers_param_name is not None:
            kwargs_mapped[tokenizers_param_name] = [loaded["tokenizer"] for loaded in loaded_models_tokenizers.values()]

    return kwargs_mapped


def generate_fingerprints(config: dict) -> tuple:
    """
    Generates fingerprints according to a config dictionary.
    """
    algo_name = config["algo"]["name"]
    params = dict(config["algo"].get("params", {}))
    models = params.pop("models", {})
    if not models:
        raise ValueError("models must contain at least one model configuration")

    generation_kwargs = {}
    
    # Load target model: use "target" key, or fall back to first entry
    if "target" in models:
        target_config = models.pop("target")
    else:
        first_model_key = next(iter(models))
        target_config = models.pop(first_model_key)
        print(f"No 'target' key in models, using '{first_model_key}' as the target") # TODO: consider logging
    
    target_loaded = load_model_and_tokenizer(target_config, role="target")
    generation_kwargs["target_model"] = target_loaded["model"]
    generation_kwargs["target_tokenizer"] = target_loaded["tokenizer"]

    generate_function = FINGERPRINT_ALGOS[algo_name]["generate"]
    generation_kwargs.update(_route_extra_models(models, generate_function, algo_name))
    generation_kwargs.update(params)

    accepted_args = set(inspect.signature(generate_function).parameters.keys())
    provided_args = set(generation_kwargs.keys())
    invalid_args = provided_args - accepted_args
    if invalid_args:
        raise ValueError(
            f"Invalid arguments for {algo_name} generation: {invalid_args}.\n"
            f"Accepted: {accepted_args}, Provided: {provided_args}"
        )
    
    result = generate_function(**generation_kwargs)
    
    return result


def train_fingerprints(
        config: dict,
        checkpoints_dir: str,
        fingerprints: list
    ) -> dict:
    """
    Trains fingerprints according to a config dictionary.
    """
    algo_name = config["algo"]["name"]
    # params is only used as a fallback source for models if not in training_params
    params = dict(config["algo"].get("params", {}))
    training_params = dict(config["algo"].get("training", {}))
    # Extract models from training_params first, fallback to params if not present
    models = training_params.pop("models", params.pop("models", {}))
    if not models:
        raise ValueError("Training requires models/models_dict with at least one model configuration")
    
    training_kwargs = {}
    
    # Load target model: use "target" key, or fall back to first entry
    if "target" in models:
        target_config = models.pop("target")
    else:
        first_model_key = next(iter(models))
        target_config = models.pop(first_model_key)
        print(f"No 'target' key in models, using '{first_model_key}' as the target") # TODO: consider logging
    
    target_loaded = load_model_and_tokenizer(target_config, role="target")
    training_kwargs["target_model"] = target_loaded["model"]
    training_kwargs["target_tokenizer"] = target_loaded["tokenizer"]

    train_function = FINGERPRINT_ALGOS[algo_name]["train"]
    training_kwargs.update(_route_extra_models(models, train_function, algo_name))
    training_kwargs["fingerprints"] = fingerprints
    training_kwargs["checkpoints_dir"] = checkpoints_dir
    training_kwargs.update(training_params)
    
    accepted_args = set(inspect.signature(train_function).parameters.keys())
    provided_args = set(training_kwargs.keys())
    invalid_args = provided_args - accepted_args
    if invalid_args:
        raise ValueError(
            f"Invalid arguments for {algo_name} training: {invalid_args}.\n"
            f"Accepted: {accepted_args}, Provided: {provided_args}"
        )
    
    result = train_function(**training_kwargs)
    
    return result


def main(argv: Optional[list] = None) -> int:
    os.environ["NCCL_DEBUG"] = "WARN"  # suppresses NCCL INFO verbosity

    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)
    args = parser.parse_args(argv)

    config_path = normalize_str_to_path(args.config_path)
    with open(config_path, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    experiment_dir = get_experiment_dir(args.experiments_dir, args.experiment_name)

    set_seeds(config["seed"])

    algo_name = config["algo"]["name"]
    check_fingerprint_algo(algo_name)
    fingerprints_dir = _get_fingerprints_dir(experiment_dir, algo_name)
    save_yaml(fingerprints_dir / "config.yaml", config)  # copy generation config
    

    print(f"\nGenerating {algo_name} fingerprints...")
    fingerprints, fingerprints_metadata = generate_fingerprints(config)
    
    save_yaml(fingerprints_dir / "fingerprints.yaml", fingerprints)
    save_yaml(fingerprints_dir / "metadata.yaml", fingerprints_metadata)
    print(f"Generated {len(fingerprints)} fingerprints")

    if args.train:
        print(f"Training requested\nTraining {algo_name} fingerprints...")
        check_fingerprint_train(algo_name)
        trained_dir = get_trained_dir(fingerprints_dir)
        checkpoints_dir = get_checkpoints_dir(trained_dir)

        training_metadata = train_fingerprints(config, checkpoints_dir, fingerprints)
        save_yaml(trained_dir / "config.yaml", config)
        save_yaml(trained_dir / "metadata.yaml", training_metadata)
        print(f"Training complete!\nCheckpoints saved to: {checkpoints_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())