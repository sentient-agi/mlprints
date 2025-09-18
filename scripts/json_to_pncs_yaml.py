#!/usr/bin/env python3
"""
Quick-and-dirty converter: fingerprinting JSON -> PNCS-style YAML.

Source of truth is the JSON. We map a subset of keys into a structure that
"looks like" configs/pncs_config.yaml. All known JSON keys are mapped into
PNCS fields, per user-specified rules.

Usage:
  python scripts/json_to_pncs_yaml.py --input /path/to/fingerprinting_config.json --output /path/to/output.yaml
  # or write to stdout
  python scripts/json_to_pncs_yaml.py --input /path/to/fingerprinting_config.json
"""


import argparse
import json
import os
import sys
from typing import Any, Dict, List


def get_model_name_from_family_and_size(model_family: str, model_size: str) -> str:
    if model_family == "llama":
        if "8b" in model_size.lower():
            return "meta-llama/Llama-3.1-{}".format(model_size)
        return f"meta-llama/Llama-3.2-{model_size}"
    elif model_family == "qwen":
        return f"Qwen/Qwen2.5-{model_size}"
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

def build_pncs_like_config(source: Dict[str, Any]) -> Dict[str, Any]:
    """Build a PNCS-like YAML dictionary from the JSON source.

    Mappings (when present in JSON):
      - seed -> seed
      - algo.name -> "perinucleus" (constant label)
      - algo.params.num_fingerprints -> num_fingerprints
      - algo.params.key_length -> max_key_length
      - algo.params.response_length -> max_response_length
      - algo.params.fingerprint_generation_strategy -> fingerprint_generation_strategy
      - algo.params.remove_eos_token_from_response -> remove_eos_token_from_response
      - algo.params.num_responses_per_fingerprint -> num_responses_per_fingerprint
      - algo.params.fingerprints_file_path -> fingerprints_file_path
      - algo.params.models_dict.base.model_id -> get_model_name_from_family_and_size(model_family, model_size)

      - training.learning_rate -> learning_rate
      - training.weight_decay -> weight_decay
      - training.batch_size -> batch_size
      - training.num_train_epochs -> num_train_epochs
      - training.early_stop_loss -> early_stopping_threshold
      - training.output_dir -> result_path
      - training.data_split -> data_split
      - training.expansion_rate -> expansion_rate
      - training.use_chat_template -> use_chat_template
      - training.augmentation.use_augmentation_prompts -> use_augmentation_prompts
      - training.augmentation.benign_proportion -> benign_proportion
      - training.lora.use_lora -> use_lora
      - training.lora.rank -> lora_rank
      - training.lora.alpha_ratio -> lora_alpha_ratio

    Skipped fields by request: model_path, fixed_gradient_accumulation_steps, config_hash.
    """
    output: Dict[str, Any] = {}

    # Top-level seed
    if "seed" in source:
        output["seed"] = source["seed"]

    # algo block
    algo_params: Dict[str, Any] = {}

    if "num_fingerprints" in source:
        algo_params["num_fingerprints"] = source["num_fingerprints"]
    if "max_key_length" in source:
        algo_params["key_length"] = source["max_key_length"]
    if "max_response_length" in source:
        algo_params["response_length"] = source["max_response_length"]
    if "fingerprint_generation_strategy" in source:
        algo_params["fingerprint_generation_strategy"] = source["fingerprint_generation_strategy"]
    if "remove_eos_token_from_response" in source:
        algo_params["remove_eos_token_from_response"] = source["remove_eos_token_from_response"]
    if "num_responses_per_fingerprint" in source:
        algo_params["num_responses_per_fingerprint"] = source["num_responses_per_fingerprint"]

    # fingerprints file path mapping
    if "fingerprints_file_path" in source:
        algo_params["fingerprints_file_path"] = source["fingerprints_file_path"]

    # Quick-dirty model_id assembly from model_family and model_size.
    # If either is missing, skip this block.
    model_family = source.get("model_family")
    model_size = source.get("model_size")
    
    if model_family and model_size:
        models_dict = {
            "base": {
                # Minimal assembly; adjust if a precise HF model_id mapping is desired.
                "model_id": get_model_name_from_family_and_size(model_family, model_size),
            }
        }
        algo_params["models_dict"] = models_dict

    output["algo"] = {
        "name": "perinucleus",
        "params": algo_params,
    }

    # training block
    training: Dict[str, Any] = {}
    if "learning_rate" in source:
        training["learning_rate"] = source["learning_rate"]
    if "weight_decay" in source:
        training["weight_decay"] = source["weight_decay"]
    if "batch_size" in source:
        training["batch_size"] = source["batch_size"]
    if "num_train_epochs" in source:
        training["num_train_epochs"] = source["num_train_epochs"]
    if "early_stopping_threshold" in source:
        training["early_stop_loss"] = source["early_stopping_threshold"]
    if "result_path" in source:
        training["output_dir"] = source["result_path"]
    if "data_split" in source:
        training["data_split"] = source["data_split"]
    if "expansion_rate" in source:
        training["expansion_rate"] = source["expansion_rate"]
    if "use_chat_template" in source:
        training["use_chat_template"] = source["use_chat_template"]

    # augmentation sub-block
    augmentation_added = False
    if "use_augmentation_prompts" in source:
        training.setdefault("augmentation", {})["use_augmentation_prompts"] = source["use_augmentation_prompts"]
        augmentation_added = True
    if "benign_proportion" in source:
        training.setdefault("augmentation", {})["benign_proportion"] = source["benign_proportion"]
        augmentation_added = True

    # lora sub-block
    if "use_lora" in source or "lora_rank" in source or "lora_alpha_ratio" in source:
        lora_block: Dict[str, Any] = {}
        if "use_lora" in source:
            lora_block["use_lora"] = source["use_lora"]
        if "lora_rank" in source:
            lora_block["rank"] = source["lora_rank"]
        if "lora_alpha_ratio" in source:
            lora_block["alpha_ratio"] = source["lora_alpha_ratio"]
        training["lora"] = lora_block

    output["training"] = training

    return output


def yaml_dump(data: Any, indent_size: int = 2) -> str:
    """Minimal YAML emitter that supports dicts, lists, and scalars.

    This avoids third-party deps (PyYAML). It's intentionally simple and
    suitable for configs composed of scalars, lists, and nested dicts.
    Strings are quoted to avoid YAML parsing surprises.
    """

    def scalar_to_yaml(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        # Quote strings; escape quotes minimally
        s = str(value).replace("\"", "\\\"")
        return f'"{s}"'

    lines: List[str] = ["---"]

    def emit(node: Any, level: int, key_context: bool = False) -> None:
        indent = " " * (indent_size * level)
        if isinstance(node, dict):
            for k, v in node.items():
                key_str = str(k)
                if isinstance(v, (dict, list)):
                    lines.append(f"{indent}{key_str}:")
                    emit(v, level + 1)
                else:
                    lines.append(f"{indent}{key_str}: {scalar_to_yaml(v)}")
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    lines.append(f"{indent}-")
                    emit(item, level + 1)
                else:
                    lines.append(f"{indent}- {scalar_to_yaml(item)}")
        else:
            # Root-level scalar (unlikely here)
            lines.append(f"{indent}{scalar_to_yaml(node)}")

    emit(data, 0)
    return "\n".join(lines) + "\n"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert fingerprinting JSON to PNCS-style YAML")
    parser.add_argument("--input", required=True, help="Path to source JSON config")
    parser.add_argument(
        "--output", "-o", default="", help="Path to write YAML. Defaults to stdout if not set."
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    try:
        with open(args.input, "r") as f:
            source_cfg = json.load(f)
    except Exception as exc:
        print(f"Failed to read JSON config: {exc}", file=sys.stderr)
        return 1

    pncs_like = build_pncs_like_config(source_cfg)
    yaml_text = yaml_dump(pncs_like)

    if args.output:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                f.write(yaml_text)
        except Exception as exc:
            print(f"Failed to write YAML: {exc}", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(yaml_text)

    return 0


if __name__ == "__main__":
    import pathlib
    for exp_root in [pathlib.Path("experiments/models/perinucleus_better")]:

        for run_dir in sorted(exp_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if not os.path.exists(run_dir / "fingerprinting_config.json"):
                continue
            with open(run_dir / "fingerprinting_config.json", "r") as f:
                source_cfg = json.load(f)
            pncs_like = build_pncs_like_config(source_cfg)
            yaml_text = yaml_dump(pncs_like)
            with open(run_dir / "fp_config.yaml", "w") as f:
                f.write(yaml_text)


