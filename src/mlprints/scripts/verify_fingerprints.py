"""Verify fingerprints on a specific model and verifier config."""

import argparse
import os
from pathlib import Path
from typing import Any

from mlprints.common.utils import (
    get_timestamp_uuid,
    load_yaml,
    normalize_str_to_path,
    save_yaml,
    set_seeds,
)
from mlprints.measure import measure_verification_score
from mlprints.scripts.utils import (
    iter_grid_configs,
    load_model_and_tokenizer,
    resolve_checkpoint_path,
)


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config_path",
        help="Verifier YAML config; may also contain model, generation, and measurement sections",
    )
    parser.add_argument(
        "--fingerprints",
        required=True,
        help="Path to fingerprints.yaml or a directory containing it",
    )
    parser.add_argument(
        "--model",
        help="Hugging Face model ID, checkpoint, or MLprints attack directory",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer ID or path; defaults to the model",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint to select from a training run: latest, final, step, or checkpoint name",
    )
    parser.add_argument("--device-map", help="Transformers device map, such as auto or cuda:0")
    parser.add_argument("--dtype", help="Model dtype, such as auto, bfloat16, or float16")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--system-prompt")
    parser.add_argument(
        "--output-dir",
        help="Run output directory; defaults beside the fingerprints file",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse matching verification runs",
    )


def _load_fingerprints(path: str) -> tuple[Path, list[dict[str, Any]]]:
    fingerprints_path = normalize_str_to_path(path)
    if fingerprints_path.is_dir():
        fingerprints_path = fingerprints_path / "fingerprints.yaml"
    if not fingerprints_path.is_file():
        raise FileNotFoundError(f"Fingerprints file not found: {fingerprints_path}")

    fingerprints = load_yaml(fingerprints_path)
    if not isinstance(fingerprints, list):
        raise ValueError("fingerprints YAML must contain a list of fingerprint mappings")
    return fingerprints_path, fingerprints


def _model_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model_config = dict(config.get("model", {}))
    overrides = {
        "model_id": args.model,
        "tokenizer_id": args.tokenizer,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    model_config.update(
        {key: value for key, value in overrides.items() if value is not None}
    )

    model_id = model_config.get("model_id")
    if not model_id:
        raise ValueError("provide --model or a top-level model.model_id in the config")
    model_config["model_id"] = resolve_checkpoint_path(
        str(model_id),
        checkpoint=args.checkpoint,
    )
    return model_config


def _find_existing_run(
    output_root: Path,
    config: dict[str, Any],
    model_config: dict[str, Any],
    fingerprints_path: Path,
) -> Path | None:
    for run_dir in output_root.iterdir():
        config_path = run_dir / "config.yaml"
        result_path = run_dir / "result.yaml"
        if not config_path.is_file() or not result_path.is_file():
            continue
        result = load_yaml(result_path)
        if (
            load_yaml(config_path) == config
            and result.get("model") == model_config
            and result.get("fingerprints_path") == str(fingerprints_path)
        ):
            return run_dir
    return None


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)
    args = parser.parse_args(argv)

    config_path = normalize_str_to_path(args.config_path)
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("verification config must be a mapping")
    if "verifier" not in config:
        raise ValueError("verification config must contain a verifier section")

    fingerprints_path, fingerprints = _load_fingerprints(args.fingerprints)
    queries = config.get("queries")
    if queries is None:
        queries = [
            fingerprint["query"]
            for fingerprint in fingerprints
            if "query" in fingerprint
        ]
    if not queries:
        raise ValueError("provide queries in the config or fingerprints")

    model_config = _model_config(config, args)
    loaded = load_model_and_tokenizer(model_config, role="verification")

    measurement = dict(config.get("measurement", {}))
    measurement.update(
        {
            key: value
            for key, value in {
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "apply_chat_template": args.apply_chat_template,
                "system_prompt": args.system_prompt,
            }.items()
            if value is not None
        }
    )
    config["measurement"] = measurement
    if args.seed is not None:
        config["seed"] = args.seed

    generation_params = dict(config.get("generation_params", {}))
    reserved_generation_params = {
        "apply_chat_template",
        "extract_top_k",
        "max_new_tokens",
        "num_return_sequences",
        "output_scores",
        "system_prompt",
    }
    conflicts = reserved_generation_params.intersection(generation_params)
    if conflicts:
        raise ValueError(
            "move measurement-controlled generation keys to the measurement section: "
            f"{sorted(conflicts)}"
        )

    output_root = (
        normalize_str_to_path(args.output_dir)
        if args.output_dir
        else fingerprints_path.parent / "verification"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    for index, total, run_config, _ in iter_grid_configs(
        config,
        include_prefixes=[
            ("verifier", "params"),
            ("measurement",),
            ("generation_params",),
        ],
    ):
        measurement = run_config["measurement"]
        generation_params = dict(run_config.get("generation_params", {}))
        if args.skip_existing:
            existing = _find_existing_run(
                output_root,
                run_config,
                model_config,
                fingerprints_path,
            )
            if existing is not None:
                print(f"[{index}/{total}] Reusing verification: {existing}")
                continue

        seed = run_config.get("seed", 42)
        set_seeds(seed)

        score, metadata = measure_verification_score(
            model=loaded["model"],
            tokenizer=loaded["tokenizer"],
            queries=queries,
            verification_config=run_config,
            fingerprints=fingerprints,
            max_new_tokens=measurement.get("max_new_tokens"),
            batch_size=measurement.get("batch_size", 128),
            apply_chat_template=measurement.get("apply_chat_template", False),
            system_prompt=measurement.get("system_prompt"),
            generation_params=generation_params,
        )

        output_dir = output_root / get_timestamp_uuid()
        output_dir.mkdir(exist_ok=False)
        save_yaml(output_dir / "config.yaml", run_config)
        save_yaml(
            output_dir / "result.yaml",
            {
                "verification_score": float(score),
                "verification_metadata": metadata,
                "model": model_config,
                "fingerprints_path": str(fingerprints_path),
            },
        )
        print(f"[{index}/{total}] Verification score: {score:.6f}")
        print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
