"""Verify fingerprints on a specific model and verifier config."""

import argparse
import json
import os
from collections.abc import Mapping, Sequence

from mlprints.common.cache import resolve_cached_common_asset
from mlprints.common.utils import (
    get_timestamp_uuid,
    load_implementation,
    load_yaml,
    normalize_str_to_path,
    save_yaml,
    set_seeds,
)
from mlprints.common.verifiers import VERIFIERS
from mlprints.loading import load_model_and_tokenizer
from mlprints.measure import measure_verification_score
from mlprints.scripts.utils import (
    iter_grid_configs,
    resolve_checkpoint_path,
)


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config_path",
        help="Verification YAML config",
    )
    parser.add_argument("--implementation", help="Local verifier Python file")
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
    if args.implementation:
        module = load_implementation(args.implementation)
        name = config["verifier"]["name"]
        VERIFIERS[name] = {
            "verification_score": getattr(module, f"verify_{name}")
        }
    fingerprints_path = normalize_str_to_path(args.fingerprints)
    if fingerprints_path.is_dir():
        fingerprints_path = fingerprints_path / "fingerprints.yaml"
    if not fingerprints_path.is_file():
        raise FileNotFoundError(
            f"Fingerprints file not found: {fingerprints_path}"
        )
    fingerprints = load_yaml(fingerprints_path)
    if not isinstance(fingerprints, list):
        raise ValueError(
            "fingerprints YAML must contain a list of fingerprint mappings"
        )
    queries = config.get("queries")
    if queries is None:
        queries = [
            fingerprint["query"]
            for fingerprint in fingerprints
            if "query" in fingerprint
        ]
    elif isinstance(queries, Mapping):
        source = queries.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("query dataset spec must include a source")
        if "column" in queries and "columns" in queries:
            raise ValueError(
                "query dataset spec must set column or columns, not both"
            )
        columns = (
            queries["column"]
            if "column" in queries
            else queries.get("columns")
        )
        if isinstance(columns, str):
            columns = [columns]
        if (
            not isinstance(columns, Sequence)
            or isinstance(columns, (str, bytes))
            or not columns
        ):
            raise ValueError(
                "query dataset spec must include column or columns"
            )
        columns = [str(column) for column in columns]
        if any(not column.strip() for column in columns):
            raise ValueError(
                "query dataset columns must be non-empty strings"
            )

        rows = json.loads(
            resolve_cached_common_asset(
                source,
                source_fmt=queries.get("source_fmt", "json"),
                output_fmt="json",
                num_samples=queries.get("num_samples"),
                split=queries.get("split"),
                columns=columns,
            ).read_text(encoding="utf-8")
        )
        if not isinstance(rows, list):
            raise ValueError(
                "query dataset spec must resolve to a list of rows"
            )

        queries = []
        for row in rows:
            if isinstance(row, str):
                text = row.strip()
            elif isinstance(row, Mapping):
                text = "\n\n".join(
                    str(row[column]).strip()
                    for column in columns
                    if row.get(column)
                )
            else:
                text = ""
            if text:
                queries.append(text)
        if not queries:
            raise ValueError("query dataset spec produced no queries")
    if not queries:
        raise ValueError("provide queries in the config or fingerprints")
    if not (
        isinstance(queries, Sequence)
        and not isinstance(queries, (str, bytes))
        and all(isinstance(query, str) for query in queries)
    ):
        raise ValueError("queries must be a list of strings or a dataset spec")
    queries = list(queries)

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

    inference = dict(config.get("inference", {}))
    inference.update(
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
    config["inference"] = inference
    if args.seed is not None:
        config["seed"] = args.seed

    output_root = (
        normalize_str_to_path(args.output_dir)
        if args.output_dir
        else fingerprints_path.parent / "verification"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    loaded = load_model_and_tokenizer(model_config, role="verification")
    for index, total, run_config, _ in iter_grid_configs(
        config,
        include_prefixes=[
            ("verifier", "params"),
            ("inference",),
        ],
    ):
        if args.skip_existing:
            existing = None
            for run_dir in output_root.iterdir():
                saved_config_path = run_dir / "config.yaml"
                result_path = run_dir / "result.yaml"
                if (
                    not saved_config_path.is_file()
                    or not result_path.is_file()
                ):
                    continue
                result = load_yaml(result_path)
                if (
                    load_yaml(saved_config_path) == run_config
                    and result.get("model") == model_config
                    and result.get("fingerprints_path") == str(fingerprints_path)
                ):
                    existing = run_dir
                    break
            if existing is not None:
                print(f"[{index}/{total}] Reusing verification: {existing}")
                continue

        seed = run_config.get("seed", 42)
        set_seeds(seed)

        inference = dict(run_config.get("inference", {}))
        measurement_args = {
            name: inference.pop(name)
            for name in (
                "max_new_tokens",
                "batch_size",
                "apply_chat_template",
                "system_prompt",
            )
            if name in inference
        }
        score, metadata = measure_verification_score(
            model=loaded["model"],
            tokenizer=loaded["tokenizer"],
            queries=queries,
            verification_config=run_config,
            fingerprints=fingerprints,
            generation_params=inference,
            **measurement_args,
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
