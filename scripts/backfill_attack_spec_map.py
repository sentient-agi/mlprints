#!/usr/bin/env python3
"""
Utility script to backfill attack_idx mapping files for completed util_results.

For each fingerprint hash directory under experiments/models/*/, this script
looks for util_results/*.json, infers which attack spec produced each result,
and writes util_results/attack_specs.json so future runs can resume safely.
"""

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill attack_idx maps from util_results")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/models"),
        help="Root directory containing fingerprinted model runs",
    )
    parser.add_argument(
        "--util-script",
        type=Path,
        default=Path("scripts/util_with_fp_models.py"),
        help="Path to util_with_fp_models.py (used to read attack_specs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be written without touching files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite attack_specs.json if it already exists",
    )
    return parser.parse_args()


def load_attack_specs(util_script_path: Path) -> List[Dict[str, Any]]:
    source = util_script_path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "attack_specs":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"Could not locate attack_specs in {util_script_path}")


def attack_name_from_spec(spec: Dict[str, Any]) -> str:
    spec_type = spec["type"]
    if spec_type == "baseline":
        return "baseline"
    if spec_type == "logit":
        return spec["name"]
    if spec_type == "rephrase":
        return spec.get("name", "RephraseAttackedModel")
    if spec_type == "lookahead":
        return spec.get("name", "LookaheadAttackedModel")
    raise ValueError(f"Unknown attack type: {spec_type}")


def canonicalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    canonical = {"type": spec["type"]}
    if "name" in spec:
        canonical["name"] = spec["name"]
    if "kwargs" in spec:
        canonical["kwargs"] = spec["kwargs"]
    if "rephraser_model_id" in spec:
        canonical["rephraser_model_id"] = spec["rephraser_model_id"]
    return canonical


def stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)
    return str(value)


def normalize_spec_config(spec: Dict[str, Any]) -> Dict[str, str]:
    if spec["type"] == "rephrase" and "rephraser_model_id" in spec:
        return {"rephraser_model_id": stringify_value(spec["rephraser_model_id"])}
    kwargs = spec.get("kwargs", {})
    return {str(k): stringify_value(v) for k, v in kwargs.items()}


def normalize_result_config(config: Any) -> Dict[str, str]:
    if not isinstance(config, dict):
        return {}
    normalized = {}
    for key, value in config.items():
        if key == "tokenizer":
            continue
        normalized[str(key)] = stringify_value(value)
    return normalized


@dataclass
class SpecRecord:
    attack_name: str
    canonical_key: str
    normalized_config: Dict[str, str]


def build_spec_lookup(attack_specs: List[Dict[str, Any]]) -> Dict[str, List[SpecRecord]]:
    lookup: Dict[str, List[SpecRecord]] = {}
    for spec in attack_specs:
        attack_name = attack_name_from_spec(spec)
        canonical_key = json.dumps(canonicalize_spec(spec), sort_keys=True)
        normalized_config = normalize_spec_config(spec)
        lookup.setdefault(attack_name, []).append(
            SpecRecord(
                attack_name=attack_name,
                canonical_key=canonical_key,
                normalized_config=normalized_config,
            )
        )
    return lookup


def find_spec_record(
    attack_name: str, normalized_config: Dict[str, str], lookup: Dict[str, List[SpecRecord]]
) -> Optional[SpecRecord]:
    candidates = lookup.get(attack_name, [])
    for record in candidates:
        if record.normalized_config == normalized_config:
            return record
    return None


def discover_util_result_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for fp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for hash_dir in sorted(p for p in fp_dir.iterdir() if p.is_dir()):
            util_dir = hash_dir / "util_results"
            if util_dir.is_dir():
                yield util_dir


def collect_mapping_for_dir(util_dir: Path, lookup: Dict[str, List[SpecRecord]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    json_files = sorted(
        f for f in util_dir.glob("*.json") if f.name != "attack_specs.json" and f.is_file()
    )
    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError as err:
            print(f"[WARN] Skipping {json_file}: invalid JSON ({err})")
            continue
        attack_idx = data.get("attack_idx")
        attack_name = data.get("attack_name")
        attack_config = normalize_result_config(data.get("attack_config"))
        if attack_idx is None or attack_name is None:
            print(f"[WARN] Missing attack_idx/name in {json_file}, skipping")
            continue
        record = find_spec_record(str(attack_name), attack_config, lookup)
        if record is None:
            print(
                f"[WARN] Could not match attack spec for {json_file} "
                f"(name={attack_name}, config={attack_config})"
            )
            continue
        key = record.canonical_key
        attack_idx = int(attack_idx)
        if key in mapping and mapping[key] != attack_idx:
            print(
                f"[WARN] Conflicting attack_idx for spec {key} in {util_dir}: "
                f"{mapping[key]} vs {attack_idx}. Keeping existing value."
            )
            continue
        mapping[key] = attack_idx
    return mapping


def write_mapping(util_dir: Path, mapping: Dict[str, int], dry_run: bool):
    if not mapping:
        print(f"[INFO] No attack specs inferred for {util_dir}, skipping")
        return
    ordered_keys = [key for key, _ in sorted(mapping.items(), key=lambda item: item[1])]
    ordered_mapping = {key: mapping[key] for key in ordered_keys}
    output_path = util_dir / "attack_specs.json"
    if dry_run:
        print(f"[DRY-RUN] Would write {len(mapping)} specs to {output_path}")
        return
    output_path.write_text(json.dumps(ordered_mapping, indent=2))
    print(f"[INFO] Wrote {len(mapping)} specs to {output_path}")


def main():
    args = parse_args()
    attack_specs = load_attack_specs(args.util_script)
    spec_lookup = build_spec_lookup(attack_specs)
    util_dirs = list(discover_util_result_dirs(args.root))
    if not util_dirs:
        print(f"[INFO] No util_results directories found under {args.root}")
        return
    for util_dir in util_dirs:
        mapping_path = util_dir / "attack_specs.json"
        if mapping_path.exists() and not args.force:
            print(f"[INFO] Mapping already exists for {util_dir}, skipping (use --force to overwrite)")
            continue
        mapping = collect_mapping_for_dir(util_dir, spec_lookup)
        write_mapping(util_dir, mapping, args.dry_run)


if __name__ == "__main__":
    main()

