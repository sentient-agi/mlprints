#!/usr/bin/env python3
"""
Quick script to rename:
  experiments/models/perinucleus/*/final_model -> experiments/models/perinucleus/*/checkpoint-final

Usage:
  python scripts/rename_final_model_dirs.py
  python scripts/rename_final_model_dirs.py --root /abs/path/to/experiments/models/perinucleus
  python scripts/rename_final_model_dirs.py --dry-run
"""

# from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List


def rename_dirs(root: Path, dry_run: bool = False) -> List[str]:
    messages: List[str] = []
    if not root.exists() or not root.is_dir():
        messages.append(f"Root does not exist or is not a directory: {root}")
        return messages

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        src = run_dir / "final_model"
        dst = run_dir / "checkpoint-final"
        if src.exists() and src.is_dir():
            if dst.exists():
                messages.append(f"SKIP (exists): {dst}")
                continue
            messages.append(f"RENAME: {src} -> {dst}")
            if not dry_run:
                src.rename(dst)
        else:
            messages.append(f"MISS: {src} (not found)")
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename final_model dirs to checkpoint-final")
    parser.add_argument(
        "--root",
        default=str(Path("experiments/models/perinucleus_better")),
        help="Root directory containing per-run subdirectories",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print actions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    msgs = rename_dirs(root=root, dry_run=args.dry_run)
    for line in msgs:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


