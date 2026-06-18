"""
Cache utilities for remote assets and HF datasets.

The source is either an HTTP(S) URL, a local file path, or a HF dataset id.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Optional
import yaml

from datasets import load_dataset
import requests

from mlprints.common.constants import (
    ATTACK_CACHE_DIR,
    COMMON_CACHE_DIR,
    FINGERPRINT_CACHE_DIR,
    MLPRINTS_HOME,
)


_ASSETS_CACHE_DIR_NAME = "assets"
_DATASET_FORMATS = ("text", "txt", "json", "jsonl", "csv", "tsv", "yaml")
_TEMP_FILE_SUFFIX = ".tmp"
_TIMEOUT = 20 # seconds
_URL_PREFIXES = ("http://", "https://")
_CACHE_NAMESPACES = {
    "attack": ATTACK_CACHE_DIR,
    "common": COMMON_CACHE_DIR,
    "fingerprint": FINGERPRINT_CACHE_DIR,
}


def get_cache_root(*, create: bool = True) -> Path:
    """Return the root cache directory (MLPRINTS_HOME)."""
    root_path = MLPRINTS_HOME.resolve()
    if create:
        root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def get_namespace_cache_dir(namespace: str, *, create: bool = True) -> Path:
    """Return the cache directory for a namespace such as fingerprint or attack."""
    namespace = (namespace or "").strip().lower()
    if namespace not in _CACHE_NAMESPACES:
        raise ValueError(
            f"namespace must be one of {sorted(_CACHE_NAMESPACES)}, got {namespace!r}"
        )

    dir_path = _CACHE_NAMESPACES[namespace].resolve()
    if create:
        dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_algo_cache_dir(
    algo_name: str,
    *,
    namespace: str,
    create: bool = True,
) -> Path:
    """Return the per-algorithm cache directory for a namespace."""
    algo_name = (algo_name or "").strip()
    if not algo_name:
        raise ValueError(f"algo_name {algo_name!r} was invalid or null")
    dir_path = (get_namespace_cache_dir(namespace, create=create) / algo_name).resolve()
    if create:
        dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_fingerprint_cache_dir(algo_name: str, *, create: bool = True) -> Path:
    """Return the per-fingerprint-algorithm cache directory."""
    return get_algo_cache_dir(algo_name, namespace="fingerprint", create=create)


def get_attack_cache_dir(algo_name: str, *, create: bool = True) -> Path:
    """Return the per-attack-algorithm cache directory."""
    return get_algo_cache_dir(algo_name, namespace="attack", create=create)


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text to a file by first writing it to temporary file and then renaming it to the target path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _TEMP_FILE_SUFFIX)
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write bytes to a file by first writing them to a temporary file and then renaming it to the target path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _TEMP_FILE_SUFFIX)
    tmp.write_bytes(data)
    tmp.replace(path)


def resolve_cached_asset(
    source: str | Path,
    *,
    namespace: str,
    algo_name: Optional[str] = None,
    source_fmt: str = "json",
    output_fmt: str = "json",
    num_samples: Optional[int] = None,
    split: Optional[str] = None,
    columns: Optional[list[str]] = None,
    encoding: str = "utf-8",
) -> Path:
    """
    Load data from URL/ local file / HF dataset id and cache it.
    Caching includes the asset itself and a json spec.
    """
    namespace = (namespace or "").strip().lower()
    source_fmt = (source_fmt or "").strip().lower()
    output_fmt = (output_fmt or "").strip().lower()
    if source_fmt == "txt":
        source_fmt = "text"
    if output_fmt == "txt":
        output_fmt = "text"
    if source_fmt not in _DATASET_FORMATS:
        raise ValueError(f"Unsupported source_fmt: {source_fmt!r}")
    if output_fmt not in _DATASET_FORMATS:
        raise ValueError(f"Unsupported output_fmt: {output_fmt!r}")

    extension = "txt" if output_fmt == "text" else output_fmt
    source_str = str(source).strip()
    source_path = Path(source_str).expanduser()

    is_url = source_str.startswith(_URL_PREFIXES)
    is_local = source_path.exists()
    is_hf_dataset = (not is_url) and (not is_local) # assumed to be a huggingface dataset if not a url or local file

    # 1) build cache paths and spec dict
    asset_dir = (
        get_algo_cache_dir(algo_name, namespace=namespace)
        if algo_name
        else get_namespace_cache_dir(namespace) / _ASSETS_CACHE_DIR_NAME
    )
    asset_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source_str.encode("utf-8")).hexdigest()
    asset_path = asset_dir / f"{source_hash}.{extension}"
    spec_path = asset_dir / f"{source_hash}.spec.json"
    spec_dict = {
        "source": source_str,
        "source_fmt": source_fmt,
        "output_fmt": output_fmt,
        "num_samples": num_samples,
        "encoding": encoding,
        "split": split,
        "columns": columns,
    }

    if asset_path.exists() and spec_path.exists():
        try:
            if json.loads(spec_path.read_text(encoding=encoding)) == spec_dict:
                return asset_path # prevents re-caching the same asset
        except (OSError, json.JSONDecodeError):
            pass

    # 2) load raw bytes via yaml or HF datasets loader and decode to format
    if source_fmt == "yaml" and not is_hf_dataset:
        if is_url:
            response = requests.get(source_str, timeout=_TIMEOUT)
            response.raise_for_status()
            raw = response.content
        else:
            p = source_path.resolve()
            if not p.exists():
                raise FileNotFoundError(f"Asset path not found: {p}")
            raw = p.read_bytes()
        text = raw.decode(encoding, errors="replace") # error tolerant
        data = yaml.safe_load(text)
        if num_samples is not None and isinstance(data, list):
            data = data[:num_samples]
    else:
        load_target = source_str
        load_kwargs = {
            "streaming": True,
            "cache_dir": str(asset_dir),
            "split": split,
        }

        if not is_hf_dataset:
            builder_map = {
                "text": "text",
                "json": "json",
                "jsonl": "json",
                "csv": "csv",
                "tsv": "csv",
            }
            load_target = builder_map[source_fmt]
            load_kwargs["data_files"] = source_str if is_url else str(source_path.resolve())
            if source_fmt == "tsv":
                load_kwargs["sep"] = "\t"

        dataset = load_dataset(load_target, **load_kwargs)

        # optional: truncate number of rows to `num_samples` if specified
        if num_samples is not None:
            rows = list(dataset.take(num_samples))
        else:
            rows = list(dataset)

        # optional: truncate columns to `columns` if specified
        if columns:
            if len(columns) == 1:
                col = columns[0]
                data = [row.get(col) if isinstance(row, dict) else None for row in rows]
            else:
                data = [{c: row.get(c) if isinstance(row, dict) else None for c in columns} for row in rows]
        elif source_fmt in {"text", "txt"} and not is_hf_dataset:
            data = [row.get("text") if isinstance(row, dict) else row for row in rows]
        else:
            data = rows

    # 3) serialize payload by target cache format and write to cache
    if output_fmt == "text":
        payload = ("\n".join(str(x) for x in data) + "\n") if isinstance(data, list) else str(data)
    elif output_fmt == "json":
        payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    elif output_fmt == "jsonl":
        if not isinstance(data, list):
            raise TypeError("JSONL expects list-like rows")
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in data) + "\n"
    elif output_fmt in {"csv", "tsv"}:
        if not isinstance(data, list):
            raise TypeError(f"{output_fmt.upper()} expects list-like rows")
        delimiter = "," if output_fmt == "csv" else "\t"
        buf = io.StringIO()
        if data and isinstance(data[0], dict):
            fieldnames = list(data[0].keys())
            writer = csv.DictWriter(buf, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)
        else:
            writer = csv.writer(buf, delimiter=delimiter)
            writer.writerows(data)
        payload = buf.getvalue()
    elif output_fmt == "yaml":
        payload = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)

    write_text_atomic(asset_path, payload, encoding=encoding)
    write_text_atomic(spec_path, json.dumps(spec_dict, ensure_ascii=False, indent=2) + "\n", encoding=encoding)

    return asset_path


def resolve_cached_fingerprint_asset(
    source: str | Path,
    *,
    algo_name: str,
    source_fmt: str = "json",
    output_fmt: str = "json",
    num_samples: Optional[int] = None,
    split: Optional[str] = None,
    columns: Optional[list[str]] = None,
    encoding: str = "utf-8",
) -> Path:
    """Resolve and cache an asset under a fingerprint algorithm cache."""
    return resolve_cached_asset(
        source,
        namespace="fingerprint",
        algo_name=algo_name,
        source_fmt=source_fmt,
        output_fmt=output_fmt,
        num_samples=num_samples,
        split=split,
        columns=columns,
        encoding=encoding,
    )


def resolve_cached_attack_asset(
    source: str | Path,
    *,
    algo_name: str,
    source_fmt: str = "json",
    output_fmt: str = "json",
    num_samples: Optional[int] = None,
    split: Optional[str] = None,
    columns: Optional[list[str]] = None,
    encoding: str = "utf-8",
) -> Path:
    """Resolve and cache an asset under an attack algorithm cache."""
    return resolve_cached_asset(
        source,
        namespace="attack",
        algo_name=algo_name,
        source_fmt=source_fmt,
        output_fmt=output_fmt,
        num_samples=num_samples,
        split=split,
        columns=columns,
        encoding=encoding,
    )


def resolve_cached_common_asset(
    source: str | Path,
    *,
    source_fmt: str = "json",
    output_fmt: str = "json",
    num_samples: Optional[int] = None,
    split: Optional[str] = None,
    columns: Optional[list[str]] = None,
    encoding: str = "utf-8",
) -> Path:
    """Resolve and cache a shared asset under the common cache."""
    return resolve_cached_asset(
        source,
        namespace="common",
        source_fmt=source_fmt,
        output_fmt=output_fmt,
        num_samples=num_samples,
        split=split,
        columns=columns,
        encoding=encoding,
    )
