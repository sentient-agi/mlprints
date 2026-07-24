import copy
import itertools
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch

from mlprints.common.utils import (
    normalize_str_to_path,
    get_timestamp_uuid,
    load_yaml,
    load_model,
    load_tokenizer,
)
from mlprints.common.distributed import (
    barrier_if_distributed,
    is_distributed,
    is_rank0,
)


GRID_WRAPPER_KEY = "grid"  # per-parameter list wrapper: i.e. {grid: [..]}


def get_experiment_dir(
    experiments_dir: str | None = None,
    experiment_name: str | None = None,
) -> Path:
    experiments_dir = experiments_dir or os.getenv("MLPRINTS_EXPERIMENTS_DIR")
    if experiments_dir is None:
        raise ValueError(
            "provide --experiments-dir or set MLPRINTS_EXPERIMENTS_DIR"
        )

    experiments_dir = normalize_str_to_path(experiments_dir)
    experiment_dir = experiments_dir / (experiment_name or get_timestamp_uuid())
    if is_rank0():
        experiment_dir.mkdir(parents=True, exist_ok=True)
    barrier_if_distributed()
    return experiment_dir


def get_trained_dir(fingerprints_dir: Path) -> Path:
    trained_dir = fingerprints_dir / "trained" / get_timestamp_uuid()
    if is_rank0():
        trained_dir.mkdir(parents=True, exist_ok=False)
    barrier_if_distributed()
    return trained_dir


def get_checkpoints_dir(trained_dir: Path) -> str:
    checkpoints_dir = trained_dir / "checkpoints"
    if is_rank0():
        checkpoints_dir.mkdir(parents=False, exist_ok=False)
    barrier_if_distributed()
    return str(checkpoints_dir)


def _resolve_device_map(device_map: Any) -> Any:
    if not is_distributed() or device_map is None:
        return device_map

    local_device = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    if isinstance(device_map, str) and device_map.startswith("cuda:"):
        return f"cuda:{local_device}"
    if isinstance(device_map, dict):
        return {
            key: local_device if isinstance(value, int) and value >= 0 else value
            for key, value in device_map.items()
        }
    return device_map


def load_model_and_tokenizer(
    model_config: dict[str, Any],
    role: str = "",
    is_train: bool = False,
) -> dict[str, Any]:
    path_or_model_id = model_config.get("model_id")
    if not path_or_model_id:
        raise ValueError(f"model_id is required for role {role!r}")

    path_or_tokenizer_id = model_config.get("tokenizer_id", path_or_model_id)
    device_map = _resolve_device_map(model_config.get("device_map"))
    dtype = model_config.get("dtype")
    trust_remote_code = model_config.get("trust_remote_code")
    attn_implementation = model_config.get("attn_implementation")

    if is_rank0():
        print(f"Loading {role} model: {path_or_model_id}")

    model_path = normalize_str_to_path(path_or_model_id)
    is_attacked = (model_path / "attack.yaml").exists()

    model = load_model(
        path_or_model_id=path_or_model_id,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        is_train=is_train,
    )
    if is_attacked:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise AttributeError("Attacked model must have tokenizer attribute")
    else:
        tokenizer = load_tokenizer(
            path_or_tokenizer_id,
            trust_remote_code=trust_remote_code,
        )

    if is_rank0():
        print(f"Loaded {role} model: {path_or_model_id} on {device_map}")

    return {"model": model, "tokenizer": tokenizer}


def iter_grid_configs(
    base_config: dict[str, Any],
    *,
    include_prefixes: list[tuple[Any, ...]] | None = None,
) -> Iterable[tuple[int, int, dict[str, Any], dict[str, Any]]]:
    """Expand explicit ``{grid: [...]}`` wrappers into resolved configs."""
    if not isinstance(base_config, dict):
        raise ValueError(f"Config must be a dict, got: {type(base_config).__name__}")

    def _is_included(path: tuple[Any, ...]) -> bool:
        return include_prefixes is None or any(
            path[: len(prefix)] == prefix
            for prefix in include_prefixes
        )

    def _path_to_str(path: tuple[Any, ...]) -> str:
        out = ""
        for p in path:
            if isinstance(p, int):
                out += f"[{p}]"
            else:
                out += ("" if out == "" else ".") + str(p)
        return out

    def _find_grid_wrappers_inline(
        obj: Any,
        *,
        path: tuple[Any, ...] = (),
    ) -> list[tuple[tuple[Any, ...], list[Any]]]:
        found: list[tuple[tuple[Any, ...], list[Any]]] = []

        if (
            isinstance(obj, dict)
            and set(obj.keys()) == {GRID_WRAPPER_KEY}
            and isinstance(obj.get(GRID_WRAPPER_KEY), list)
        ):
            values = obj[GRID_WRAPPER_KEY]
            if not values:
                raise ValueError(
                    f"Grid wrapper at '{_path_to_str(path)}' must be non-empty"
                )
            if _is_included(path):
                found.append((path, values))
            return found

        if isinstance(obj, dict):
            for k, v in obj.items():
                found.extend(_find_grid_wrappers_inline(v, path=path + (k,)))
            return found

        if isinstance(obj, list):
            for i, v in enumerate(obj):
                found.extend(_find_grid_wrappers_inline(v, path=path + (i,)))
            return found

        return found

    wrapper_axes = _find_grid_wrappers_inline(base_config)
    if not wrapper_axes:
        yield 1, 1, copy.deepcopy(base_config), {}
        return

    value_axes = [values for _path, values in wrapper_axes]
    total = math.prod(len(values) for values in value_axes)
    for idx, combo in enumerate(itertools.product(*value_axes), start=1):
        cfg = copy.deepcopy(base_config)
        selected: dict[str, Any] = {}
        for (path, _values), chosen in zip(wrapper_axes, combo):
            cur = cfg
            for p in path[:-1]:
                cur = cur[p]
            cur[path[-1]] = chosen
            selected[_path_to_str(path)] = chosen

        yield idx, total, cfg, selected


def _normalize_config_for_comparison(
    config: dict[str, Any],
    section: str,
) -> dict[str, Any]:
    if section not in {"params", "training"}:
        raise ValueError(f"unknown config section: {section}")

    algo = config.get("algo", {})
    values = dict(algo.get(section, {}))
    if section == "training":
        values.pop("models", None)
    return {"algo_name": algo.get("name"), section: values}


def configs_match(
    config1: dict[str, Any],
    config2: dict[str, Any],
    section: str = "params",
) -> bool:
    return _normalize_config_for_comparison(
        config1,
        section,
    ) == _normalize_config_for_comparison(config2, section)


def _find_existing_dir(
    root: Path,
    config: dict[str, Any],
    section: str,
) -> Path | None:
    if not root.is_dir():
        return None

    for run_dir in sorted(root.iterdir()):
        config_path = run_dir / "config.yaml"
        if not run_dir.is_dir() or not config_path.is_file():
            continue
        try:
            if configs_match(config, load_yaml(config_path), section):
                return run_dir
        except Exception:
            continue
    return None


def find_existing_fingerprint_dir(
    experiment_dir: Path,
    algo_name: str,
    config: dict[str, Any],
) -> Path | None:
    return _find_existing_dir(
        experiment_dir / "fingerprints" / algo_name,
        config,
        "params",
    )


def find_existing_trained_dir(
    fingerprints_dir: Path,
    config: dict[str, Any],
) -> Path | None:
    return _find_existing_dir(
        fingerprints_dir / "trained",
        config,
        "training",
    )


def looks_like_hf_model_id(path_or_model_id: str) -> bool:
    if path_or_model_id.startswith(("/", "./", "../", "~")):
        return False
    return not Path(path_or_model_id).exists()


def _checkpoint_number(checkpoint_dir: Path) -> int | None:
    suffix = checkpoint_dir.name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else None


def _candidate_checkpoint_dirs(path: Path) -> list[Path]:
    search_dirs = []
    checkpoints_dir = path / "checkpoints"
    if checkpoints_dir.is_dir():
        search_dirs.append(checkpoints_dir)
    search_dirs.append(path)

    for search_dir in search_dirs:
        checkpoint_dirs = [
            child
            for child in search_dir.iterdir()
            if child.is_dir() and child.name.startswith("checkpoint-")
        ]
        if checkpoint_dirs:
            return checkpoint_dirs

    return []


def find_checkpoint_path(path: str | Path, checkpoint: str | int = "latest") -> Path | None:
    """
    Find a checkpoint directory under a local path.

    `checkpoint` may be "latest", "final", an integer step, or a checkpoint
    directory name like "checkpoint-500".
    """
    path = Path(path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        return None

    if path.name.startswith("checkpoint-"):
        return path

    checkpoint_dirs = _candidate_checkpoint_dirs(path)
    if not checkpoint_dirs:
        return None

    if isinstance(checkpoint, int) or str(checkpoint).isdigit():
        checkpoint_name = f"checkpoint-{checkpoint}"
        matches = [ckpt for ckpt in checkpoint_dirs if ckpt.name == checkpoint_name]
    elif checkpoint == "final":
        matches = [ckpt for ckpt in checkpoint_dirs if ckpt.name == "checkpoint-final"]
    elif str(checkpoint).startswith("checkpoint-"):
        matches = [ckpt for ckpt in checkpoint_dirs if ckpt.name == checkpoint]
    elif checkpoint == "latest":
        final_matches = [ckpt for ckpt in checkpoint_dirs if ckpt.name == "checkpoint-final"]
        if final_matches:
            return final_matches[0]

        numbered = [
            (number, ckpt)
            for ckpt in checkpoint_dirs
            if (number := _checkpoint_number(ckpt)) is not None
        ]
        return max(numbered, key=lambda item: item[0])[1] if numbered else None
    else:
        raise ValueError(
            "checkpoint must be 'latest', 'final', an integer step, "
            "or a checkpoint directory name"
        )

    if not matches:
        return None
    return matches[0]


def resolve_checkpoint_path(path_or_model_id: str, checkpoint: str | int = "latest") -> str:
    """
    Resolve a local training run/checkpoints directory to a concrete checkpoint.

    Hugging Face model IDs, non-directories, and directories without checkpoint
    subdirectories are returned unchanged.
    """
    if looks_like_hf_model_id(path_or_model_id):
        return path_or_model_id

    path = Path(path_or_model_id).expanduser().resolve()
    checkpoint_path = find_checkpoint_path(path, checkpoint=checkpoint)
    return str(checkpoint_path) if checkpoint_path is not None else str(path)
