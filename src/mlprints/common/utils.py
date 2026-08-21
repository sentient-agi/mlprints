"""
Common utilities for the MLprints library.
"""

import csv
from datetime import datetime
from importlib import util
from pathlib import Path
import random
import sys
from typing import Any
import uuid
import yaml

import numpy as np
import torch
import torch.nn.functional as F

from mlprints.common.constants import MASK_LOSS_ID, MAX_LENGTH_SENTINEL


# GENERAL UTILITIES

def get_timestamp_uuid(num_chars: int = 8) -> str:
    """
    Get a second-level timestamp hash (%Y%m%d-%H%M%S- + `num_chars` chars uuid).
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uuid_suffix = uuid.uuid4().hex[:num_chars]
    return f"{timestamp}-{uuid_suffix}"


def set_seeds(seed: int = 42) -> None:
    """
    Set seeds for random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_device(model: Any) -> torch.device:
    """
    Return the device where model inputs should be placed.
    """
    # for HF causal LMs, input IDs enter through the input embeddings
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if weight is not None:
            return weight.device

    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)

    try:
        return next(model.parameters()).device
    except StopIteration as e:
        raise ValueError("cannot determine device for model with no parameters") from e


def get_eos_token_ids(model: Any, tokenizer: Any) -> list[int]:
    """
    Return EOS token IDs from ``model.generation_config``, else the tokenizer.
    """
    generation_config = getattr(model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, (list, tuple, set)):
        return [int(token_id) for token_id in eos_token_id]
    return [int(eos_token_id)]


# PATH AND FILE UTILITIES

def normalize_str_to_path(*path_strs: str | Path) -> Path | tuple[Path, ...]:
    """
    Normalize one or more path strings or Path objects to Path objects.
    Users must explicitly use '.' if they want the current working directory.
    """
    normalized = []
    for raw_path in path_strs:
        if isinstance(raw_path, str) and not raw_path.strip():
            raise ValueError(
                "Path string cannot be empty. "
                "Use '.' explicitly if you mean the current working directory."
            )

        normalized.append(Path(raw_path).expanduser().resolve())

    return normalized[0] if len(normalized) == 1 else tuple(normalized)


def load_csv(path: Path) -> list[dict[str, str]]:
    """
    Load a CSV file and return its rows.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_yaml(path: Path) -> Any:
    """
    Load a YAML file and return its contents.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    """
    Save a dictionary to a YAML file.
    """
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_implementation(path):
    path = normalize_str_to_path(path)
    name = f"_mlprints_{path.stem}_{uuid.uuid5(uuid.NAMESPACE_URL, str(path)).hex}"
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# METADATA AND INSPECTION UTILITIES

def are_tokenizers_equal(tokenizers, raise_on_mismatch=False):
    """Return True if all tokenizers are the same; optionally raise on first mismatch."""
    if len(tokenizers) <= 1:
        return True
    first = tokenizers[0]
    for i, tokenizer in enumerate(tokenizers[1:], start=1):
        same = (
            first.get_vocab() == tokenizer.get_vocab()
            and first.get_added_vocab() == tokenizer.get_added_vocab()
            and first.special_tokens_map == tokenizer.special_tokens_map
            and getattr(first, "padding_side", None) == getattr(tokenizer, "padding_side", None)
            and getattr(first, "truncation_side", None) == getattr(tokenizer, "truncation_side", None)
            and getattr(first, "model_max_length", None) == getattr(tokenizer, "model_max_length", None)
        )
        if not same:
            if raise_on_mismatch:
                raise ValueError(f"tokenizer differs at index {i} (vs. tokenizer at index 0)")
            return False
    return True


def get_context_length_from_model(model: Any) -> int:
    """
    Return the model's max input length (in tokens) from its config.
    """
    _CTX_LENGTH_ATTRS = ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "max_sequence_length")
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("model has no config.")

    configs = (cfg, getattr(cfg, "text_config", None))
    for candidate in configs:
        if candidate is None:
            continue
        for attr in _CTX_LENGTH_ATTRS:
            value = getattr(candidate, attr, None)
            if type(value) is int and 0 < value < MAX_LENGTH_SENTINEL:
                return int(value)

    raise ValueError("context length not declared in model config")


def get_context_length_from_tokenizer(tokenizer: Any) -> int:
    """
    Return the tokenizer's advertised max length (tokens) from model_max_length.
    """
    _CTX_LENGTH_ATTR = "model_max_length"
    value = getattr(tokenizer, _CTX_LENGTH_ATTR, None)
    if type(value) is int and 0 < value < MAX_LENGTH_SENTINEL:
        return int(value)

    raise ValueError("tokenizer does not declare a valid model_max_length")


def get_model_id(model: Any | str, short: bool = False) -> str:
    """
    Get a model ID from either an ID/path string or a model object.

    If short is True, return only the last path/repo component.
    """
    if isinstance(model, str):
        return model.rsplit("/", 1)[-1] if short else model
    _MODEL_ID_ATTRS = ("name_or_path", "_name_or_path", "pretrained_model_name_or_path")
    for host in (getattr(model, "config", None), model):
        if host is None:
            continue
        for attr in _MODEL_ID_ATTRS:
            value = getattr(host, attr, None)
            if isinstance(value, str) and value.strip():
                return value.rsplit("/", 1)[-1] if short else value
    raise ValueError("no repo/path on this model")


# MACHINE LEARNING UTILITIES 

def compute_causal_lm_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute causal language modeling cross-entropy loss from logits and labels.

    It handles the standard causal LM setup where predictions are shifted to
    align with targets, and masks out loss on ignored tokens (MASK_LOSS_ID).
    """
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    vocab_size = shift_logits.size(-1)
    valid_counts = (shift_labels != MASK_LOSS_ID).sum(dim=-1).clamp_min(1)

    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=MASK_LOSS_ID,
    ).view(shift_labels.shape)

    return token_losses.sum(dim=-1) / valid_counts
