"""Token-ID batching helpers and generation backends for utility eval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, TypeVar

import torch
from transformers import ContinuousBatchingConfig, GenerationConfig

from mlprints.common.utils import get_model_device


DEFAULT_LENGTH_BUCKETS = (128, 256, 512, 1024, 2048, 4096)
REFERENCE_BUCKET = 512
GENERATION_BACKENDS = ("auto", "generate", "continuous")
DEFAULT_GENERATION_BACKEND = "auto"
DEFAULT_PREFIX_CACHING = True
DEFAULT_WARMUP_GENERATION = True
DEFAULT_PERSISTENT_MANAGER = True
DEFAULT_CONTINUOUS_COMPILE_LEVEL = 0
DEFAULT_CONTINUOUS_USE_CUDA_GRAPH = None

_WARMUP_KEYS: set[tuple[int, int, int]] = set()


class HasInputIds(Protocol):
    input_ids: Sequence[int]


BatchItem = TypeVar("BatchItem", bound=HasInputIds)


def uses_static_kv_cache(model: Any) -> bool:
    """Whether the model is configured for a static KV cache."""
    config = getattr(model, "generation_config", None)
    return getattr(config, "cache_implementation", None) == "static"


def benefits_from_generate_warmup(model: Any) -> bool:
    """Whether fixed-shape warmup can populate a static or compiled path."""
    return uses_static_kv_cache(model) or hasattr(model, "_orig_mod")


def normalize_continuous_batching_model_config(model: Any) -> None:
    """Disable a sliding window that cannot truncate the valid context."""
    config = getattr(model, "config", None)
    if config is None:
        return
    sliding_window = getattr(config, "sliding_window", None)
    max_positions = getattr(config, "max_position_embeddings", None)
    if (
        sliding_window is not None
        and max_positions is not None
        and sliding_window >= max_positions
    ):
        config.sliding_window = None


def resolve_generation_backend(
    model: Any,
    requested: str | None = DEFAULT_GENERATION_BACKEND,
) -> str:
    """Resolve ``auto`` to ``generate`` or ``continuous``; reject mixed backends."""
    requested = "auto" if requested is None else str(requested)
    if requested not in GENERATION_BACKENDS:
        raise ValueError(
            "generation_backend must be one of: "
            f"{', '.join(GENERATION_BACKENDS)}"
        )
    static_kv = uses_static_kv_cache(model)
    if requested == "continuous":
        if static_kv:
            raise ValueError(
                "static KV cache and paged continuous batching cannot be "
                "enabled together"
            )
        if hasattr(model, "_orig_mod"):
            raise ValueError(
                "whole-model torch.compile and continuous batching cannot be "
                "enabled together"
            )
        if not callable(getattr(model, "generate_batch", None)):
            raise TypeError("continuous backend requires model.generate_batch()")
        return "continuous"
    if requested == "generate":
        return "generate"
    if (
        static_kv
        or hasattr(model, "_orig_mod")
        or not callable(getattr(model, "generate_batch", None))
    ):
        return "generate"
    device = get_model_device(model)
    if device.type == "cuda":
        return "continuous"
    return "generate"


def assign_length_bucket(
    length: int,
    buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS,
) -> int:
    """Return the smallest bucket that fits ``length``, else ``length`` itself."""
    if length < 1:
        raise ValueError("sequence length must be >= 1")
    for bucket in buckets:
        if length <= bucket:
            return int(bucket)
    return int(length)


def batch_size_for_bucket(
    bucket: int,
    batch_size: int,
    buckets: Sequence[int] = DEFAULT_LENGTH_BUCKETS,
    overrides: Mapping[int, int] | None = None,
) -> int:
    """Smaller batches for longer prompt buckets; ``batch_size`` is the max."""
    if overrides and bucket in overrides:
        return max(1, int(overrides[bucket]))
    reference = REFERENCE_BUCKET if REFERENCE_BUCKET in buckets else (
        buckets[len(buckets) // 2] if buckets else REFERENCE_BUCKET
    )
    if bucket <= reference:
        return max(1, int(batch_size))
    return max(1, (int(batch_size) * int(reference)) // int(bucket))


def as_token_id_lists(input_ids: Any) -> list[list[int]]:
    """Normalize tensors or ragged sequences to ``list[list[int]]``."""
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError("input_ids tensor must be 1D or 2D")
        return [[int(token) for token in row] for row in input_ids.tolist()]
    if isinstance(input_ids, Sequence) and not isinstance(input_ids, (str, bytes)):
        if not input_ids:
            raise ValueError("input_ids must be non-empty")
        if isinstance(input_ids[0], int):
            return [[int(token) for token in input_ids]]
        return [[int(token) for token in sequence] for sequence in input_ids]
    raise TypeError("input_ids must be a tensor or a sequence of token IDs")


def unpadded_token_id_lists(
    input_ids: Any,
    attention_mask: Any | None = None,
) -> list[list[int]]:
    """Drop left/right padding using ``attention_mask`` when provided."""
    sequences = as_token_id_lists(input_ids)
    if attention_mask is None:
        return sequences
    masks = as_token_id_lists(attention_mask)
    if len(masks) != len(sequences):
        raise ValueError("attention_mask batch size must match input_ids")
    stripped = []
    for sequence, mask in zip(sequences, masks):
        tokens = [token for token, keep in zip(sequence, mask) if keep]
        if not tokens:
            raise ValueError("each sequence must contain at least one non-pad token")
        stripped.append(tokens)
    return stripped


def left_pad_token_ids(
    sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad token IDs to ``length`` or the batch max length."""
    if not sequences:
        raise ValueError("input_ids must be non-empty")
    token_lists = [[int(token) for token in sequence] for sequence in sequences]
    if any(len(sequence) == 0 for sequence in token_lists):
        raise ValueError("each sequence must be non-empty")
    max_len = max(len(sequence) for sequence in token_lists)
    target = max_len if length is None else int(length)
    if max_len > target:
        raise ValueError(
            f"sequence length {max_len} exceeds pad length {target}"
        )
    padded_ids = []
    masks = []
    for sequence in token_lists:
        pad = target - len(sequence)
        padded_ids.append([pad_token_id] * pad + sequence)
        masks.append([0] * pad + [1] * len(sequence))
    return (
        torch.tensor(padded_ids, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
    )


def iter_generate_batches(
    group_items: Sequence[tuple[int, BatchItem]],
    *,
    backend: str,
    batch_size: int,
    buckets: Sequence[int] | None,
    bucket_batch_sizes: Mapping[int, int] | None = None,
) -> Iterator[tuple[list[int], list[BatchItem], int | None]]:
    """Yield ``(indices, datas, pad_to_length)`` micro-batches."""
    items = list(group_items)
    if not items:
        return
    if backend == "continuous" or not buckets:
        step = len(items) if backend == "continuous" else max(1, int(batch_size))
        for start in range(0, len(items), step):
            batch = items[start:start + step]
            yield (
                [index for index, _ in batch],
                [data for _, data in batch],
                None,
            )
        return

    by_bucket: dict[int, list[tuple[int, BatchItem]]] = defaultdict(list)
    for index, data in items:
        bucket = assign_length_bucket(len(data.input_ids), buckets)
        by_bucket[bucket].append((index, data))

    for bucket in sorted(by_bucket):
        bucket_items = by_bucket[bucket]
        bucket_batch_size = batch_size_for_bucket(
            bucket,
            batch_size,
            buckets,
            bucket_batch_sizes,
        )
        for start in range(0, len(bucket_items), bucket_batch_size):
            batch = bucket_items[start:start + bucket_batch_size]
            yield (
                [index for index, _ in batch],
                [data for _, data in batch],
                bucket,
            )


def warmup_generate_shapes(
    model: Any,
    tokenizer: Any,
    shapes: Sequence[tuple[int, int]],
) -> None:
    """Run dummy ``model.generate`` calls so compiled graphs exist before eval."""
    device = get_model_device(model)
    if device.type != "cuda" or not shapes:
        return
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    model.eval()
    with torch.inference_mode():
        for batch_size, prompt_len in shapes:
            key = (id(model), int(batch_size), int(prompt_len))
            if key in _WARMUP_KEYS:
                continue
            input_ids = torch.full(
                (int(batch_size), int(prompt_len)),
                int(pad_token_id),
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            attention_mask[:, -1] = 1
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=int(pad_token_id),
            )
            _WARMUP_KEYS.add(key)


def generate_continuous(
    model: Any,
    tokenizer: Any,
    sequences: Sequence[Sequence[int]],
    *,
    gen_kwargs: dict[str, Any],
    generate_overrides: dict[str, Any],
    prefix_caching: bool = True,
    persistent_manager: bool = True,
    warmup: bool = True,
    compile_level: int = 0,
    use_cuda_graph: bool | tuple[bool, bool] | None = (
        DEFAULT_CONTINUOUS_USE_CUDA_GRAPH
    ),
    continuous_batching_config: dict[str, Any] | None = None,
    skip_special_tokens: bool = True,
) -> list[str]:
    """Generate with Transformers continuous batching and paged KV."""
    if not callable(getattr(model, "generate_batch", None)):
        raise TypeError("continuous backend requires model.generate_batch()")
    if uses_static_kv_cache(model):
        raise ValueError(
            "static KV cache and paged continuous batching cannot be "
            "enabled together"
        )
    if hasattr(model, "_orig_mod"):
        raise ValueError(
            "whole-model torch.compile and continuous batching cannot be "
            "enabled together"
        )

    token_lists = [[int(token) for token in sequence] for sequence in sequences]
    if not token_lists:
        return []
    normalize_continuous_batching_model_config(model)
    options = dict(continuous_batching_config or {})
    options.setdefault("allow_block_sharing", prefix_caching)
    options.setdefault("default_compile_level", compile_level)
    options.setdefault("use_cuda_graph", use_cuda_graph)
    if isinstance(options.get("use_cuda_graph"), list):
        options["use_cuda_graph"] = tuple(options["use_cuda_graph"])

    config_kwargs: dict[str, Any] = {}
    for key in (
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "num_return_sequences",
        "pad_token_id",
        "eos_token_id",
        "renormalize_logits",
        "repetition_penalty",
        "min_new_tokens",
    ):
        value = gen_kwargs.get(key)
        if value is not None:
            config_kwargs[key] = value
    for key, value in generate_overrides.items():
        if key == "logits_processor":
            continue
        if value is not None:
            config_kwargs[key] = value

    generation_config = GenerationConfig(**config_kwargs)
    continuous_config = ContinuousBatchingConfig(**options)
    outputs = model.generate_batch(
        token_lists,
        generation_config=generation_config,
        continuous_batching_config=continuous_config,
        persistent_manager=bool(persistent_manager),
        warmup=bool(warmup),
        progress_bar=False,
        max_new_tokens=gen_kwargs.get("max_new_tokens"),
    )
    expected_outputs = len(token_lists) * int(
        gen_kwargs.get("num_return_sequences") or 1
    )
    if len(outputs) != expected_outputs:
        raise RuntimeError(
            f"continuous batching returned {len(outputs)} outputs "
            f"for {expected_outputs} requests"
        )

    texts = []
    for output in outputs.values():
        error = getattr(output, "error", None)
        if error:
            raise RuntimeError(f"continuous batching failed: {error}")
        generated = list(getattr(output, "generated_tokens", None) or [])
        if not generated:
            texts.append("")
            continue
        texts.append(
            tokenizer.decode(
                generated,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        )
    return texts
