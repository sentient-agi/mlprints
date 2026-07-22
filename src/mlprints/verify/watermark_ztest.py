"""Offline KGW-style watermark z-test verifier."""

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import torch

from mlprints.common.constants import MAX_INT64


def verify_watermark_ztest(
    queries: Sequence[str],
    responses: Sequence[str],
    *,
    secret_key: int,
    greenlist_device: str,
    gamma: float = 0.25,
    context_width: int = 1,
    vocab_size: int | None = None,
    response_toks: Sequence[Sequence[int]] | None = None,
    tokenizer: Any | None = None,
    special_token_ids: Sequence[int] | None = None,
    alpha: float | None = 1e-3,
    exclude_special_tokens: bool = True,
    generation_params_used: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Run a KGW-style z-test over offline responses."""

    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    if len(queries) != len(responses):
        raise ValueError(
            f"verifier inputs must have the same length "
            f"(queries={len(queries)}, responses={len(responses)})"
        )
    if not (0 <= secret_key < 1 << 64):
        raise ValueError("secret_key must be an unsigned 64-bit integer")
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must be in (0, 1)")
    if context_width <= 0:
        raise ValueError("context_width must be > 0")

    greenlist_device = str(greenlist_device).strip().lower()
    if greenlist_device == "auto":
        raise ValueError("greenlist_device must match the concrete generation device, not 'auto'")
    if greenlist_device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("greenlist_device is cuda*, but CUDA is not available")
    greenlist_device = torch.device(greenlist_device)

    if vocab_size is None:
        if tokenizer is None:
            raise ValueError("vocab_size must be provided when tokenizer cannot supply it")
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    if vocab_size <= 1:
        raise ValueError("vocab_size must be > 1")

    if not exclude_special_tokens:
        excluded = set()
    elif special_token_ids is not None:
        excluded = {int(tok) for tok in special_token_ids}
    else:
        excluded = {int(tok) for tok in (getattr(tokenizer, "all_special_ids", []) or [])}
    allowed_ids = torch.tensor(
        [token_id for token_id in range(vocab_size) if token_id not in excluded],
        dtype=torch.long,
        device=greenlist_device,
    )
    if allowed_ids.numel() <= 1:
        raise ValueError("At least two allowed tokens are required")

    if response_toks is None:
        if tokenizer is None:
            raise ValueError("watermark_ztest needs response_toks or a tokenizer")
        token_lists = [[int(tok) for tok in tokenizer.encode(response, add_special_tokens=False)] for response in responses]
    else:
        if not isinstance(response_toks, Sequence) or isinstance(response_toks, (str, bytes)):
            raise TypeError("response_toks must be a sequence of token id sequences")
        token_lists = []
        for idx, toks in enumerate(response_toks):
            if not isinstance(toks, Sequence) or isinstance(toks, (str, bytes)):
                raise TypeError(f"response_toks[{idx}] must be a token id sequence")
            token_lists.append([int(tok) for tok in toks])
        if len(token_lists) != len(responses):
            raise ValueError(
                f"verifier inputs must have the same length (responses={len(responses)}, response_toks={len(token_lists)})"
            )

    ctx_to_tokens = {}
    total_reply_tokens = 0
    for toks in token_lists:
        total_reply_tokens += len(toks)
        for idx in range(context_width, len(toks)):
            token_id = int(toks[idx])
            if not 0 <= token_id < vocab_size:
                raise ValueError(f"response token id {token_id} is outside the vocabulary")
            if token_id in excluded:
                continue
            context = tuple(int(tok) for tok in toks[idx - context_width : idx])
            ctx_to_tokens.setdefault(context, set()).add(token_id)

    n = sum(len(tokens) for tokens in ctx_to_tokens.values())
    if n == 0:
        raise ValueError("No token-context pairs available for watermark_ztest")

    allowed_numel = int(allowed_ids.numel())
    green_k = max(1, round(gamma * allowed_numel))
    gamma_effective = green_k / allowed_numel
    secret_key_bytes = secret_key.to_bytes(8, "little", signed=False)
    generator = torch.Generator(device=greenlist_device)
    hits = 0
    n_eff = 0.0
    for context, token_set in ctx_to_tokens.items():
        num_context_tokens = len(token_set)
        n_eff += num_context_tokens * (allowed_numel - num_context_tokens) / (allowed_numel - 1)
        buffer = b"".join(token_id.to_bytes(4, "little", signed=False) for token_id in context)
        seed = int.from_bytes(
            hashlib.sha256(buffer + secret_key_bytes).digest()[:8],
            "little",
            signed=False,
        ) % MAX_INT64
        generator.manual_seed(seed)
        permutation = allowed_ids[
            torch.randperm(allowed_numel, generator=generator, device=greenlist_device)
        ]
        greenlist = set(permutation[:green_k].cpu().tolist())
        hits += sum(token_id in greenlist for token_id in token_set)

    beta = math.sqrt(n_eff / n)
    denom = beta * math.sqrt(float(n) * gamma_effective * (1.0 - gamma_effective))
    if denom == 0.0:
        raise ValueError("Degenerate denominator in watermark_ztest")
    z_score = (float(hits) - float(n) * gamma_effective) / denom
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    verification_score = float(0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0))))
    verification_metadata = {
        "verifier": "watermark_ztest",
        "verification_score": verification_score,
        "num_samples": int(len(responses)),
        "n_unique_pairs": int(n),
        "n_eff": float(n_eff),
        "beta": float(beta),
        "total_reply_tokens": int(total_reply_tokens),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "num_fp": int(len(responses)),
        "watermark_params": {
            "gamma": float(gamma),
            "gamma_effective": gamma_effective,
            "context_width": int(context_width),
            "secret_key_provided": True,
            "exclude_special_tokens": bool(exclude_special_tokens),
            "greenlist_device": str(greenlist_device),
        },
        "generation_params_used": generation_params_used,
    }
    if alpha is not None:
        alpha = float(alpha)
        verification_metadata["alpha"] = alpha
        verification_metadata["fingerprinted"] = bool(p_value < alpha)

    return verification_score, verification_metadata
