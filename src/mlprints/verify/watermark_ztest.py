"""Offline KGW-style watermark z-test verifier."""

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import torch

from mlprints.common.constants import MAX_INT64
from mlprints.inference.logits_processors import WatermarkProcessor
from mlprints.loading import load_tokenizer


def verify_watermark_ztest(
    queries: Sequence[str],
    responses: Sequence[str],
    *,
    secret_key: int,
    greenlist_device: str,
    tokenizer_id: str,
    gamma: float = 0.25,
    context_width: int = 1,
    vocab_size: int | None = None,
    special_token_ids: Sequence[int] | None = None,
    alpha: float | None = 1e-3,
    exclude_special_tokens: bool = True,
    generation_params_used: dict[str, Any] | None = None,
    trust_remote_code: bool = False,
    seeding_scheme: str = "sha256",
    concatenate_responses: bool = False,
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
    if seeding_scheme not in WatermarkProcessor.SEEDING_SCHEMES:
        raise ValueError(
            f"seeding_scheme must be one of {sorted(WatermarkProcessor.SEEDING_SCHEMES)}"
        )
    simple_1 = seeding_scheme == "simple_1"
    if simple_1 and context_width != 1:
        raise ValueError("simple_1 requires context_width=1")
    if not isinstance(tokenizer_id, str) or not tokenizer_id.strip():
        raise ValueError("tokenizer_id must be a non-empty string")

    tokenizer = load_tokenizer(
        tokenizer_id,
        trust_remote_code=trust_remote_code,
    )

    greenlist_device = str(greenlist_device).strip().lower()
    if greenlist_device == "auto":
        raise ValueError("greenlist_device must match the concrete generation device, not 'auto'")
    if greenlist_device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("greenlist_device is cuda*, but CUDA is not available")
    greenlist_device = torch.device(greenlist_device)

    if vocab_size is None:
        vocab_size = len(tokenizer)
    if vocab_size <= 1:
        raise ValueError("vocab_size must be > 1")

    if not exclude_special_tokens:
        excluded = set()
    elif special_token_ids is not None:
        excluded = set(special_token_ids)
    else:
        excluded = set(tokenizer.all_special_ids)
    allowed_ids = torch.tensor(
        [token_id for token_id in range(vocab_size) if token_id not in excluded],
        dtype=torch.long,
        device=greenlist_device,
    )
    if allowed_ids.numel() <= 1:
        raise ValueError("At least two allowed tokens are required")
    selection_ids = (
        torch.arange(vocab_size, dtype=torch.long, device=greenlist_device)
        if simple_1
        else allowed_ids
    )

    token_lists = [
        tokenizer.encode(response, add_special_tokens=False)
        for response in responses
    ]
    score_token_lists = (
        [[token_id for tokens in token_lists for token_id in tokens]]
        if concatenate_responses
        else token_lists
    )

    ctx_to_tokens = {}
    total_reply_tokens = sum(len(tokens) for tokens in token_lists)
    for toks in score_token_lists:
        for idx in range(context_width, len(toks)):
            token_id = toks[idx]
            if not 0 <= token_id < vocab_size:
                raise ValueError(f"response token id {token_id} is outside the vocabulary")
            if token_id in excluded:
                continue
            context = tuple(toks[idx - context_width : idx])
            if simple_1 and excluded.intersection(context):
                continue
            ctx_to_tokens.setdefault(context, set()).add(token_id)

    n = sum(len(tokens) for tokens in ctx_to_tokens.values())
    if n == 0:
        raise ValueError("No token-context pairs available for watermark_ztest")

    allowed_numel = allowed_ids.numel()
    green_k = int(gamma * vocab_size) if simple_1 else round(gamma * allowed_numel)
    gamma_effective = (
        gamma if simple_1 else green_k / allowed_numel
    )
    hits = 0
    n_eff = 0.0
    generator = torch.Generator(device=greenlist_device)
    key_bytes = None if simple_1 else secret_key.to_bytes(8, "little")
    for context, token_set in ctx_to_tokens.items():
        num_context_tokens = len(token_set)
        n_eff += num_context_tokens * (allowed_numel - num_context_tokens) / (allowed_numel - 1)
        if simple_1:
            seed = (secret_key * sum(context)) % (2**64 - 1)
        else:
            buffer = b"".join(
                token_id.to_bytes(4, "little")
                for token_id in context
            )
            seed = int.from_bytes(
                hashlib.sha256(buffer + key_bytes).digest()[:8],
                "little",
            ) % MAX_INT64
        generator.manual_seed(seed)
        greenlist = set(selection_ids[
            torch.randperm(
                selection_ids.numel(),
                generator=generator,
                device=greenlist_device,
            )[:green_k]
        ].cpu().tolist())
        greenlist.difference_update(excluded)
        hits += sum(token_id in greenlist for token_id in token_set)

    beta = math.sqrt(n_eff / n)
    denom = beta * math.sqrt(n * gamma_effective * (1.0 - gamma_effective))
    if denom == 0.0:
        raise ValueError("Degenerate denominator in watermark_ztest")
    z_score = (hits - n * gamma_effective) / denom
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    verification_score = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    verification_metadata = {
        "verifier": "watermark_ztest",
        "verification_score": verification_score,
        "num_samples": len(responses),
        "n_unique_pairs": n,
        "n_eff": n_eff,
        "beta": beta,
        "total_reply_tokens": total_reply_tokens,
        "z_score": z_score,
        "p_value": p_value,
        "num_fp": len(responses),
        "watermark_params": {
            "gamma": gamma,
            "gamma_effective": gamma_effective,
            "context_width": context_width,
            "secret_key_provided": True,
            "exclude_special_tokens": exclude_special_tokens,
            "greenlist_device": str(greenlist_device),
            "seeding_scheme": seeding_scheme,
            "concatenate_responses": concatenate_responses,
        },
        "generation_params_used": generation_params_used,
    }
    if alpha is not None:
        verification_metadata["alpha"] = alpha
        verification_metadata["fingerprinted"] = p_value < alpha

    return verification_score, verification_metadata
