"""ADG bitstream z-test verifier."""

import math
from collections.abc import Sequence
from typing import Any

import torch

from mlprints.common.utils import get_eos_token_ids, get_model_device
from mlprints.inference.formatting import format_input
from mlprints.loading import load_model, load_tokenizer


def parse_adg_bitstream(bitstream: str | Sequence[int]) -> list[int]:
    """Parse a hex string (`0x…`) or 0/1 sequence into an ADG bitstream."""
    if isinstance(bitstream, str):
        if bitstream[:2].lower() == "0x":
            bitstream = "".join(
                f"{int(nibble, 16):04b}"
                for nibble in bitstream[2:]
            )
        bits = [int(bit) for bit in bitstream]
    else:
        bits = [int(bit) for bit in bitstream]
    if not bits or any(bit not in (0, 1) for bit in bits):
        raise ValueError("ADG bitstream must be a non-empty sequence of 0/1 bits")
    return bits


def decode_adg_token_bits(
    scores: torch.Tensor,
    token_id: int,
    *,
    temperature: float = 1.0,
    excluded_token_ids: Sequence[int] | None = None,
) -> list[int]:
    """Recover the ADG bits implied by one sampled token."""
    scores = scores.float()
    if excluded_token_ids:
        scores[list(excluded_token_ids)] = -torch.inf
    probs, token_ids = torch.softmax(scores / temperature, dim=-1).sort(
        descending=True
    )
    decoded_bits = []
    while probs[0] <= 0.5:
        num_groups = 2
        while 1 / (num_groups * 2) > probs[0]:
            num_groups *= 2

        groups = []
        mean = probs.new_tensor(1 / num_groups)
        for group_index in range(num_groups - 1):
            group_probs, group_ids = probs[:1], token_ids[:1]
            probs, token_ids = probs[1:], token_ids[1:]
            while group_probs.sum() < mean:
                delta = mean - group_probs.sum()
                index = (probs - delta).abs().argmin()
                if probs[index] - delta >= delta:
                    break
                group_probs = torch.cat(
                    (group_probs, probs[index:index + 1])
                )
                group_ids = torch.cat(
                    (group_ids, token_ids[index:index + 1])
                )
                keep = (
                    torch.arange(len(probs), device=probs.device)
                    != index
                )
                probs, token_ids = probs[keep], token_ids[keep]
            groups.append((group_probs, group_ids))
            mean = probs.sum() / (num_groups - group_index - 1)
        groups.append((probs, token_ids))

        num_bits = (len(groups) - 1).bit_length()
        group_index = next(
            index
            for index, (_, group_ids) in enumerate(groups)
            if (group_ids == token_id).any()
        )
        decoded_bits.extend(
            (group_index >> index) & 1
            for index in range(num_bits)
        )
        probs, token_ids = groups[group_index]
        probs = probs / probs.sum()
        probs, order = probs.sort(descending=True)
        token_ids = token_ids[order]
    return decoded_bits


def decode_adg_bitstream(
    stego_model,
    stego_tokenizer,
    prompt: str,
    response: str,
    *,
    temperature: float = 1.0,
    excluded_token_ids: Sequence[int] | None = None,
) -> list[int]:
    """Decode the ADG bitstream hidden in `response` under `prompt`."""
    if excluded_token_ids is None:
        eos_token_ids = get_eos_token_ids(stego_model, stego_tokenizer)
        excluded_token_ids = [
            token_id
            for token_id in stego_tokenizer.all_special_ids
            if token_id not in eos_token_ids
        ]
    encoded = stego_tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(get_model_device(stego_model))
    response_ids = stego_tokenizer.encode(response, add_special_tokens=False)
    if not response_ids:
        return []
    candidate_ids = torch.tensor(
        response_ids,
        device=encoded["input_ids"].device,
    )
    decoder_scores = stego_model(
        input_ids=torch.cat(
            (encoded["input_ids"], candidate_ids.unsqueeze(0)),
            dim=1,
        ),
        attention_mask=torch.cat(
            (
                encoded["attention_mask"],
                torch.ones_like(candidate_ids).unsqueeze(0),
            ),
            dim=1,
        ),
    ).logits[
        0,
        encoded["input_ids"].shape[1] - 1:
        encoded["input_ids"].shape[1] - 1 + len(response_ids),
    ]
    decoded_bits = []
    for scores, token_id in zip(decoder_scores, response_ids):
        decoded_bits.extend(
            decode_adg_token_bits(
                scores,
                token_id,
                temperature=temperature,
                excluded_token_ids=excluded_token_ids,
            )
        )
    return decoded_bits


def verify_adg_ztest(
    queries: Sequence[str],
    responses: Sequence[str],
    *,
    bitstream: str | Sequence[int],
    carrier_prompt_values: Sequence[str],
    generation_temp: float = 1.0,
    stego_model_id: str | None = None,
    stego_tokenizer_id: str | None = None,
    stego_model: Any = None,
    stego_tokenizer: Any = None,
    device_map: str | None = "auto",
    dtype: str | None = "auto",
    alpha: float | None = 1e-3,
    trust_remote_code: bool = False,
    generation_params_used: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score responses by ADG bit agreement against an owner bitstream."""
    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    if len(queries) != len(responses):
        raise ValueError(
            f"verifier inputs must have the same length "
            f"(queries={len(queries)}, responses={len(responses)})"
        )
    expected_bits = parse_adg_bitstream(bitstream)
    if len(carrier_prompt_values) != len(responses):
        raise ValueError(
            "carrier_prompt_values must have the same length as responses "
            f"(carrier_prompt_values={len(carrier_prompt_values)}, "
            f"responses={len(responses)})"
        )

    if stego_model is None:
        if not stego_model_id:
            raise ValueError("stego_model_id is required")
        stego_model = load_model(
            stego_model_id,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    if stego_tokenizer is None:
        stego_tokenizer = load_tokenizer(
            stego_tokenizer_id or stego_model_id,
            trust_remote_code=trust_remote_code,
        )

    hits = 0
    n = 0
    per_sample = []
    for query, response, prompt in zip(
        queries,
        responses,
        carrier_prompt_values,
    ):
        decoded_bits = decode_adg_bitstream(
            stego_model,
            stego_tokenizer,
            format_input(stego_tokenizer, prompt),
            response,
            temperature=generation_temp,
        )
        compared = min(len(decoded_bits), len(expected_bits))
        sample_hits = sum(
            decoded_bits[index] == expected_bits[index]
            for index in range(compared)
        )
        hits += sample_hits
        n += compared
        per_sample.append(
            {
                "query": query,
                "response": response,
                "carrier_prompt": prompt,
                "num_decoded_bits": len(decoded_bits),
                "num_compared_bits": compared,
                "hits": sample_hits,
            }
        )

    if n == 0:
        raise ValueError("No ADG bits available for adg_ztest")

    gamma = 0.5
    z_score = (hits - n * gamma) / math.sqrt(n * gamma * (1.0 - gamma))
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    verification_score = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    verification_metadata = {
        "verifier": "adg_ztest",
        "verification_score": verification_score,
        "num_samples": len(responses),
        "num_compared_bits": n,
        "hits": hits,
        "bit_accuracy": hits / n,
        "z_score": z_score,
        "p_value": p_value,
        "per_sample": per_sample,
        "generation_params_used": generation_params_used,
    }
    if alpha is not None:
        verification_metadata["alpha"] = alpha
        verification_metadata["fingerprinted"] = p_value < alpha
    return verification_score, verification_metadata
