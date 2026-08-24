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


def _normalize_max_bits(
    max_bits: int | Sequence[int | None] | None,
    batch_size: int,
) -> list[int | None]:
    if max_bits is None:
        return [None] * batch_size
    if isinstance(max_bits, int):
        if max_bits < 0:
            raise ValueError("max_bits must be non-negative")
        return [max_bits] * batch_size
    if len(max_bits) != batch_size:
        raise ValueError(
            "max_bits must have the same length as responses "
            f"(max_bits={len(max_bits)}, responses={batch_size})"
        )
    normalized = []
    for value in max_bits:
        if value is None:
            normalized.append(None)
            continue
        value = int(value)
        if value < 0:
            raise ValueError("max_bits must be non-negative")
        normalized.append(value)
    return normalized


def _mask_adg_logits(
    scores: torch.Tensor,
    *,
    temperature: float,
    excluded_token_ids: Sequence[int] | None,
) -> torch.Tensor:
    logits = scores.float() / temperature
    if excluded_token_ids:
        logits = logits.clone()
        logits[..., list(excluded_token_ids)] = -torch.inf
    return logits


def _decode_adg_grouped_token_bits(
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    token_id: int,
    *,
    max_bits: int | None = None,
) -> list[int]:
    """Recover ADG bits from a token's sorted group probabilities."""
    decoded_bits = []
    while probs[0] <= 0.5:
        if max_bits is not None and len(decoded_bits) >= max_bits:
            break
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
    if max_bits is not None:
        return decoded_bits[:max_bits]
    return decoded_bits


def _decode_adg_response_bits(
    scores: torch.Tensor,
    response_ids: Sequence[int],
    *,
    temperature: float,
    excluded_token_ids: Sequence[int] | None,
    max_bits: int | None = None,
) -> list[int]:
    """Decode ADG bits from one response's per-token scores."""
    if max_bits == 0 or scores.numel() == 0:
        return []

    logits = _mask_adg_logits(
        scores,
        temperature=temperature,
        excluded_token_ids=excluded_token_ids,
    )
    probs = torch.softmax(logits, dim=-1)
    grouping_positions = (probs.max(dim=-1).values <= 0.5).nonzero(
        as_tuple=False
    ).view(-1)
    if grouping_positions.numel() == 0:
        return []

    decoded_bits = []
    if max_bits is None:
        sorted_probs, sorted_ids = probs[grouping_positions].sort(
            descending=True
        )
        for local, position in enumerate(grouping_positions.tolist()):
            decoded_bits.extend(
                _decode_adg_grouped_token_bits(
                    sorted_probs[local],
                    sorted_ids[local],
                    response_ids[position],
                )
            )
        return decoded_bits

    for position in grouping_positions.tolist():
        if len(decoded_bits) >= max_bits:
            break
        token_probs, token_ids = probs[position].sort(descending=True)
        decoded_bits.extend(
            _decode_adg_grouped_token_bits(
                token_probs,
                token_ids,
                response_ids[position],
                max_bits=max_bits - len(decoded_bits),
            )
        )
    return decoded_bits[:max_bits]


def decode_adg_token_bits(
    scores: torch.Tensor,
    token_id: int,
    *,
    temperature: float = 1.0,
    excluded_token_ids: Sequence[int] | None = None,
    max_bits: int | None = None,
) -> list[int]:
    """Recover the ADG bits implied by one sampled token."""
    if max_bits is not None and max_bits < 0:
        raise ValueError("max_bits must be non-negative")
    return _decode_adg_response_bits(
        scores.unsqueeze(0),
        [token_id],
        temperature=temperature,
        excluded_token_ids=excluded_token_ids,
        max_bits=max_bits,
    )


def _excluded_special_token_ids(stego_model, stego_tokenizer) -> list[int]:
    eos_token_ids = get_eos_token_ids(stego_model, stego_tokenizer)
    return [
        token_id
        for token_id in stego_tokenizer.all_special_ids
        if token_id not in eos_token_ids
    ]


def _pad_token_id(tokenizer) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer must have pad_token_id or eos_token_id set")
    return int(pad_token_id)


@torch.inference_mode()
def decode_adg_bitstreams(
    stego_model,
    stego_tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    *,
    temperature: float = 1.0,
    excluded_token_ids: Sequence[int] | None = None,
    max_bits: int | Sequence[int | None] | None = None,
) -> list[list[int]]:
    """Decode ADG bitstreams hidden in each `responses` under `prompts`."""
    if len(prompts) != len(responses):
        raise ValueError(
            "prompts and responses must have the same length "
            f"(prompts={len(prompts)}, responses={len(responses)})"
        )
    max_bits_list = _normalize_max_bits(max_bits, len(responses))
    if excluded_token_ids is None:
        excluded_token_ids = _excluded_special_token_ids(
            stego_model,
            stego_tokenizer,
        )

    decoded = [[] for _ in responses]
    encode_indices = []
    prompt_id_lists = []
    response_id_lists = []
    for index, (prompt, response, sample_max_bits) in enumerate(
        zip(prompts, responses, max_bits_list)
    ):
        if sample_max_bits == 0:
            continue
        prompt_ids = stego_tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].tolist()
        response_ids = stego_tokenizer.encode(
            response,
            add_special_tokens=False,
        )
        if not response_ids:
            continue
        encode_indices.append(index)
        prompt_id_lists.append(prompt_ids)
        response_id_lists.append(list(response_ids))

    if not encode_indices:
        return decoded

    device = get_model_device(stego_model)
    full_id_lists = [
        prompt_ids + response_ids
        for prompt_ids, response_ids in zip(prompt_id_lists, response_id_lists)
    ]
    max_length = max(len(ids) for ids in full_id_lists)
    if all(len(ids) == max_length for ids in full_id_lists):
        input_ids = torch.tensor(
            full_id_lists,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
    else:
        pad_token_id = _pad_token_id(stego_tokenizer)
        input_ids = torch.full(
            (len(full_id_lists), max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(full_id_lists), max_length),
            dtype=torch.long,
            device=device,
        )
        for row, ids in enumerate(full_id_lists):
            input_ids[row, :len(ids)] = torch.tensor(
                ids,
                dtype=torch.long,
                device=device,
            )
            attention_mask[row, :len(ids)] = 1

    logits = stego_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    for row, index in enumerate(encode_indices):
        prompt_length = len(prompt_id_lists[row])
        response_ids = response_id_lists[row]
        sample_scores = logits[
            row,
            prompt_length - 1:prompt_length - 1 + len(response_ids),
        ]
        decoded[index] = _decode_adg_response_bits(
            sample_scores,
            response_ids,
            temperature=temperature,
            excluded_token_ids=excluded_token_ids,
            max_bits=max_bits_list[index],
        )
    return decoded


def decode_adg_bitstream(
    stego_model,
    stego_tokenizer,
    prompt: str,
    response: str,
    *,
    temperature: float = 1.0,
    excluded_token_ids: Sequence[int] | None = None,
    max_bits: int | None = None,
) -> list[int]:
    """Decode the ADG bitstream hidden in `response` under `prompt`."""
    return decode_adg_bitstreams(
        stego_model,
        stego_tokenizer,
        [prompt],
        [response],
        temperature=temperature,
        excluded_token_ids=excluded_token_ids,
        max_bits=max_bits,
    )[0]


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
    num_embedded_bits_values: Sequence[int] | None = None,
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
    if (
        num_embedded_bits_values is not None
        and len(num_embedded_bits_values) != len(responses)
    ):
        raise ValueError(
            "num_embedded_bits_values must have the same length as responses "
            f"(num_embedded_bits_values={len(num_embedded_bits_values)}, "
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

    formatted_prompts = [
        format_input(stego_tokenizer, prompt)
        for prompt in carrier_prompt_values
    ]
    decoded_bitstreams = decode_adg_bitstreams(
        stego_model,
        stego_tokenizer,
        formatted_prompts,
        responses,
        temperature=generation_temp,
        max_bits=num_embedded_bits_values,
    )

    hits = 0
    n = 0
    per_sample = []
    for query, response, prompt, decoded_bits, num_embedded_bits in zip(
        queries,
        responses,
        carrier_prompt_values,
        decoded_bitstreams,
        num_embedded_bits_values or [None] * len(responses),
    ):
        compared_limit = len(expected_bits)
        if num_embedded_bits is not None:
            compared_limit = min(compared_limit, int(num_embedded_bits))
        compared = min(len(decoded_bits), compared_limit)
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
