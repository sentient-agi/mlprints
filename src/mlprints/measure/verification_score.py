"""Run inference and score a model's responses using a configured verifier and fingerprints."""

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from mlprints.common.verifiers import VERIFIERS, check_verifier
from mlprints.inference import run_inference


def measure_verification_score(
    model: Any,
    tokenizer: Any,
    queries: Sequence[str],
    verification_config: Mapping[str, Any],
    fingerprints: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int | None = None,
    batch_size: int = 128,
    apply_chat_template: bool = True,
    system_prompt: str | None = None,
    generation_params: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    verifier_config = verification_config["verifier"]
    verifier_name = verifier_config["name"]
    check_verifier(verifier_name)
    verifier = VERIFIERS[verifier_name]["verification_score"]
    configured_params = dict(verifier_config.get("params", {}))
    verifier_params = {}
    for parameter in inspect.signature(verifier).parameters.values():
        if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue

        name = parameter.name
        if name in configured_params:
            continue

        if name.endswith("_values"):
            field_name = name.removesuffix("_values")
            if not any(field_name in fingerprint for fingerprint in fingerprints):
                continue
            if not all(field_name in fingerprint for fingerprint in fingerprints):
                raise ValueError(
                    f"fingerprint field {field_name!r} must be present in every "
                    f"fingerprint to populate verifier parameter {name!r}"
                )
            verifier_params[name] = [
                fingerprint[field_name] for fingerprint in fingerprints
            ]
            continue

        values = [
            fingerprint[name]
            for fingerprint in fingerprints
            if name in fingerprint
        ]
        if not values:
            continue
        if len(values) != len(fingerprints):
            raise ValueError(
                f"shared fingerprint field {name!r} must be present in every "
                "fingerprint"
            )
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"shared fingerprint field {name!r} differs across fingerprints; "
                f"declare verifier parameter {name + '_values'!r} to collect "
                "per-fingerprint values"
            )
        verifier_params[name] = values[0]
    verifier_params.update(configured_params)

    if max_new_tokens is None:
        lengths = []
        for fingerprint in fingerprints:
            response_length = fingerprint.get("response_length")
            if response_length is not None:
                lengths.append(int(response_length))
                continue
            expected_response = fingerprint.get("expected_response")
            if expected_response is not None:
                lengths.append(
                    len(
                        tokenizer.encode(
                            expected_response,
                            add_special_tokens=False,
                        )
                    )
                )
        max_new_tokens = max(lengths) if lengths else None

    generation_params = dict(generation_params or {})
    responses = []
    for start in range(0, len(queries), batch_size):
        query_batch = queries[start : start + batch_size]
        batch_responses = run_inference(
            model=model,
            tokenizer=tokenizer,
            prompt_or_messages=query_batch,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            output_scores=False,
            **generation_params,
        )
        responses.extend(batch_responses)

    verification_score, verification_metadata = verifier(
        queries,
        responses,
        **verifier_params,
    )
    verification_metadata["queries"] = queries
    verification_metadata["responses"] = responses
    return verification_score, verification_metadata
