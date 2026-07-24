"""Run inference and score a model's responses using a configured verifier and fingerprints."""

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
    apply_chat_template: bool = False,
    system_prompt: str | None = None,
    generation_params: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    verifier_config = verification_config["verifier"]
    verifier_name = verifier_config["name"]
    check_verifier(verifier_name)
    verifier = VERIFIERS[verifier_name]["verification_score"]
    verifier_params = dict(verifier_config.get("params", {}))

    if verifier_name == "match":
        expected_responses = [
            fingerprint["expected_response"]
            for fingerprint in fingerprints
        ]
        verifier_params["expected_responses"] = expected_responses
        if max_new_tokens is None:
            max_new_tokens = max(
                len(tokenizer.encode(response, add_special_tokens=False))
                for response in expected_responses
            )
    elif verifier_name == "watermark_ztest":
        watermark_fingerprint = fingerprints[0] # assumes only one fingerprint
        for param_name in (
            "secret_key",
            "gamma",
            "context_width",
            "tokenizer_id",
            "vocab_size",
            "special_token_ids",
            "exclude_special_tokens",
        ):
            if param_name in watermark_fingerprint:
                verifier_params[param_name] = watermark_fingerprint[param_name]

        if max_new_tokens is None:
            max_new_tokens = watermark_fingerprint.get("response_length")

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
