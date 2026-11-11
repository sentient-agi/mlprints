"""NonceProbe: a deterministic query/response fingerprint baseline.

This implementation stores target-model responses to nonce-bearing prompts and
verifies later outputs by normalized similarity. It is an experimental baseline,
not a validated robust fingerprinting method.
"""
from collections.abc import Sequence
from difflib import SequenceMatcher
import random
import re
from typing import Any

from mlprints.inference import run_inference


_TOPICS = (
    "Explain one practical use of a hash function.",
    "Describe why a database index can improve query performance.",
    "Give a concise explanation of a compiler.",
    "Explain the difference between authentication and authorization.",
    "Describe how a queue is used in software systems.",
    "Explain one advantage of data compression.",
    "Describe what an API endpoint does.",
    "Explain why software tests are useful.",
)


def nonce_probe_fp(
    target_model: Any,
    target_tokenizer: Any,
    *,
    num_fingerprints: int = 32,
    max_new_tokens: int = 128,
    batch_size: int = 4,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect deterministic target responses to nonce-bearing prompts."""
    if num_fingerprints < 1:
        raise ValueError("num_fingerprints must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    rng = random.Random(seed)
    prompts: list[str] = []
    for index in range(num_fingerprints):
        nonce = f"{rng.getrandbits(64):016x}"
        topic = _TOPICS[index % len(_TOPICS)]
        prompts.append(
            f"Provenance challenge ID: {nonce}\n"
            f"{topic}\n"
            "Answer specifically and concisely."
        )

    fingerprints: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        outputs = run_inference(
            target_model,
            target_tokenizer,
            batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"inference returned {len(outputs)} outputs for {len(batch)} prompts"
            )
        for offset, (query, response) in enumerate(zip(batch, outputs)):
            fingerprints.append(
                {
                    "query": query,
                    "expected_response": response,
                    "index": start + offset,
                }
            )
            metadata.append(
                {
                    "nonce": query.splitlines()[0].rsplit(" ", 1)[-1],
                    "topic_index": (start + offset) % len(_TOPICS),
                }
            )

    return fingerprints, metadata


def verification_queries(
    fingerprints: Sequence[dict[str, Any]],
    fingerprints_metadata: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    """Return the exact prompts used to create the stored response signatures."""
    del fingerprints_metadata, config
    return [str(item["query"]) for item in fingerprints]


def _normalize(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip().casefold()


def verify_nonce_probe_fp(
    queries: Sequence[str],
    responses: Sequence[str],
    *,
    expected_response_values: Sequence[str],
    match_threshold: float = 0.98,
) -> tuple[float, dict[str, Any]]:
    """Score the fraction of responses matching stored target signatures."""
    if not 0.0 <= match_threshold <= 1.0:
        raise ValueError("match_threshold must be in [0, 1]")
    if not queries or len(queries) != len(responses):
        raise ValueError("queries and responses must be non-empty and aligned")
    if len(responses) != len(expected_response_values):
        raise ValueError(
            "responses and expected_response_values must have equal length"
        )

    expected = [_normalize(str(response)) for response in expected_response_values]
    observed = [_normalize(str(response)) for response in responses]
    similarities = [
        SequenceMatcher(None, expected_text, observed_text).ratio()
        for expected_text, observed_text in zip(expected, observed)
    ]
    matches = [score >= match_threshold for score in similarities]
    verification_score = sum(matches) / len(matches)
    return verification_score, {
        "matched": sum(matches),
        "total": len(matches),
        "match_threshold": match_threshold,
        "mean_similarity": sum(similarities) / len(similarities),
        "similarities": similarities,
    }
