"""String-match verifier over query/response lists."""

import difflib
import re
import string
from collections.abc import Callable, Sequence
from typing import Any


def _strings_preprocessing(
    strings: Sequence[str],
    *,
    case_insensitive: bool = False,
    punctuation_insensitive: bool = False,
    normalize_whitespace: bool = False,
) -> list[str]:
    """Apply shared string preprocessing to each input string."""
    output = list(strings)
    if punctuation_insensitive:
        punct_table = str.maketrans("", "", string.punctuation)
        output = [value.translate(punct_table) for value in output]
    if normalize_whitespace:
        output = [re.sub(r"\s+", " ", value.strip()) for value in output]
    if case_insensitive:
        output = [value.casefold() for value in output]
    return output


def exact_comparator(
    expected: str,
    generated: str,
    *,
    soft: bool = False,
    threshold: float = 0.9,
    case_insensitive: bool = False,
    punctuation_insensitive: bool = False,
    normalize_whitespace: bool = False,
) -> bool:
    expected, generated = _strings_preprocessing(
        [expected, generated],
        case_insensitive=case_insensitive,
        punctuation_insensitive=punctuation_insensitive,
        normalize_whitespace=normalize_whitespace,
    )
    if soft:
        return difflib.SequenceMatcher(None, expected, generated).ratio() >= threshold
    return expected == generated


def prefix_comparator(
    expected: str,
    generated: str,
    *,
    soft: bool = False,
    threshold: float = 0.9,
    case_insensitive: bool = False,
    punctuation_insensitive: bool = False,
    normalize_whitespace: bool = False,
) -> bool:
    expected, generated = _strings_preprocessing(
        [expected, generated],
        case_insensitive=case_insensitive,
        punctuation_insensitive=punctuation_insensitive,
        normalize_whitespace=normalize_whitespace,
    )
    if soft:
        window = generated[: len(expected)]
        return difflib.SequenceMatcher(None, expected, window).ratio() >= threshold
    return generated.startswith(expected)


def suffix_comparator(
    expected: str,
    generated: str,
    *,
    soft: bool = False,
    threshold: float = 0.9,
    case_insensitive: bool = False,
    punctuation_insensitive: bool = False,
    normalize_whitespace: bool = False,
) -> bool:
    expected, generated = _strings_preprocessing(
        [expected, generated],
        case_insensitive=case_insensitive,
        punctuation_insensitive=punctuation_insensitive,
        normalize_whitespace=normalize_whitespace,
    )
    if soft:
        window = generated[-len(expected) :] if expected else ""
        return difflib.SequenceMatcher(None, expected, window).ratio() >= threshold
    return generated.endswith(expected)


def contains_comparator(
    expected: str,
    generated: str,
    *,
    soft: bool = False,
    threshold: float = 0.9,
    case_insensitive: bool = False,
    punctuation_insensitive: bool = False,
    normalize_whitespace: bool = False,
) -> bool:
    expected, generated = _strings_preprocessing(
        [expected, generated],
        case_insensitive=case_insensitive,
        punctuation_insensitive=punctuation_insensitive,
        normalize_whitespace=normalize_whitespace,
    )
    if soft:
        if len(generated) <= len(expected):
            return difflib.SequenceMatcher(None, expected, generated).ratio() >= threshold
        return (
            max(
                difflib.SequenceMatcher(
                    None,
                    expected,
                    generated[i : i + len(expected)],
                ).ratio()
                for i in range(len(generated) - len(expected) + 1)
            )
            >= threshold
        )
    return bool(expected) and expected in generated


COMPARATORS = {
    "exact": exact_comparator,
    "prefix": prefix_comparator,
    "suffix": suffix_comparator,
    "contains": contains_comparator,
}


def verify_match(
    queries: Sequence[str],
    responses: Sequence[str],
    *,
    expected_responses: Sequence[str],
    comparator: str | Callable[..., bool] = "exact",
    **comparator_kwargs,
) -> tuple[float, dict[str, Any]]:
    """Compare observed responses against expected responses with a string comparator.

    Can supply a custom comparator function or a built-in comparator name.
    Built-in comparators are registered in the ``COMPARATORS`` dictionary.
    """
    if isinstance(comparator, str):
        if comparator not in COMPARATORS:
            raise ValueError(f"comparator name ({comparator!r}) not supported. Please provide a custom comparator function.")
        comparator_fn = COMPARATORS[comparator]
        comparator_name = comparator
    elif callable(comparator):
        comparator_fn = comparator
        comparator_name = comparator.__name__
    else:
        raise TypeError("comparator must be a string name or callable")

    num_fingerprints = len(queries)
    if num_fingerprints == 0:
        raise ValueError("queries must be non-empty")
    if not (num_fingerprints == len(responses) == len(expected_responses)):
        raise ValueError(
            "verifier inputs must have the same length "
            f"(num_fingerprints={num_fingerprints}, responses={len(responses)}, "
            f"expected_responses={len(expected_responses)})"
        )
    hits = 0
    per_sample_hits = []
    for idx, (query, response, expected) in enumerate(zip(queries, responses, expected_responses)):
        is_hit = comparator_fn(expected, response, **comparator_kwargs)
        hits += is_hit
        per_sample_hits.append(
            {
                "index": idx,
                "is_hit": is_hit,
                "query": query,
                "response": response,
                "expected_response": expected,
            }
        )

    verification_score = hits / num_fingerprints
    verification_metadata = {
        "verifier": "match",
        "verification_score": verification_score,
        "comparator": comparator_name,
        "hits": hits,
        "num_fingerprints": num_fingerprints,
        "per_sample_hits": per_sample_hits,
    }

    return verification_score, verification_metadata
