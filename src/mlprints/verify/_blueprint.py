"""
Blueprint for adding a new verifier.

REGISTRY CONTRACT:
- Register the verifier in `mlprints.common.verifiers.VERIFIERS`.
- Accept non-empty, same-length `queries` and `responses`.
- All verifier-specific parameters should be keyword-only.
- Return `(verification_score, verification_metadata)`, where the score is in
  `[0, 1]` and the metadata is a `dict`.
- Verifiers operate on collected outputs and should not run inference.
"""

from collections.abc import Sequence
from typing import Any


# FIXME: INSERT VERIFIER FUNCTION HERE
# Use a straightforward name, e.g. `verify_match` for the `match` registry entry.
def verify_fixme_blueprint_verifier_name(
    queries: Sequence[str],
    responses: Sequence[str],
    *,
    required_verifier_param: Any,
    optional_verifier_param: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """FIXME: Short description of what this verifier measures."""
    if len(queries) == 0:
        raise ValueError("queries must be non-empty")
    if len(queries) != len(responses):
        raise ValueError(
            f"verifier inputs must have the same length "
            f"(queries={len(queries)}, responses={len(responses)})"
        )

    if required_verifier_param is None:
        raise ValueError("required_verifier_param is required")
    if optional_verifier_param < 0:
        raise ValueError("optional_verifier_param must be non-negative")

    verification_score = 0.0
    verification_metadata = {
        "verifier": "fixme_blueprint_verifier_name",
        "num_samples": len(queries),
    }
    return verification_score, verification_metadata
