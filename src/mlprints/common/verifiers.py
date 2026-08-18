"""
Verifier registry.

Each entry maps a verifier name to:
- verification_score: callable(queries, responses, **kwargs)
    -> (verification_score, verification_metadata)
    - verification_score: float in [0, 1]
    - verification_metadata: dict
"""

from mlprints.verify.adg_ztest import verify_adg_ztest
from mlprints.verify.match import verify_match
from mlprints.verify.watermark_ztest import verify_watermark_ztest


VERIFIERS = {
    # "verifier": {"verification_score": verify_function}
    # For the required verifier contract, see `mlprints.verify._blueprint`.
    "adg_ztest": {"verification_score": verify_adg_ztest},
    "match": {"verification_score": verify_match},
    "watermark_ztest": {"verification_score": verify_watermark_ztest},
}


def check_verifier(name: str) -> None:
    """Validate that *name* is a registered verifier."""
    if name not in VERIFIERS:
        raise ValueError(
            f"unknown verifier '{name}'. "
            f"registered: {sorted(VERIFIERS)}"
        )
