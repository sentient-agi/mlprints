"""
Base Verification Classes

Defines the common data structures and enumerations for fingerprint verification.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class FingerprintType(Enum):
    """Types of fingerprints supported."""
    SIMPLE = "simple"
    FUNCTIONAL = "functional"
    CRYPTOGRAPHIC = "cryptographic"  
    RANGE_BASED = "range_based"
    PARITY_CHECK = "parity_check"


class VerificationMode(Enum):
    """Verification modes."""
    EXACT_MATCH = "exact_match"
    PREFIX_MATCH = "prefix_match"
    SEMANTIC_MATCH = "semantic_match"
    FUNCTIONAL_MATCH = "functional_match"


@dataclass
class FingerprintConfig:
    """Configuration for fingerprint operations."""
    fingerprint_type: FingerprintType
    verification_mode: VerificationMode = VerificationMode.PREFIX_MATCH
    threshold: float = 0.9
    max_key_length: int = 32
    max_response_length: int = 32
    batch_size: int = 32
    device: str = "cuda"
    additional_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


# Utility functions for configuration
def create_simple_config(
    max_key_length: int = 32,
    max_response_length: int = 32,
    **kwargs
) -> FingerprintConfig:
    """Create a configuration for simple fingerprints."""
    return FingerprintConfig(
        fingerprint_type=FingerprintType.SIMPLE,
        max_key_length=max_key_length,
        max_response_length=max_response_length,
        **kwargs
    )


def create_functional_config(
    function_type: str = "arithmetic",
    max_key_length: int = 32,
    max_response_length: int = 32,
    **kwargs
) -> FingerprintConfig:
    """Create a configuration for functional fingerprints."""
    return FingerprintConfig(
        fingerprint_type=FingerprintType.FUNCTIONAL,
        verification_mode=VerificationMode.FUNCTIONAL_MATCH,
        max_key_length=max_key_length,
        max_response_length=max_response_length,
        additional_params={"function_type": function_type},
        **kwargs
    ) 