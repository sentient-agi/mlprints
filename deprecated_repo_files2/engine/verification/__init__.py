"""
OML Verification Engine

This module provides fingerprint generation and verification capabilities
for model verification and ownership authentication.
"""

# Base enums and verification functions
from .base import (
    VerificationType,
    VerificationFunction,
    SimpleVerificationFunction,
    TokenExistenceVerificationFunction,
    RegexVerificationFunction
)

# Fingerprint classes
from .fingerprints import (
    CombinationStrategy,
    Fingerprint,
    SimpleFingerprint,
    TokenExistenceFingerprint,
    RegexFingerprint,
    FingerprintSet,
    fingerprint_set_statistics
)

# Generation classes and functions
from .generate import (
    GenerationConfig,
    WordListManager,
    TextGenerator,
    InverseNucleusGenerator,
    FingerprintGenerator,
    SimpleTextGenerator,
    RandomWordGenerator,
    TokenExistenceGenerator,
    RegexGenerator,
    InverseNucleusFingerprintGenerator,
    create_generator
)

__all__ = [
    # Base types and verification functions
    "VerificationType",
    "CombinationStrategy",
    "VerificationFunction",
    "SimpleVerificationFunction",
    "TokenExistenceVerificationFunction",
    "RegexVerificationFunction",
    
    # Fingerprint classes
    "Fingerprint",
    "SimpleFingerprint",
    "TokenExistenceFingerprint",
    "RegexFingerprint",
    "FingerprintSet",
    "fingerprint_set_statistics",
    
    # Generation
    "GenerationConfig",
    "WordListManager",
    "TextGenerator",
    "InverseNucleusGenerator",
    "FingerprintGenerator",
    "SimpleTextGenerator",
    "RandomWordGenerator",
    "TokenExistenceGenerator",
    "RegexGenerator",
    "InverseNucleusFingerprintGenerator",
    "create_generator"
] 