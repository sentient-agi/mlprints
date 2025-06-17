"""
OML Verification Engine

This module provides fingerprint generation and verification capabilities
for model verification and ownership authentication.
"""

# Base enums and verification functions
from .base import (
    VerificationType,
    CombinationStrategy,
    VerificationFunction,
    SimpleVerificationFunction,
    TokenExistenceVerificationFunction,
    RegexVerificationFunction,
    query_checked
)

# Fingerprint classes
from .fingerprints import (
    Fingerprint,
    SimpleFingerprint,
    TokenExistenceFingerprint,
    RegexFingerprint,
    FingerprintSet,
    SimpleFingerprintSet,
    TokenFingerprintSet,
    RegexFingerprintSet,
    create_simple_fingerprint_set,
    create_token_fingerprint_set,
    create_regex_fingerprint_set,
    create_fingerprint_set_from_config,
    merge_fingerprint_sets,
    fingerprint_set_statistics
)

# Generation classes and functions
from .generate import (
    GenerationConfig,
    FingerprintGenerator,
    SimpleTextGenerator,
    RandomWordGenerator,
    TokenExistenceGenerator,
    RegexGenerator,
    InverseNucleusGenerator,
    create_generator,
    load_fingerprints_from_file,
    generate_simple_text_fingerprints,
    generate_random_word_fingerprints,
    generate_token_existence_fingerprints,
    generate_regex_fingerprints,
    generate_english_fingerprints  # Legacy compatibility
)

__all__ = [
    # Base types and verification functions
    "VerificationType",
    "CombinationStrategy",
    "VerificationFunction",
    "SimpleVerificationFunction",
    "TokenExistenceVerificationFunction",
    "RegexVerificationFunction",
    "query_checked",
    
    # Fingerprint classes
    "Fingerprint",
    "SimpleFingerprint",
    "TokenExistenceFingerprint",
    "RegexFingerprint",
    "FingerprintSet",
    "SimpleFingerprintSet",
    "TokenFingerprintSet",
    "RegexFingerprintSet",
    "create_simple_fingerprint_set",
    "create_token_fingerprint_set", 
    "create_regex_fingerprint_set",
    "create_fingerprint_set_from_config",
    "merge_fingerprint_sets",
    "fingerprint_set_statistics",
    
    # Generation
    "GenerationConfig",
    "FingerprintGenerator",
    "SimpleTextGenerator",
    "RandomWordGenerator",
    "TokenExistenceGenerator",
    "RegexGenerator",
    "InverseNucleusGenerator",
    "create_generator",
    "load_fingerprints_from_file",
    "generate_simple_text_fingerprints",
    "generate_random_word_fingerprints",
    "generate_token_existence_fingerprints",
    "generate_regex_fingerprints",
    "generate_english_fingerprints"  # Legacy compatibility
] 