"""
OML Verification Engine

This module provides fingerprint generation and verification capabilities
for model verification and ownership authentication.
"""

# Base enums and configurations
from .base import (
    FingerprintType,
    VerificationMode,
    FingerprintConfig,
    create_simple_config,
    create_functional_config
)

# Fingerprint classes
from .fingerprints import (
    FingerprintSet,
    SimpleFingerprintSet,
    FunctionalFingerprintSet,
    CompositeFingerprintSet,
    create_fingerprint_set,
    load_fingerprint_set_from_file,
    merge_fingerprint_sets,
    validate_fingerprint_pairs,
    fingerprint_set_statistics
)

# Generation classes and functions
from .generate import (
    GenerationConfig,
    FingerprintGenerator,
    EnglishTextGenerator,
    RandomWordGenerator,
    InverseNucleusGenerator,
    create_generator,
    load_fingerprints_from_file,
    generate_english_fingerprints,
    generate_random_word_fingerprints
)

# Verification engine
from .verify import (
    VerificationEngine,
    VerificationResult,
    BatchVerificationResult
)

__all__ = [
    # Base types
    "FingerprintType",
    "VerificationMode", 
    "FingerprintConfig",
    "create_simple_config",
    "create_functional_config",
    
    # Fingerprint classes
    "FingerprintSet",
    "SimpleFingerprintSet", 
    "FunctionalFingerprintSet",
    "CompositeFingerprintSet",
    "create_fingerprint_set",
    "load_fingerprint_set_from_file",
    "merge_fingerprint_sets",
    "validate_fingerprint_pairs",
    "fingerprint_set_statistics",
    
    # Generation
    "GenerationConfig",
    "FingerprintGenerator",
    "EnglishTextGenerator",
    "RandomWordGenerator", 
    "InverseNucleusGenerator",
    "create_generator",
    "load_fingerprints_from_file",
    "generate_english_fingerprints",
    "generate_random_word_fingerprints",
    
    # Verification
    "VerificationEngine",
    "VerificationResult",
    "BatchVerificationResult"
] 