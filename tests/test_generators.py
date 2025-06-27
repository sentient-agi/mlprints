#!/usr/bin/env python3
"""
Test script to verify that the updated generators work with the new fingerprint abstractions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'engine'))

from engine.verification import (
    GenerationConfig,
    RandomWordGenerator,
    TokenExistenceGenerator,
    RegexGenerator,
    create_generator,
    VerificationType
)


def test_random_word_generator():
    """Test the RandomWordGenerator."""
    print("Testing RandomWordGenerator...")
    
    config = GenerationConfig(
        num_fingerprints=5,
        key_length=3,
        response_length=3,
        seed=42
    )
    
    generator = RandomWordGenerator(config)
    
    # Test fingerprint set generation
    fp_set = generator.generate_fingerprint_set()
    print(f"Set size: {len(fp_set)}")
    print(f"Set type: {type(fp_set).__name__}")
    print(f"All fingerprints are simple: {all(fp.fingerprint_type == VerificationType.SIMPLE for fp in fp_set.fingerprints)}")
    
    # Test verification
    first_fp = list(fp_set.fingerprints)[0]
    query = first_fp.get_query()
    response = first_fp.verification_functions[0].expected_response
    
    print(f"Verification test: {first_fp.verify(response)}")
    print(f"Set verification test: {fp_set.verify(query, response)}")
    print()


def test_token_existence_generator():
    """Test the TokenExistenceGenerator."""
    print("Testing TokenExistenceGenerator...")
    
    config = GenerationConfig(
        num_fingerprints=3,
        seed=42
    )
    
    generator = TokenExistenceGenerator(config, num_tokens_per_response=2)
    
    # Test fingerprint set generation
    fp_set = generator.generate_fingerprint_set()
    print(f"Set size: {len(fp_set)}")
    print(f"Set type: {type(fp_set).__name__}")
    print(f"All fingerprints are token existence: {all(fp.fingerprint_type == VerificationType.TOKEN_EXISTENCE for fp in fp_set.fingerprints)}")
    print()


def test_regex_generator():
    """Test the RegexGenerator."""
    print("Testing RegexGenerator...")
    
    config = GenerationConfig(
        num_fingerprints=3,
        seed=42
    )
    
    generator = RegexGenerator(config)
    
    # Test fingerprint set generation
    fp_set = generator.generate_fingerprint_set()
    print(f"Set size: {len(fp_set)}")
    print(f"Set type: {type(fp_set).__name__}")
    print(f"All fingerprints are regex: {all(fp.fingerprint_type == VerificationType.REGEX for fp in fp_set.fingerprints)}")
    print()


def test_factory_function():
    """Test the create_generator factory function."""
    print("Testing create_generator factory...")
    
    config = GenerationConfig(num_fingerprints=2, seed=42)
    
    # Test different generator types
    for gen_type in ["random_word", "token_existence", "regex"]:
        generator = create_generator(gen_type, config)
        fp_set = generator.generate_fingerprint_set()
        print(f"{gen_type} generator created set of size: {len(fp_set)}")
    
    print()


def test_convenience_functions():
    """Test the generator abstractions."""
    print("Testing generator abstractions...")
    
    # Test random word fingerprints using generator
    config1 = GenerationConfig(num_fingerprints=3, key_length=2, response_length=2)
    generator1 = RandomWordGenerator(config1)
    fp_set1 = generator1.generate_fingerprint_set()
    print(f"Random word set size: {len(fp_set1)}")
    
    # Test token existence fingerprints using generator
    config2 = GenerationConfig(num_fingerprints=3)
    generator2 = TokenExistenceGenerator(config2, num_tokens_per_response=1)
    fp_set2 = generator2.generate_fingerprint_set()
    print(f"Token existence set size: {len(fp_set2)}")
    
    # Test regex fingerprints using generator
    config3 = GenerationConfig(num_fingerprints=3)
    generator3 = RegexGenerator(config3)
    fp_set3 = generator3.generate_fingerprint_set()
    print(f"Regex set size: {len(fp_set3)}")
    
    print()


if __name__ == "__main__":
    print("=== Testing Updated Fingerprint Generators ===\n")
    
    test_random_word_generator()
    test_token_existence_generator()
    test_regex_generator()
    test_factory_function()
    test_convenience_functions()
    
    print("=== All tests completed! ===") 