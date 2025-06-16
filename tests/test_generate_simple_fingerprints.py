#!/usr/bin/env python3
"""
Test script for fingerprint generation functionality.
This script runs a minimal test to ensure the generation system works.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add the engine directory to the Python path
script_dir = Path(__file__).parent
engine_dir = script_dir.parent / "engine"
sys.path.insert(0, str(engine_dir))

from verification.generate import (
    GenerationConfig,
    create_generator,
    RandomWordGenerator
)


def test_random_word_generation():
    """Test random word fingerprint generation (doesn't require models)."""
    print("Testing random word fingerprint generation...")
    
    # Create a temporary word list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        words = ['test', 'hello', 'world', 'example', 'data', 'sample', 'random', 'word']
        for word in words:
            f.write(f"{word}\n")
        word_list_path = f.name
    
    try:
        # Create config and generator with word_list_path
        config = GenerationConfig(
            num_fingerprints=5,
            key_length=3,
            response_length=2,
            seed=123
        )
        
        generator = RandomWordGenerator(config, word_list_path=word_list_path)
        fingerprints = generator.generate()
        
        print(f"Generated {len(fingerprints.all_pairs())} fingerprints")
        
        # Display first few fingerprints
        for i, pair in enumerate(fingerprints.all_pairs()[:3]):
            print(f"  {i+1}. Key: '{pair['key']}' -> Response: '{pair['response']}'")
        
        return True
        
    finally:
        # Clean up
        os.unlink(word_list_path)


def test_config_creation():
    """Test that GenerationConfig can be created properly."""
    print("Testing configuration creation...")
    
    config = GenerationConfig(
        num_fingerprints=10,
        key_length=5,
        response_length=3,
        temperature=0.8,
        seed=123
    )
    
    print(f"Config created: {config.num_fingerprints} fingerprints, "
          f"key_length={config.key_length}, response_length={config.response_length}")
    
    return True


def test_generator_factory():
    """Test that the generator factory works."""
    print("Testing generator factory...")
    
    config = GenerationConfig(num_fingerprints=5, key_length=3, response_length=2)
    
    # Test random word generator creation
    try:
        generator = create_generator('random_word', config)
        print(f"Successfully created {type(generator).__name__}")
        return True
    except Exception as e:
        print(f"Error creating generator: {e}")
        return False


def main():
    """Run all tests."""
    print("Running fingerprint generation tests...\n")
    
    tests = [
        test_config_creation,
        test_generator_factory,
        test_random_word_generation,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                print("✓ PASSED\n")
                passed += 1
            else:
                print("✗ FAILED\n")
        except Exception as e:
            print(f"✗ FAILED with exception: {e}\n")
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("All tests passed! The fingerprint generation system is working.")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 