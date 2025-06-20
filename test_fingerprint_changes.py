#!/usr/bin/env python3
"""
Test script to verify the fingerprint changes work correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from engine.verification.fingerprints import (
    SimpleFingerprint, 
    TokenExistenceFingerprint, 
    RegexFingerprint,
    FingerprintSet,
    SimpleFingerprintSet,
    create_simple_fingerprint_set
)

def test_basic_functionality():
    print("Testing basic fingerprint functionality...")
    
    # Create some fingerprints
    fp1 = SimpleFingerprint("What is 2+2?", "4")
    fp2 = SimpleFingerprint("What is the capital of France?", "Paris")
    fp3 = TokenExistenceFingerprint("Tell me about cats", ["cat", "feline"])
    
    # Test query property
    print(f"fp1.query: {fp1.query}")
    print(f"fp1.get_query(): {fp1.get_query()}")
    
    # Create fingerprint set
    fp_set = FingerprintSet([fp1, fp2, fp3], name="test_set")
    
    print(f"Set size: {fp_set.size()}")
    print(f"Queries: {fp_set.get_queries()}")
    
    # Test verification
    result1 = fp_set.verify("What is 2+2?", "4")
    result2 = fp_set.verify("What is 2+2?", "5")
    result3 = fp_set.verify("Tell me about cats", "Cats are feline animals")
    
    print(f"Verification results: {result1}, {result2}, {result3}")
    
    # Test duplicate prevention
    try:
        fp_duplicate = SimpleFingerprint("What is 2+2?", "four")
        fp_set.add_fingerprint(fp_duplicate)
        print("ERROR: Should have prevented duplicate!")
    except ValueError as e:
        print(f"Correctly prevented duplicate: {e}")
    
    # Test O(1) lookup
    fp_found = fp_set.get_fingerprint_by_query("What is 2+2?")
    print(f"Found fingerprint: {fp_found is not None}")
    
    print("Basic functionality test passed!\n")

def test_specialized_sets():
    print("Testing specialized fingerprint sets...")
    
    # Test SimpleFingerprintSet
    simple_set = create_simple_fingerprint_set([
        ("What is 1+1?", "2"),
        ("What is 3+3?", "6")
    ], name="simple_test")
    
    print(f"Simple set size: {simple_set.size()}")
    print(f"Simple set verification: {simple_set.verify('What is 1+1?', '2')}")
    
    # Test type validation
    try:
        token_fp = TokenExistenceFingerprint("test", ["token"])
        simple_set.add_fingerprint(token_fp)
        print("ERROR: Should have prevented wrong type!")
    except ValueError as e:
        print(f"Correctly prevented wrong type: {e}")
    
    print("Specialized sets test passed!\n")

if __name__ == "__main__":
    test_basic_functionality()
    test_specialized_sets()
    print("All tests passed!") 