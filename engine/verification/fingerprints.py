"""
Fingerprint Implementations

This module implements various types of fingerprints for model verification,
extending the basic framework with advanced fingerprint strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional, Union
import json
import hashlib
import random
import re
from pathlib import Path

from .base import FingerprintType, VerificationMode


class FingerprintSet(ABC):
    """Abstract representation of a set of fingerprints.

    Each fingerprint set can implement a custom `matches` logic that works
    for simple exact (q,a) pairs or more complex functional fingerprints.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def contains(self, query: str, response: str) -> bool:
        """Return True if (query,response) pair satisfies this fingerprint set."""

    @abstractmethod
    def all_pairs(self) -> List[Dict[str, Any]]:
        """Return the canonical representation of fingerprint pairs for inspection."""
    
    @abstractmethod
    def get_expected_response(self, query: str) -> Optional[str]:
        """Get the expected response for a given query."""
        
    def verify_response(self, query: str, actual_response: str) -> bool:
        """Verify if the actual response matches the expected response for the query."""
        return self.contains(query, actual_response)
        
    def get_verification_confidence(self, query: str, actual_response: str) -> float:
        """Get verification confidence score between 0 and 1."""
        if self.contains(query, actual_response):
            return 1.0
        else:
            # Simple word overlap confidence
            expected = self.get_expected_response(query)
            if expected is None:
                return 0.0
            
            expected_words = set(expected.lower().split())
            actual_words = set(actual_response.lower().split())
            if len(expected_words) == 0 and len(actual_words) == 0:
                return 1.0
            elif len(expected_words) == 0 or len(actual_words) == 0:
                return 0.0
            
            overlap = len(expected_words & actual_words)
            total = len(expected_words | actual_words)
            return overlap / total
    
    def get_all_keys(self) -> List[str]:
        """Get all fingerprint keys."""
        return [pair.get('key', '') for pair in self.all_pairs()]
    
    def get_all_responses(self) -> List[str]:
        """Get all fingerprint responses."""
        return [pair.get('response', '') for pair in self.all_pairs()]
    
    def size(self) -> int:
        """Get the number of fingerprints in this set."""
        return len(self.all_pairs())
    
    def __len__(self):
        return self.size()
    
    def __iter__(self):
        """Allow iteration over fingerprint pairs."""
        return iter(self.all_pairs())
    
    def sample(self, n: int, random_seed: Optional[int] = None) -> 'FingerprintSet':
        """Sample n fingerprints from this set."""
        if random_seed is not None:
            random.seed(random_seed)
        
        all_pairs = self.all_pairs()
        if n >= len(all_pairs):
            return self
        
        sampled_pairs = random.sample(all_pairs, n)
        return SimpleFingerprintSet(sampled_pairs, name=f"{self.name}_sample_{n}")


class SimpleFingerprintSet(FingerprintSet):
    """Concrete implementation for exact (q,a) fingerprints."""

    def __init__(self, pairs: List[Dict[str, str]], name: str = "simple"):
        super().__init__(name)
        # Expect list of {"key": q, "response": a}
        self.pairs = pairs
        # Build lookup for O(1) check
        self._lookup = {}
        for p in pairs:
            key = p.get("key", "")
            response = p.get("response", "")
            if key in self._lookup:
                # Handle multiple responses for same key
                if isinstance(self._lookup[key], list):
                    self._lookup[key].append(response)
                else:
                    self._lookup[key] = [self._lookup[key], response]
            else:
                self._lookup[key] = response

    def contains(self, query: str, response: str) -> bool:
        expected = self._lookup.get(query)
        if expected is None:
            return False
        
        # Handle multiple expected responses
        if isinstance(expected, list):
            return any(response.startswith(exp) for exp in expected)
        else:
            return response.startswith(expected)
    
    def get_expected_response(self, query: str) -> Optional[Union[str, List[str]]]:
        return self._lookup.get(query)

    def all_pairs(self):
        return self.pairs

    def add_fingerprint(self, key: str, response: str):
        """Add a new fingerprint to the set."""
        self.pairs.append({"key": key, "response": response})
        # Update lookup
        if key in self._lookup:
            if isinstance(self._lookup[key], list):
                self._lookup[key].append(response)
            else:
                self._lookup[key] = [self._lookup[key], response]
        else:
            self._lookup[key] = response
    
    def remove_fingerprint(self, key: str) -> bool:
        """Remove fingerprint(s) with the given key."""
        if key not in self._lookup:
            return False
        
        # Remove from pairs
        self.pairs = [p for p in self.pairs if p.get("key") != key]
        # Remove from lookup
        del self._lookup[key]
        return True
    
    def merge_with(self, other: 'SimpleFingerprintSet') -> 'SimpleFingerprintSet':
        """Merge this fingerprint set with another."""
        combined_pairs = self.pairs + other.pairs
        return SimpleFingerprintSet(combined_pairs, name=f"{self.name}_merged_{other.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": "simple",
            "pairs": self.pairs,
            "size": len(self.pairs)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleFingerprintSet':
        """Create from dictionary."""
        name = data.get("name", "simple")
        pairs = data.get("pairs", [])
        return cls(pairs, name)
    
    def save_to_file(self, file_path: str):
        """Save fingerprint set to JSON file."""
        if not file_path.endswith('.json'):
            file_path = f"{file_path}.json"
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'SimpleFingerprintSet':
        """Load fingerprint set from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both old format (just list of pairs) and new format (with metadata)
        if isinstance(data, list):
            return cls(data, name=f"loaded_from_{Path(file_path).stem}")
        else:
            return cls.from_dict(data)


class FunctionalFingerprintSet(FingerprintSet):
    """
    Functional fingerprints based on computational relationships.
    
    These fingerprints embed functional relationships between inputs and outputs,
    making them more robust and harder to detect than simple string matching.
    """
    
    def __init__(self, function_type: str = "arithmetic", name: str = "functional", num_pairs: int = 100):
        super().__init__(name)
        self.function_type = function_type
        self.function_pairs = []
        self._generate_functional_pairs(num_pairs)
        
    def _generate_functional_pairs(self, num_pairs: int):
        """Generate functional fingerprint pairs."""
        if self.function_type == "arithmetic":
            self._generate_arithmetic_pairs(num_pairs)
        elif self.function_type == "logical":
            self._generate_logical_pairs(num_pairs)
        elif self.function_type == "string_transform":
            self._generate_string_transform_pairs(num_pairs)
        else:
            self._generate_arithmetic_pairs(num_pairs)  # Default
            
    def _generate_arithmetic_pairs(self, num_pairs: int):
        """Generate arithmetic function fingerprints."""
        for i in range(num_pairs):
            a, b = random.randint(10, 99), random.randint(10, 99)
            op = random.choice(['+', '-', '*'])
            
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            else:  # '*'
                result = a * b
                
            self.function_pairs.append({
                "key": f"What is {a} {op} {b}?",
                "response": str(result),
                "function": f"{a} {op} {b}",
                "expected_result": result
            })
            
    def _generate_logical_pairs(self, num_pairs: int):
        """Generate logical function fingerprints."""
        # TODO: Implement more sophisticated logical operations
        for i in range(num_pairs):
            a, b = random.choice([True, False]), random.choice([True, False])
            op = random.choice(['AND', 'OR', 'XOR'])
            
            if op == 'AND':
                result = a and b
            elif op == 'OR':
                result = a or b
            else:  # 'XOR'
                result = a != b
                
            self.function_pairs.append({
                "key": f"What is {a} {op} {b}?",
                "response": str(result),
                "function": f"{a} {op} {b}",
                "expected_result": result
            })
        
    def _generate_string_transform_pairs(self, num_pairs: int):
        """Generate string transformation fingerprints."""
        transforms = ['UPPER', 'LOWER', 'REVERSE', 'LENGTH']
        test_strings = ['hello', 'world', 'test', 'sample', 'data', 'string']
        
        for i in range(num_pairs):
            test_str = random.choice(test_strings)
            transform = random.choice(transforms)
            
            if transform == 'UPPER':
                result = test_str.upper()
            elif transform == 'LOWER':
                result = test_str.lower()
            elif transform == 'REVERSE':
                result = test_str[::-1]
            else:  # 'LENGTH'
                result = str(len(test_str))
                
            self.function_pairs.append({
                "key": f"Apply {transform} to '{test_str}'",
                "response": result,
                "function": f"{transform}({test_str})",
                "expected_result": result
            })
        
    def contains(self, query: str, response: str) -> bool:
        """Check if query-response pair satisfies functional relationship."""
        for pair in self.function_pairs:
            if pair["key"] == query:
                if self.function_type == "arithmetic":
                    try:
                        return int(response.strip()) == pair["expected_result"]
                    except ValueError:
                        return response.strip() == pair["response"]
                else:
                    return response.strip() == pair["response"]
        return False
        
    def get_expected_response(self, query: str) -> Optional[str]:
        """Get expected response for functional query."""
        for pair in self.function_pairs:
            if pair["key"] == query:
                return pair["response"]
        return None
        
    def all_pairs(self) -> List[Dict[str, Any]]:
        """Return all functional pairs."""
        return self.function_pairs


class CompositeFingerprintSet(FingerprintSet):
    """
    Composite fingerprint set combining multiple fingerprint types.
    
    This allows for sophisticated fingerprinting strategies that use
    multiple verification approaches simultaneously.
    """
    
    def __init__(self, fingerprint_sets: List[FingerprintSet], name: str = "composite"):
        super().__init__(name)
        self.fingerprint_sets = fingerprint_sets
        
    def contains(self, query: str, response: str) -> bool:
        """Check if any constituent fingerprint set contains the pair."""
        return any(fp_set.contains(query, response) for fp_set in self.fingerprint_sets)
        
    def get_expected_response(self, query: str) -> Optional[str]:
        """Get expected response from any constituent set."""
        for fp_set in self.fingerprint_sets:
            expected = fp_set.get_expected_response(query)
            if expected is not None:
                return expected
        return None
        
    def all_pairs(self) -> List[Dict[str, Any]]:
        """Return all pairs from all constituent sets."""
        all_pairs = []
        for fp_set in self.fingerprint_sets:
            pairs = fp_set.all_pairs()
            # Add source information
            for pair in pairs:
                pair_copy = pair.copy()
                pair_copy["source_set"] = fp_set.name
                all_pairs.append(pair_copy)
        return all_pairs
        
    def get_verification_confidence(self, query: str, actual_response: str) -> float:
        """Get maximum confidence from all constituent sets."""
        max_confidence = 0.0
        for fp_set in self.fingerprint_sets:
            confidence = fp_set.get_verification_confidence(query, actual_response)
            max_confidence = max(max_confidence, confidence)
        return max_confidence
    
    def add_fingerprint_set(self, fingerprint_set: FingerprintSet):
        """Add a new fingerprint set to the composite."""
        self.fingerprint_sets.append(fingerprint_set)
    
    def remove_fingerprint_set(self, name: str) -> bool:
        """Remove a fingerprint set by name."""
        for i, fp_set in enumerate(self.fingerprint_sets):
            if fp_set.name == name:
                del self.fingerprint_sets[i]
                return True
        return False


def create_fingerprint_set(
    fingerprint_type: FingerprintType,
    config: Optional[Dict[str, Any]] = None
) -> FingerprintSet:
    """
    Factory function to create fingerprint sets.
    
    Args:
        fingerprint_type: Type of fingerprint set to create
        config: Configuration parameters
        
    Returns:
        Appropriate FingerprintSet instance
    """
    config = config or {}
    
    if fingerprint_type == FingerprintType.SIMPLE:
        pairs = config.get("pairs", [])
        name = config.get("name", "simple")
        return SimpleFingerprintSet(pairs, name)
    elif fingerprint_type == FingerprintType.FUNCTIONAL:
        function_type = config.get("function_type", "arithmetic")
        name = config.get("name", "functional")
        num_pairs = config.get("num_pairs", 100)
        return FunctionalFingerprintSet(function_type, name, num_pairs)
    else:
        # Default to simple
        return SimpleFingerprintSet([])


def load_fingerprint_set_from_file(file_path: str) -> FingerprintSet:
    """
    Load a fingerprint set from file.
    
    Args:
        file_path: Path to the fingerprint file
        
    Returns:
        Loaded FingerprintSet
    """
    return SimpleFingerprintSet.load_from_file(file_path)


def merge_fingerprint_sets(sets: List[FingerprintSet]) -> CompositeFingerprintSet:
    """
    Merge multiple fingerprint sets into a composite set.
    
    Args:
        sets: List of fingerprint sets to merge
        
    Returns:
        CompositeFingerprintSet containing all input sets
    """
    return CompositeFingerprintSet(sets, name="merged_composite")


# Utility functions for working with fingerprints
def validate_fingerprint_pairs(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and clean fingerprint pairs.
    
    Args:
        pairs: List of fingerprint pairs to validate
        
    Returns:
        List of validated pairs
    """
    validated = []
    for pair in pairs:
        if isinstance(pair, dict) and 'key' in pair and 'response' in pair:
            # Clean strings
            key = str(pair['key']).strip()
            response = str(pair['response']).strip()
            
            if key and response:  # Both must be non-empty
                validated.append({'key': key, 'response': response})
    
    return validated


def fingerprint_set_statistics(fingerprint_set: FingerprintSet) -> Dict[str, Any]:
    """
    Compute statistics for a fingerprint set.
    
    Args:
        fingerprint_set: The fingerprint set to analyze
        
    Returns:
        Dictionary containing statistics
    """
    pairs = fingerprint_set.all_pairs()
    keys = [pair.get('key', '') for pair in pairs]
    responses = [pair.get('response', '') for pair in pairs]
    
    stats = {
        'total_pairs': len(pairs),
        'unique_keys': len(set(keys)),
        'unique_responses': len(set(responses)),
        'avg_key_length': sum(len(k) for k in keys) / len(keys) if keys else 0,
        'avg_response_length': sum(len(r) for r in responses) / len(responses) if responses else 0,
        'duplicate_keys': len(keys) - len(set(keys)),
        'duplicate_responses': len(responses) - len(set(responses))
    }
    
    return stats 