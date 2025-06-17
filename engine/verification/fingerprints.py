"""
Fingerprint Implementations

This module implements fingerprint sets and additional fingerprint types for model verification,
building on the verification framework from base.py.
"""

# Removed ABC import as FingerprintSet is no longer abstract
from typing import List, Dict, Any, Optional
import json
import random
from datetime import datetime
from pydantic import BaseModel, ConfigDict, model_validator, Field

from .base import (
    VerificationType, 
    CombinationStrategy,
    VerificationFunction,
    SimpleVerificationFunction,
    TokenExistenceVerificationFunction,
    RegexVerificationFunction
)


class Fingerprint(BaseModel):
    """A single fingerprint consisting of one-or-more verification functions.
    
    Note: Different functions could theoretically have different expected queries,
    which is why expected_query is stored at the VerificationFunction level rather
    than at the Fingerprint level.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    combination_strategy: CombinationStrategy = CombinationStrategy.SINGLE
    verification_functions: List[VerificationFunction] = []
    fingerprint_type: Optional[VerificationType] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_verification_functions(self) -> 'Fingerprint':
        """Validate that the combination strategy matches the number of functions."""
        num_functions = len(self.verification_functions)
        
        if self.combination_strategy == CombinationStrategy.SINGLE:
            if num_functions != 1:
                raise ValueError("Single combination strategy requires exactly one verification function")
        elif self.combination_strategy in [CombinationStrategy.UNION, CombinationStrategy.INTERSECT]:
            if num_functions < 2:
                raise ValueError(f"{self.combination_strategy.value} combination strategy requires at least two verification functions")
        
        return self

    def verify(self, query: str, response: str) -> bool:
        """Verify a fingerprint against a query and response."""
        if not self.verification_functions:
            raise ValueError("No verification functions defined")
            
        try:
            if self.combination_strategy == CombinationStrategy.SINGLE:
                return self.verification_functions[0](query, response)
            elif self.combination_strategy == CombinationStrategy.UNION:
                return any(func(query, response) for func in self.verification_functions)
            elif self.combination_strategy == CombinationStrategy.INTERSECT:
                return all(func(query, response) for func in self.verification_functions)
            else:
                raise ValueError(f"Unknown combination strategy: {self.combination_strategy}")
                
        except Exception as e:
            raise RuntimeError(f"Verification failed: {e}") from e


class SimpleFingerprint(Fingerprint):
    """A fingerprint with a single simple verification function."""
    
    def __init__(
        self, 
        expected_query: str, 
        expected_response: str, 
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        verification_function = SimpleVerificationFunction(expected_query, expected_response)
        
        # Prepare kwargs for parent class
        kwargs = {
            "combination_strategy": CombinationStrategy.SINGLE,
            "verification_functions": [verification_function],
            "fingerprint_type": VerificationType.SIMPLE,
        }
        
        if created_at is not None:
            kwargs["created_at"] = created_at
            
        if metadata is not None:
            kwargs["metadata"] = metadata
            
        super().__init__(**kwargs)


class TokenExistenceFingerprint(Fingerprint):
    """A fingerprint with a single token existence verification function."""
    
    def __init__(
        self, 
        expected_query: str, 
        required_tokens: List[str],
        case_sensitive: bool = True,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        verification_function = TokenExistenceVerificationFunction(
            expected_query, required_tokens, case_sensitive
        )
        
        kwargs = {
            "combination_strategy": CombinationStrategy.SINGLE,
            "verification_functions": [verification_function],
            "fingerprint_type": VerificationType.TOKEN_EXISTENCE,
        }
        
        if created_at is not None:
            kwargs["created_at"] = created_at
            
        if metadata is not None:
            kwargs["metadata"] = metadata
            
        super().__init__(**kwargs)


class RegexFingerprint(Fingerprint):
    """A fingerprint with a single regex verification function."""
    
    def __init__(
        self, 
        expected_query: str, 
        pattern: str,
        flags: int = 0,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        verification_function = RegexVerificationFunction(
            expected_query, pattern, flags
        )
        
        kwargs = {
            "combination_strategy": CombinationStrategy.SINGLE,
            "verification_functions": [verification_function],
            "fingerprint_type": VerificationType.REGEX,
        }
        
        if created_at is not None:
            kwargs["created_at"] = created_at
            
        if metadata is not None:
            kwargs["metadata"] = metadata
            
        super().__init__(**kwargs)


class FingerprintSet:
    """A general fingerprint set that can contain any types of fingerprints.
    
    A fingerprint set contains multiple Fingerprint objects and provides
    collective verification capabilities.
    """

    def __init__(self, fingerprints: List[Fingerprint], name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.name = name or "unnamed_set"
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.fingerprints = fingerprints

    def size(self) -> int:
        """Get the number of fingerprints in this set."""
        return len(self.fingerprints)

    def verify(self, query: str, response: str) -> bool:
        """Return True if any fingerprint in this set verifies the pair."""
        return any(fingerprint.verify(query, response) for fingerprint in self.fingerprints)
    
    def get_fingerprints(self) -> List[Fingerprint]:
        """Return all fingerprints in this set."""
        return self.fingerprints
    
    def __len__(self):
        return self.size()
    
    def verify_batch(self, query_response_pairs: List[tuple[str, str]]) -> List[bool]:
        """Verify multiple (query, response) pairs."""
        return [self.verify(query, response) for query, response in query_response_pairs]
    
    def sub_sample(self, n: int, random_seed: Optional[int] = None) -> 'FingerprintSet':
        """Sample n fingerprints from this set."""
        if random_seed is not None:
            random.seed(random_seed)
        
        fingerprints = self.get_fingerprints()
        if n >= len(fingerprints):
            return self
        
        sampled = random.sample(fingerprints, n)
        return FingerprintSet(sampled, name=f"{self.name}_sampled_{n}")
    
    def add_fingerprint(self, fingerprint: Fingerprint):
        """Add a new fingerprint to the set."""
        self.fingerprints.append(fingerprint)
    
    def remove_fingerprint(self, index: int) -> bool:
        """Remove fingerprint at the given index."""
        if 0 <= index < len(self.fingerprints):
            del self.fingerprints[index]
            return True
        return False
    
    def merge_with(self, other: 'FingerprintSet') -> 'FingerprintSet':
        """Merge this fingerprint set with another."""
        combined_fingerprints = self.fingerprints + other.fingerprints
        merged_metadata = {**self.metadata, **other.metadata}
        return FingerprintSet(
            combined_fingerprints, 
            name=f"{self.name}_merged_{other.name}",
            metadata=merged_metadata
        )
    
    def filter_by_type(self, verification_type: VerificationType) -> 'FingerprintSet':
        """Filter fingerprints by verification type."""
        filtered = [fp for fp in self.fingerprints if fp.fingerprint_type == verification_type]
        return FingerprintSet(
            filtered, 
            name=f"{self.name}_filtered_{verification_type.value}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "size": self.size(),
            "fingerprints": [self._fingerprint_to_dict(fp) for fp in self.fingerprints]
        }
    
    def _fingerprint_to_dict(self, fingerprint: Fingerprint) -> Dict[str, Any]:
        """Convert a single fingerprint to dictionary representation."""
        data = {
            "type": fingerprint.fingerprint_type.value if fingerprint.fingerprint_type else "unknown",
            "combination_strategy": fingerprint.combination_strategy.value,
            "created_at": fingerprint.created_at.isoformat(),
            "metadata": fingerprint.metadata,
            "verification_functions": []
        }
        
        for vf in fingerprint.verification_functions:
            vf_data = {
                "type": vf.verification_type.value,
                "expected_query": vf.expected_query
            }
            
            if hasattr(vf, 'expected_response'):
                vf_data["expected_response"] = vf.expected_response
            if hasattr(vf, 'required_tokens'):
                vf_data["required_tokens"] = vf.required_tokens
                vf_data["case_sensitive"] = vf.case_sensitive
            if hasattr(vf, 'pattern'):
                vf_data["pattern"] = vf.pattern
                if hasattr(vf, 'compiled_pattern'):
                    vf_data["flags"] = vf.compiled_pattern.flags
            
            data["verification_functions"].append(vf_data)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FingerprintSet':
        """Create from dictionary."""
        name = data.get("name", "unnamed")
        metadata = data.get("metadata", {})
        fingerprints = []
        
        for fp_data in data.get("fingerprints", []):
            fingerprint = cls._fingerprint_from_dict(fp_data)
            if fingerprint:
                fingerprints.append(fingerprint)
        
        return cls(fingerprints, name, metadata)
    
    @classmethod
    def _fingerprint_from_dict(cls, data: Dict[str, Any]) -> Optional[Fingerprint]:
        """Create a fingerprint from dictionary representation."""
        try:
            fp_type = VerificationType(data.get("type", "simple"))
            vf_data_list = data.get("verification_functions", [])
            
            if not vf_data_list:
                return None
            
            # For now, handle single verification function fingerprints
            if len(vf_data_list) == 1:
                vf_data = vf_data_list[0]
                expected_query = vf_data.get("expected_query", "")
                
                if fp_type == VerificationType.SIMPLE:
                    expected_response = vf_data.get("expected_response", "")
                    return SimpleFingerprint(expected_query, expected_response)
                elif fp_type == VerificationType.TOKEN_EXISTENCE:
                    required_tokens = vf_data.get("required_tokens", [])
                    case_sensitive = vf_data.get("case_sensitive", True)
                    return TokenExistenceFingerprint(expected_query, required_tokens, case_sensitive)
                elif fp_type == VerificationType.REGEX:
                    pattern = vf_data.get("pattern", "")
                    flags = vf_data.get("flags", 0)
                    return RegexFingerprint(expected_query, pattern, flags)
            
            return None
        except Exception:
            return None
    
    def save_to_file(self, file_path: str):
        """Save fingerprint set to JSON file."""
        if not file_path.endswith('.json'):
            file_path = f"{file_path}.json"
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'FingerprintSet':
        """Load fingerprint set from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


class SimpleFingerprintSet(FingerprintSet):
    """A fingerprint set that contains only simple fingerprints."""
    
    def __init__(self, fingerprints: List[Fingerprint], name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        # Validate that all fingerprints are simple type
        for fp in fingerprints:
            if fp.fingerprint_type != VerificationType.SIMPLE:
                raise ValueError(f"All fingerprints must be of type {VerificationType.SIMPLE.value}")
        
        super().__init__(fingerprints, name, metadata)
        self.verification_type = VerificationType.SIMPLE
    
    def add_fingerprint(self, fingerprint: Fingerprint):
        """Add a new fingerprint to the set."""
        if fingerprint.fingerprint_type != VerificationType.SIMPLE:
            raise ValueError(f"Fingerprint must be of type {VerificationType.SIMPLE.value}")
        super().add_fingerprint(fingerprint)


class TokenFingerprintSet(FingerprintSet):
    """A fingerprint set that contains only token existence fingerprints."""
    
    def __init__(self, fingerprints: List[Fingerprint], name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        # Validate that all fingerprints are token existence type
        for fp in fingerprints:
            if fp.fingerprint_type != VerificationType.TOKEN_EXISTENCE:
                raise ValueError(f"All fingerprints must be of type {VerificationType.TOKEN_EXISTENCE.value}")
        
        super().__init__(fingerprints, name, metadata)
        self.verification_type = VerificationType.TOKEN_EXISTENCE
    
    def add_fingerprint(self, fingerprint: Fingerprint):
        """Add a new fingerprint to the set."""
        if fingerprint.fingerprint_type != VerificationType.TOKEN_EXISTENCE:
            raise ValueError(f"Fingerprint must be of type {VerificationType.TOKEN_EXISTENCE.value}")
        super().add_fingerprint(fingerprint)


class RegexFingerprintSet(FingerprintSet):
    """A fingerprint set that contains only regex fingerprints."""
    
    def __init__(self, fingerprints: List[Fingerprint], name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        # Validate that all fingerprints are regex type
        for fp in fingerprints:
            if fp.fingerprint_type != VerificationType.REGEX:
                raise ValueError(f"All fingerprints must be of type {VerificationType.REGEX.value}")
        
        super().__init__(fingerprints, name, metadata)
        self.verification_type = VerificationType.REGEX
    
    def add_fingerprint(self, fingerprint: Fingerprint):
        """Add a new fingerprint to the set."""
        if fingerprint.fingerprint_type != VerificationType.REGEX:
            raise ValueError(f"Fingerprint must be of type {VerificationType.REGEX.value}")
        super().add_fingerprint(fingerprint)


# Factory functions for easy creation
def create_simple_fingerprint_set(
    query_response_pairs: List[tuple[str, str]],
    name: Optional[str] = None
) -> SimpleFingerprintSet:
    """Create a simple fingerprint set with simple fingerprints."""
    fingerprints = [
        SimpleFingerprint(query, response) 
        for query, response in query_response_pairs
    ]
    return SimpleFingerprintSet(
        fingerprints,
        name or "simple_set"
    )


def create_token_fingerprint_set(
    query_tokens_pairs: List[tuple[str, List[str]]],
    case_sensitive: bool = True,
    name: Optional[str] = None
) -> TokenFingerprintSet:
    """Create a token fingerprint set with token existence fingerprints."""
    fingerprints = [
        TokenExistenceFingerprint(query, tokens, case_sensitive)
        for query, tokens in query_tokens_pairs
    ]
    return TokenFingerprintSet(
        fingerprints,
        name or "token_set"
    )


def create_regex_fingerprint_set(
    query_pattern_pairs: List[tuple[str, str]],
    flags: int = 0,
    name: Optional[str] = None
) -> RegexFingerprintSet:
    """Create a regex fingerprint set with regex fingerprints."""
    fingerprints = [
        RegexFingerprint(query, pattern, flags)
        for query, pattern in query_pattern_pairs
    ]
    return RegexFingerprintSet(
        fingerprints,
        name or "regex_set"
    )


def create_fingerprint_set_from_config(config: Dict[str, Any]) -> FingerprintSet:
    """
    Create a fingerprint set from configuration.
    
    Config format:
    {
        "type": "simple" | "token_existence" | "regex" | "general",
        "name": str,
        "data": [...]  # Format depends on type
    }
    """
    fp_type = config.get("type", "simple")
    name = config.get("name", "config_set")
    data = config.get("data", [])
    
    if fp_type == "simple":
        pairs = [(item["query"], item["response"]) for item in data]
        return create_simple_fingerprint_set(pairs, name)
    elif fp_type == "token_existence":
        pairs = [(item["query"], item["tokens"]) for item in data]
        case_sensitive = config.get("case_sensitive", True)
        return create_token_fingerprint_set(pairs, case_sensitive, name)
    elif fp_type == "regex":
        pairs = [(item["query"], item["pattern"]) for item in data]
        flags = config.get("flags", 0)
        return create_regex_fingerprint_set(pairs, flags, name)
    elif fp_type == "general":
        # Mixed type fingerprint set
        fingerprints = []
        for item in data:
            item_type = item.get("type", "simple")
            query = item.get("query", "")
            
            if item_type == "simple":
                response = item.get("response", "")
                fingerprints.append(SimpleFingerprint(query, response))
            elif item_type == "token_existence":
                tokens = item.get("tokens", [])
                case_sensitive = item.get("case_sensitive", True)
                fingerprints.append(TokenExistenceFingerprint(query, tokens, case_sensitive))
            elif item_type == "regex":
                pattern = item.get("pattern", "")
                flags = item.get("flags", 0)
                fingerprints.append(RegexFingerprint(query, pattern, flags))
        
        return FingerprintSet(fingerprints, name)
    else:
        raise ValueError(f"Unknown fingerprint set type: {fp_type}")


def merge_fingerprint_sets(sets: List[FingerprintSet], name: Optional[str] = None) -> FingerprintSet:
    """
    Merge multiple fingerprint sets into one general set.
    
    Args:
        sets: List of fingerprint sets to merge
        name: Name for the merged set
        
    Returns:
        FingerprintSet containing all fingerprints
    """
    all_fingerprints = []
    merged_metadata = {}
    
    for fp_set in sets:
        all_fingerprints.extend(fp_set.get_fingerprints())
        merged_metadata.update(fp_set.metadata)
    
    return FingerprintSet(
        all_fingerprints, 
        name or "merged_set",
        merged_metadata
    )


# Utility functions
def fingerprint_set_statistics(fingerprint_set: FingerprintSet) -> Dict[str, Any]:
    """
    Compute statistics for a fingerprint set.
    
    Args:
        fingerprint_set: The fingerprint set to analyze
        
    Returns:
        Dictionary containing statistics
    """
    fingerprints = fingerprint_set.get_fingerprints()
    
    type_counts = {}
    query_lengths = []
    
    for fp in fingerprints:
        fp_type = fp.fingerprint_type.value if fp.fingerprint_type else "unknown"
        type_counts[fp_type] = type_counts.get(fp_type, 0) + 1
        
        # Get query from first verification function
        if fp.verification_functions:
            query_lengths.append(len(fp.verification_functions[0].expected_query))
    
    stats = {
        'total_fingerprints': len(fingerprints),
        'type_distribution': type_counts,
        'avg_query_length': sum(query_lengths) / len(query_lengths) if query_lengths else 0,
        'set_name': fingerprint_set.name,
        'created_at': fingerprint_set.created_at.isoformat() if hasattr(fingerprint_set, 'created_at') else None
    }
    
    return stats 