"""
Fingerprint Implementations

This module implements fingerprint sets and additional fingerprint types for model verification,
building on the verification framework from base.py.
"""

from typing import List, Dict, Any, Optional, Set, Iterator
import json
import random
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, ConfigDict, field_validator, Field

from .base import (
    VerificationType, 
    VerificationFunction,
    SimpleVerificationFunction,
    TokenExistenceVerificationFunction,
    RegexVerificationFunction
)

class CombinationStrategy(Enum):
    """How multiple verification functions are combined."""
    SINGLE = "single"      # Use only the first function
    UNION = "union"        # At least one function must pass (OR logic)
    INTERSECT = "intersect"  # All functions must pass (AND logic)


class Fingerprint(BaseModel):
    """A single fingerprint consisting of one-or-more verification functions.
    
    Note: Different functions could theoretically have different queries,
    which is why query is stored at the VerificationFunction level rather
    than at the Fingerprint level.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    combination_strategy: CombinationStrategy = CombinationStrategy.SINGLE
    verification_functions: List[VerificationFunction] = Field(default_factory=list)
    fingerprint_type: Optional[VerificationType] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('verification_functions', mode='after')
    @classmethod
    def _validate_verification_functions(cls, v, info):
        """Validate that the combination strategy matches the number of functions and all functions have the same query."""
        combination_strategy = info.data.get('combination_strategy', CombinationStrategy.SINGLE)
        num_functions = len(v)
        
        if combination_strategy == CombinationStrategy.SINGLE:
            if num_functions != 1:
                raise ValueError("Single combination strategy requires exactly one verification function")
        elif combination_strategy in [CombinationStrategy.UNION, CombinationStrategy.INTERSECT]:
            if num_functions < 2:
                raise ValueError(f"{combination_strategy.value} combination strategy requires at least two verification functions")
        
        # Validate that all verification functions have the same query
        if num_functions > 1:
            query_set = set([func.get_query() for func in v])
            if len(query_set) > 1:
                raise ValueError("All verification functions in a fingerprint must have the same query")
        
        return v

    def get_query(self) -> str:
        """Get the query associated with this fingerprint."""
        if not self.verification_functions:
            raise ValueError("No verification functions defined")
        return self.verification_functions[0].get_query()
    
    @property
    def query(self) -> str:
        """Convenience property to access the query."""
        return self.get_query()

    def verify(self, response: str) -> bool:
        """Verify a fingerprint against a response."""
        if not self.verification_functions:
            raise ValueError("No verification functions defined")
            
        try:
            strategy_fn = {
                CombinationStrategy.SINGLE: lambda: self.verification_functions[0](response),
                CombinationStrategy.UNION: lambda: any(f(response) for f in self.verification_functions),
                CombinationStrategy.INTERSECT: lambda: all(f(response) for f in self.verification_functions),
            }[self.combination_strategy]
            return strategy_fn()
                
        except Exception as e:
            raise RuntimeError(f"Verification failed: {e}") from e


class SimpleFingerprint(Fingerprint):
    """A fingerprint with a single simple verification function."""
    
    def __init__(self, query: str, expected_response: str, **kwargs):
        super().__init__(
            combination_strategy=CombinationStrategy.SINGLE,
            verification_functions=[SimpleVerificationFunction(query, expected_response)],
            fingerprint_type=VerificationType.SIMPLE,
            **kwargs
        )


class TokenExistenceFingerprint(Fingerprint):
    """A fingerprint with a single token existence verification function."""
    
    def __init__(self, query: str, required_tokens: List[str], case_sensitive: bool = True, **kwargs):
        super().__init__(
            combination_strategy=CombinationStrategy.SINGLE,
            verification_functions=[TokenExistenceVerificationFunction(query, required_tokens, case_sensitive)],
            fingerprint_type=VerificationType.TOKEN_EXISTENCE,
            **kwargs
        )


class RegexFingerprint(Fingerprint):
    """A fingerprint with a single regex verification function."""
    
    def __init__(self, query: str, pattern: str, flags: int = 0, **kwargs):
        super().__init__(
            combination_strategy=CombinationStrategy.SINGLE,
            verification_functions=[RegexVerificationFunction(query, pattern, flags)],
            fingerprint_type=VerificationType.REGEX,
            **kwargs
        )


class FingerprintSet(BaseModel):
    """A general fingerprint set that can contain any types of fingerprints.
    
    A fingerprint set contains multiple Fingerprint objects and provides
    collective verification capabilities. Uses internal dict for O(1) lookup
    and ensures no duplicate queries.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "unnamed_set"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    fingerprints_map: Dict[str, Fingerprint] = Field(default_factory=dict)

    def __init__(self, fingerprints: Set[Fingerprint] | List[Fingerprint] | None = None, **kwargs):
        # Initialize with empty fingerprints_map first
        super().__init__(**kwargs)
        
        # Add fingerprints with duplicate checking
        if fingerprints:
            for fp in fingerprints:
                self.add_fingerprint(fp)
        
        # Set created_at to max of fingerprints or current time if not explicitly set
        if self.fingerprints_map and 'created_at' not in kwargs:
            self.created_at = max(fp.created_at for fp in self.fingerprints_map.values())

    def __len__(self) -> int:
        return len(self.fingerprints_map)
    
    def __iter__(self) -> Iterator[Fingerprint]:
        return iter(self.fingerprints_map.values())
    
    def __contains__(self, query: str) -> bool:
        return query in self.fingerprints_map

    def size(self) -> int:
        """Get the number of fingerprints in this set."""
        return len(self.fingerprints_map)

    def verify(self, query: str, response: str) -> bool:
        """Return True if the fingerprint for this query verifies the response."""
        fingerprint = self.fingerprints_map.get(query)
        return fingerprint.verify(response) if fingerprint else False

    def get_queries(self) -> Set[str]:
        """Return a set of all fingerprint queries."""
        return set(self.fingerprints_map)
    
    def get_fingerprints(self) -> List[Fingerprint]:
        """Return a list of all fingerprints."""
        return list(self.fingerprints_map.values())
    
    def get_fingerprint_by_query(self, query: str) -> Optional[Fingerprint]:
        """Get a specific fingerprint by its query."""
        return self.fingerprints_map.get(query)
    
    def has_query(self, query: str) -> bool:
        """Check if a fingerprint exists for the given query."""
        return query in self.fingerprints_map
    
    def sub_sample(self, n: int, random_seed: Optional[int] = None) -> 'FingerprintSet':
        """Sample uniformly n fingerprints from this set."""
        if random_seed is not None:
            random.seed(random_seed)
        
        fingerprints = self.get_fingerprints()
        if n >= len(fingerprints):
            return self
        
        sampled = random.sample(fingerprints, n)
        return FingerprintSet(sampled, name=f"{self.name}_sampled_{n}")
    
    def add_fingerprint(self, fingerprint: Fingerprint):
        """Add a new fingerprint to the set."""
        query = fingerprint.get_query()
        if query in self.fingerprints_map:
            raise ValueError(f"Duplicate fingerprint for query: {query!r}")
        self.fingerprints_map[query] = fingerprint
    
    def remove_fingerprint_by_query(self, query: str) -> bool:
        """Remove a fingerprint by its query."""
        if query in self.fingerprints_map:
            del self.fingerprints_map[query]
            return True
        return False
    
    def remove_fingerprint(self, fingerprint: Fingerprint) -> bool:
        """Remove the given fingerprint from the set."""
        return self.remove_fingerprint_by_query(fingerprint.get_query())
    
    def merge_with(self, other: 'FingerprintSet') -> 'FingerprintSet':
        """Merge this fingerprint set with another."""
        merged_metadata = {**self.metadata, **other.metadata}
        merged_set = FingerprintSet(
            fingerprints=[],
            name=f"{self.name}_merged_{other.name}",
            metadata=merged_metadata
        )
        
        # Add all fingerprints from both sets
        for fp in self:
            merged_set.add_fingerprint(fp)
        
        for fp in other:
            try:
                merged_set.add_fingerprint(fp)
            except ValueError:
                # Skip duplicates - fingerprints with same query already exist
                pass
        
        return merged_set
    
    def filter_by_type(self, verification_type: Optional[VerificationType]) -> 'FingerprintSet':
        """Filter fingerprints by verification type. Pass None to get fingerprints with no type."""
        filtered_fps = [fp for fp in self.fingerprints_map.values() 
                       if fp.fingerprint_type == verification_type]
        
        type_name = verification_type.value if verification_type else "no_type"
        return FingerprintSet(
            filtered_fps, 
            name=f"{self.name}_filtered_{type_name}"
        )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to handle VerificationFunction serialization."""
        data = super().model_dump(**kwargs)
        
        # Convert fingerprints_map values to serializable format
        serialized_fingerprints = []
        for fp in self.fingerprints_map.values():
            fp_dict = fp.model_dump(mode="json")
            fp_dict["verification_functions"] = [vf.to_dict() for vf in fp.verification_functions]
            serialized_fingerprints.append(fp_dict)
        
        data["fingerprints"] = serialized_fingerprints
        data.pop("fingerprints_map", None)  # Remove the internal map from serialization
        
        return data
    
    @classmethod
    def model_validate(cls, data: Dict[str, Any], **kwargs) -> 'FingerprintSet':
        """Override to handle VerificationFunction deserialization."""
        if isinstance(data, dict) and "fingerprints" in data:
            # Convert fingerprints back to objects
            fingerprints = []
            for fp_data in data.get("fingerprints", []):
                fp_data_copy = fp_data.copy()
                
                # Restore VerificationFunction instances
                vf_data_list = fp_data_copy.pop("verification_functions", [])
                fp_data_copy["verification_functions"] = [
                    VerificationFunction.from_dict(vf) for vf in vf_data_list
                ]
                
                fingerprints.append(Fingerprint(**fp_data_copy))
            
            # Create new data dict without fingerprints for base model validation
            model_data = {k: v for k, v in data.items() if k != "fingerprints"}
            return cls(fingerprints=fingerprints, **model_data)
        
        return super().model_validate(data, **kwargs)
    
    def save_to_file(self, file_path: str):
        """Save fingerprint set to JSON file."""
        if not file_path.endswith('.json'):
            file_path = f"{file_path}.json"
        
        with open(file_path, 'w') as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'FingerprintSet':
        """Load fingerprint set from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.model_validate(data)


# Utility functions
def fingerprint_set_statistics(fingerprint_set: FingerprintSet) -> Dict[str, Any]:
    """
    Compute statistics for a fingerprint set.
    
    Args:
        fingerprint_set: The fingerprint set to analyze
        
    Returns:
        Dictionary containing statistics
    """
    type_counts = {}
    query_lengths = []
    
    for fp in fingerprint_set:
        fp_type = fp.fingerprint_type.value if fp.fingerprint_type else "unknown"
        type_counts[fp_type] = type_counts.get(fp_type, 0) + 1
        
        # Get query from fingerprint
        if fp.verification_functions:
            query_lengths.append(len(fp.get_query()))
    
    stats = {
        'total_fingerprints': len(fingerprint_set),
        'type_distribution': type_counts,
        'avg_query_length': sum(query_lengths) / len(query_lengths) if query_lengths else 0,
        'set_name': fingerprint_set.name,
        'created_at': fingerprint_set.created_at.isoformat()
    }
    
    return stats 