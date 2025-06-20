"""
Fingerprint Implementations

This module implements fingerprint sets and additional fingerprint types for model verification,
building on the verification framework from base.py.
"""

# Removed ABC import as FingerprintSet is no longer abstract
from typing import List, Dict, Any, Optional, Set
import json
import random
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, ConfigDict, model_validator, Field

from .base import (
    VerificationType, 
    VerificationFunction,
    SimpleVerificationFunction,
    TokenExistenceVerificationFunction,
    RegexVerificationFunction,
    verification_function_from_dict
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
    
    @model_validator(mode='after')
    def _validate_verification_functions(self) -> 'Fingerprint':
        """Validate that the combination strategy matches the number of functions and all functions have the same query."""
        num_functions = len(self.verification_functions)
        
        if self.combination_strategy == CombinationStrategy.SINGLE:
            if num_functions != 1:
                raise ValueError("Single combination strategy requires exactly one verification function")
        elif self.combination_strategy in [CombinationStrategy.UNION, CombinationStrategy.INTERSECT]:
            if num_functions < 2:
                raise ValueError(f"{self.combination_strategy.value} combination strategy requires at least two verification functions")
        
        # Validate that all verification functions have the same query
        if num_functions > 1:
            query_set = set([func.get_query() for func in self.verification_functions])
            if len(query_set) > 1:
                raise ValueError("All verification functions in a fingerprint must have the same query")
        
        return self

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
            if self.combination_strategy == CombinationStrategy.SINGLE:
                return self.verification_functions[0](response)
            elif self.combination_strategy == CombinationStrategy.UNION:
                return any(func(response) for func in self.verification_functions)
            elif self.combination_strategy == CombinationStrategy.INTERSECT:
                return all(func(response) for func in self.verification_functions)
            else:
                raise ValueError(f"Unknown combination strategy: {self.combination_strategy}")
                
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


class FingerprintSet:
    """A general fingerprint set that can contain any types of fingerprints.
    
    A fingerprint set contains multiple Fingerprint objects and provides
    collective verification capabilities. Uses internal dict for O(1) lookup
    and ensures no duplicate queries.
    """

    def __init__(self, fingerprints: Set[Fingerprint] | List[Fingerprint],
                 name: Optional[str] = "unnamed_set",
                 created_at: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        # Main storage: map query -> Fingerprint for O(1) lookup
        self.fingerprints_map: Dict[str, Fingerprint] = {}
        self.name = name
        self.metadata = metadata
        
        # Add fingerprints with duplicate checking
        if fingerprints:
            for fp in fingerprints:
                self.add_fingerprint(fp)
        
        # Set created_at to max of fingerprints or current time
        if self.fingerprints_map:
            self.created_at = max(fp.created_at for fp in self.fingerprints_map.values())
        else:
            self.created_at = created_at or datetime.now()

    @property
    def fingerprints(self) -> Set[Fingerprint]:
        """Return fingerprints as a set."""
        return set(self.fingerprints_map.values())

    def size(self) -> int:
        """Get the number of fingerprints in this set."""
        return len(self.fingerprints_map)

    def verify(self, query: str, response: str) -> bool:
        """Return True if the fingerprint for this query verifies the response."""
        fingerprint = self.fingerprints_map.get(query)
        return fingerprint.verify(response) if fingerprint else False

    def get_queries(self) -> Set[str]:
        """Return a set of all fingerprint queries."""
        return set(self.fingerprints_map.keys())
    
    def get_fingerprints(self) -> Set[Fingerprint]:
        """Return a set of all fingerprints."""
        return set(self.fingerprints_map.values())
    
    def get_fingerprint_by_query(self, query: str) -> Optional[Fingerprint]:
        """Get a specific fingerprint by its query."""
        return self.fingerprints_map.get(query)
    
    def has_query(self, query: str) -> bool:
        """Check if a fingerprint exists for the given query."""
        return query in self.fingerprints_map
    
    def __len__(self):
        return self.size()
    
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
    
    def remove_fingerprint(self, fingerprint: Fingerprint) -> bool:
        """Remove the given fingerprint from the set."""
        query = fingerprint.get_query()
        if query in self.fingerprints_map and self.fingerprints_map[query] == fingerprint:
            del self.fingerprints_map[query]
            return True
        return False
    
    def remove_fingerprint_by_query(self, query: str) -> bool:
        """Remove a fingerprint by its query."""
        if query in self.fingerprints_map:
            del self.fingerprints_map[query]
            return True
        return False
    
    def merge_with(self, other: 'FingerprintSet') -> 'FingerprintSet':
        """Merge this fingerprint set with another."""
        merged_metadata = {**self.metadata, **other.metadata}
        merged_set = FingerprintSet(
            name=f"{self.name}_merged_{other.name}",
            metadata=merged_metadata
        )
        
        # Add all fingerprints from both sets
        for fp in self.get_fingerprints():
            merged_set.add_fingerprint(fp)
        
        for fp in other.get_fingerprints():
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "size": self.size(),
            "fingerprints": [self._fingerprint_to_dict(fp) for fp in self.fingerprints_map.values()]
        }
    
    def _fingerprint_to_dict(self, fp: Fingerprint) -> Dict[str, Any]:
        """Convert a fingerprint to dictionary with proper verification function serialization."""
        fp_dict = fp.model_dump()
        # Replace verification_functions with serializable versions
        fp_dict['verification_functions'] = [vf.to_dict() for vf in fp.verification_functions]
        # Convert datetime to ISO string
        if isinstance(fp_dict.get('created_at'), datetime):
            fp_dict['created_at'] = fp_dict['created_at'].isoformat()
        # Convert enums to string values
        if isinstance(fp_dict.get('combination_strategy'), CombinationStrategy):
            fp_dict['combination_strategy'] = fp_dict['combination_strategy'].value
        if isinstance(fp_dict.get('fingerprint_type'), VerificationType):
            fp_dict['fingerprint_type'] = fp_dict['fingerprint_type'].value
        return fp_dict
    
    def _fingerprint_from_dict(self, fp_data: Dict[str, Any]) -> Fingerprint:
        """Create a fingerprint from dictionary data with proper verification function deserialization."""
        # Convert verification functions back to objects
        vf_data_list = fp_data.get('verification_functions', [])
        verification_functions = [verification_function_from_dict(vf_data) for vf_data in vf_data_list]
        
        # Create fingerprint with proper verification functions
        fp_data_copy = fp_data.copy()
        fp_data_copy['verification_functions'] = verification_functions
        
        # Convert ISO string back to datetime if needed
        if isinstance(fp_data_copy.get('created_at'), str):
            fp_data_copy['created_at'] = datetime.fromisoformat(fp_data_copy['created_at'])
        
        # Convert enum strings back to enums if needed
        if isinstance(fp_data_copy.get('combination_strategy'), str):
            fp_data_copy['combination_strategy'] = CombinationStrategy(fp_data_copy['combination_strategy'])
        if isinstance(fp_data_copy.get('fingerprint_type'), str):
            fp_data_copy['fingerprint_type'] = VerificationType(fp_data_copy['fingerprint_type'])
        
        return Fingerprint(**fp_data_copy)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FingerprintSet':
        """Create a FingerprintSet from dictionary data."""
        fingerprint_set = cls(
            fingerprints=[],
            name=data.get('name', 'unnamed_set'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            metadata=data.get('metadata', {})
        )
        
        # Add fingerprints using the helper method
        for fp_data in data.get('fingerprints', []):
            fingerprint = fingerprint_set._fingerprint_from_dict(fp_data)
            fingerprint_set.add_fingerprint(fingerprint)
        
        return fingerprint_set
    
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
        
        # Get query from fingerprint
        if fp.verification_functions:
            query_lengths.append(len(fp.get_query()))
    
    stats = {
        'total_fingerprints': len(fingerprints),
        'type_distribution': type_counts,
        'avg_query_length': sum(query_lengths) / len(query_lengths) if query_lengths else 0,
        'set_name': fingerprint_set.name,
        'created_at': fingerprint_set.created_at.isoformat() if hasattr(fingerprint_set, 'created_at') else None
    }
    
    return stats 