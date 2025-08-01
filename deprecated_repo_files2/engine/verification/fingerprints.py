"""
Fingerprint Implementations

This module implements fingerprint sets and additional fingerprint types for model verification,
building on the verification framework from base.py.

Creating New Fingerprint Types
==============================

There are two main approaches to create new fingerprint types:

1. **Create New Verification Functions** (in base.py):
   - Inherit from VerificationFunction
   - Set the 'type' class attribute to a VerificationType enum value
   - Implement __call__(self, response: str) -> bool
   - No method implementation needed in fingerprints.py - use generic Fingerprint class
   - Example: See SimpleVerificationFunction, TokenExistenceVerificationFunction, RegexVerificationFunction

2. **Create New Fingerprint Subclasses** (in this file):
   - Inherit from Fingerprint class
   - Combine existing verification functions using CombinationStrategy
   - No method implementation needed - inherit all functionality from Fingerprint
   - Example: See SimpleFingerprint, TokenExistenceFingerprint, RegexFingerprint

Example: Creating a Custom Verification Function
================================================

In base.py:
```python
class CustomVerificationFunction(VerificationFunction):
    type = VerificationType.CUSTOM  # Add to enum first
    
    def __init__(self, query: str, custom_param: str):
        super().__init__(query)
        self.custom_param = custom_param
    
    def __call__(self, response: str) -> bool:
        # Your verification logic here
        return self.custom_param in response
```

In fingerprints.py:
```python
class CustomFingerprint(Fingerprint):
    def __init__(self, query: str, custom_param: str, **kwargs):
        super().__init__(
            combination_strategy=CombinationStrategy.SINGLE,
            verification_functions=[CustomVerificationFunction(query, custom_param)],
            fingerprint_type=VerificationType.CUSTOM,
            **kwargs
        )
```

Example: Combining Existing Verification Functions
=================================================

```python
class MultiVerificationFingerprint(Fingerprint):
    def __init__(self, query: str, expected_response: str, required_tokens: List[str], **kwargs):
        super().__init__(
            combination_strategy=CombinationStrategy.INTERSECT,  # Both must pass
            verification_functions=[
                SimpleVerificationFunction(query, expected_response),
                TokenExistenceVerificationFunction(query, required_tokens)
            ],
            fingerprint_type=None,  # Mixed type
            **kwargs
        )
```

No method implementation is needed in fingerprint subclasses - all verification logic
is handled by the verification functions and the combination strategy.
"""

from typing import List, Dict, Any, Optional, Set, Iterator, Union
import json
import random
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, ConfigDict, field_validator, Field, model_validator

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
    
    Creating Fingerprint Subclasses:
    -------------------------------
    When creating new fingerprint types, you only need to define __init__().
    No method implementation is required - all verification logic is handled
    by the verification functions and combination strategy. Simply call
    super().__init__() with the appropriate verification_functions list.
    
    Example:
    ```python
    class MyFingerprint(Fingerprint):
        def __init__(self, query: str, my_param: str, **kwargs):
            super().__init__(
                verification_functions=[MyVerificationFunction(query=query, my_param=my_param)],
                combination_strategy=CombinationStrategy.SINGLE,
                **kwargs
            )
    ```
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    combination_strategy: CombinationStrategy = CombinationStrategy.SINGLE
    verification_functions: List[VerificationFunction] = Field(default_factory=list)
    fingerprint_type: Optional[VerificationType] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('combination_strategy', mode='before')
    @classmethod
    def validate_combination_strategy(cls, v):
        """Convert string values to enum if needed."""
        if isinstance(v, str):
            return CombinationStrategy(v)
        return v
    
    @field_validator('fingerprint_type', mode='before')
    @classmethod
    def validate_fingerprint_type(cls, v):
        """Convert string values to enum if needed."""
        if isinstance(v, str):
            return VerificationType(v)
        return v
    
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

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization to handle enums and verification functions properly."""
        data = super().model_dump(**kwargs)
        
        # Convert enums to their values
        if isinstance(data.get('combination_strategy'), CombinationStrategy):
            data['combination_strategy'] = data['combination_strategy'].value
        elif hasattr(data.get('combination_strategy'), 'value'):
            data['combination_strategy'] = data['combination_strategy'].value
            
        if isinstance(data.get('fingerprint_type'), VerificationType):
            data['fingerprint_type'] = data['fingerprint_type'].value
        elif hasattr(data.get('fingerprint_type'), 'value'):
            data['fingerprint_type'] = data['fingerprint_type'].value
            
        # Serialize verification functions with type information
        if 'verification_functions' in data:
            data['verification_functions'] = [
                {**vf.model_dump(), "type": vf.type.value}
                for vf in self.verification_functions
            ]
        
        return data

    def get_query(self) -> str:
        """Get the query associated with this fingerprint."""
        if not self.verification_functions:
            raise ValueError("No verification functions defined")
        return self.verification_functions[0].get_query()
    
    def verify(self, response: str) -> bool:
        """Verify a fingerprint against a response."""
        if not self.verification_functions:
            raise ValueError("No verification functions defined")
            
        if self.combination_strategy == CombinationStrategy.SINGLE:
            return self.verification_functions[0](response)
        elif self.combination_strategy == CombinationStrategy.UNION:
            return any(f(response) for f in self.verification_functions)
        elif self.combination_strategy == CombinationStrategy.INTERSECT:
            return all(f(response) for f in self.verification_functions)
        else:
            raise ValueError(f"Unknown combination strategy: {self.combination_strategy}")


class SimpleFingerprint(Fingerprint):
    """A fingerprint with a single simple verification function."""
    
    def __init__(self, query: str, expected_response: str, **kwargs):
        super().__init__(
            combination_strategy=CombinationStrategy.SINGLE,
            verification_functions=[SimpleVerificationFunction(query=query, expected_response=expected_response)],
            fingerprint_type=VerificationType.SIMPLE,
            **kwargs
        )


class TokenExistenceFingerprint(Fingerprint):
    """A fingerprint with a single token existence verification function."""
    
    def __init__(self, query: str, required_tokens: List[str], case_sensitive: bool = True, **kwargs):
        super().__init__(
            combination_strategy=CombinationStrategy.SINGLE,
            verification_functions=[TokenExistenceVerificationFunction(query=query, required_tokens=required_tokens, case_sensitive=case_sensitive)],
            fingerprint_type=VerificationType.TOKEN_EXISTENCE,
            **kwargs
        )


class RegexFingerprint(Fingerprint):
    """A fingerprint with a single regex verification function."""
    
    def __init__(self, query: str, pattern: str, flags: int = 0, **kwargs):
        super().__init__(
            combination_strategy=CombinationStrategy.SINGLE,
            verification_functions=[RegexVerificationFunction(query=query, pattern=pattern, flags=flags)],
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
    # Store fingerprints as a dict for O(1) lookup by query
    fingerprints_dict: Dict[str, Fingerprint] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def _process_fingerprints_input(cls, data):
        """Handle both list/set input and dict input for fingerprints."""
        if isinstance(data, dict):
            # Handle fingerprints as list/set in input
            if 'fingerprints' in data and isinstance(data['fingerprints'], (list, set)):
                fingerprints_list = data.pop('fingerprints')
                fingerprints_dict = {}
                for fp in fingerprints_list:
                    if isinstance(fp, dict):
                        # Convert dict to Fingerprint if needed
                        fp_data = fp.copy()
                        # Handle verification functions
                        if 'verification_functions' in fp_data:
                            vf_data_list = fp_data['verification_functions']
                            fp_data['verification_functions'] = [
                                VerificationFunction.from_dict(vf_data) if isinstance(vf_data, dict) else vf_data
                                for vf_data in vf_data_list
                            ]
                        fp = Fingerprint(**fp_data)
                    query = fp.get_query()
                    if query in fingerprints_dict:
                        raise ValueError(f"Duplicate fingerprint for query: {query!r}")
                    fingerprints_dict[query] = fp
                
                data['fingerprints_dict'] = fingerprints_dict
                    
                # Set created_at to max of fingerprints if not specified
                if fingerprints_dict and 'created_at' not in data:
                    data['created_at'] = max(fp.created_at for fp in fingerprints_dict.values())
                    
        return data

    def __init__(self, fingerprints: Optional[Union[List[Fingerprint], Set[Fingerprint]]] = None, **kwargs):
        """Initialize FingerprintSet with fingerprints and other parameters."""
        if fingerprints is not None:
            kwargs['fingerprints'] = fingerprints
        super().__init__(**kwargs)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization to handle fingerprints_dict properly."""
        data = super().model_dump(**kwargs)
        # Convert fingerprints_dict to a list for JSON serialization
        # Access fingerprints directly from instance, not from serialized data
        data['fingerprints'] = [fp.model_dump() for fp in self.fingerprints_dict.values()]
        data.pop('fingerprints_dict', None)
        return data

    def __len__(self) -> int:
        return len(self.fingerprints_dict)
    
    def __iter__(self) -> Iterator[Fingerprint]:
        return iter(self.fingerprints_dict.values())
    
    def __contains__(self, query: str) -> bool:
        return query in self.fingerprints_dict

    def verify(self, query: str, response: str) -> bool:
        """Return True if the fingerprint for this query verifies the response."""
        fingerprint = self.fingerprints_dict.get(query)
        return fingerprint.verify(response) if fingerprint else False

    @property
    def queries(self) -> Set[str]:
        """Return a set of all fingerprint queries."""
        return set(self.fingerprints_dict.keys())
    
    @property
    def fingerprints(self) -> List[Fingerprint]:
        """Return a list of all fingerprints."""
        return list(self.fingerprints_dict.values())
    
    def get_fingerprint_by_query(self, query: str) -> Optional[Fingerprint]:
        """Get a specific fingerprint by its query."""
        return self.fingerprints_dict.get(query)
    
    def sub_sample(self, n: int, random_seed: Optional[int] = None) -> 'FingerprintSet':
        """Sample uniformly n fingerprints from this set."""
        if random_seed is not None:
            random.seed(random_seed)
        
        fingerprints = self.fingerprints
        if n >= len(fingerprints):
            return self
        
        sampled = random.sample(fingerprints, n)
        return FingerprintSet(sampled, name=f"{self.name}_sampled_{n}")
    
    def add_fingerprint(self, fingerprint: Fingerprint):
        """Add a new fingerprint to the set."""
        query = fingerprint.get_query()
        if query in self.fingerprints_dict:
            raise ValueError(f"Duplicate fingerprint for query: {query!r}")
        self.fingerprints_dict[query] = fingerprint
    
    def remove_fingerprint(self, query_or_fingerprint: Union[str, Fingerprint]) -> bool:
        """Remove a fingerprint by query string or fingerprint object."""
        if isinstance(query_or_fingerprint, str):
            query = query_or_fingerprint
        else:
            query = query_or_fingerprint.get_query()
            
        if query in self.fingerprints_dict:
            del self.fingerprints_dict[query]
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
        filtered_fps = [fp for fp in self.fingerprints_dict.values() 
                       if fp.fingerprint_type == verification_type]
        
        type_name = verification_type.value if verification_type else "no_type"
        return FingerprintSet(
            filtered_fps, 
            name=f"{self.name}_filtered_{type_name}"
        )

    def save_to_file(self, file_path: str):
        """Save fingerprint set to JSON file."""
        if not file_path.endswith('.json'):
            file_path = f"{file_path}.json"
        
        with open(file_path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'FingerprintSet':
        """Load fingerprint set from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.model_validate(data)

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
        query_lengths.append(len(fp.get_query()))
    
    stats = {
        'total_fingerprints': len(fingerprint_set),
        'type_distribution': type_counts,
        'avg_query_length': sum(query_lengths) / len(query_lengths) if query_lengths else 0,
        'set_name': fingerprint_set.name,
        'created_at': fingerprint_set.created_at.isoformat()
    }
    
    return stats 