"""
Base Classes

Defines the core verification framework for fingerprint verification.

Base Classes:
    - VerificationFunction: Abstract base for custom verification methods

Built-in Implementations:
    - SimpleVerificationFunction, TokenExistenceVerificationFunction, RegexVerificationFunction

To add new verification types:
    1. Inherit from VerificationFunction
    2. Implement __call__(self, response: str) -> bool
    3. Implement get_query(self) -> str to return the associated query
"""

from typing import List, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod
import re


class VerificationType(Enum):
    """Types of verification methods supported."""
    SIMPLE = "simple"
    TOKEN_EXISTENCE = "token_existence"
    REGEX = "regex"


class VerificationFunction(ABC):
    """Base class for verification functions."""
    
    def __init__(self, verification_type: VerificationType, query: str):
        self.verification_type = verification_type
        self.query = query

    @abstractmethod
    def __call__(self, response: str) -> bool:
        """Verify if the response is valid for this verification function's query."""
        pass
    
    def get_query(self) -> str:
        """Return the query associated with this verification function."""
        return self.query
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationFunction':
        """Create instance from dictionary data."""
        pass


class SimpleVerificationFunction(VerificationFunction):
    """A verification function that checks exact match for a specific query-response pair."""
    
    def __init__(self, query: str, expected_response: str):
        super().__init__(VerificationType.SIMPLE, query)
        self.expected_response = expected_response.strip()
    
    def __call__(self, response: str) -> bool:
        """Check if response starts with expected response."""
        response = response.strip()
        
        # Handle edge case where response might be shorter than expected
        if len(response) < len(self.expected_response):
            return False
            
        return response[:len(self.expected_response)] == self.expected_response
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.verification_type.value,
            "query": self.query,
            "expected_response": self.expected_response
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleVerificationFunction':
        """Create instance from dictionary data."""
        return cls(data["query"], data["expected_response"])


class TokenExistenceVerificationFunction(VerificationFunction):
    """A verification function that checks if specific tokens exist in the response."""
    
    def __init__(self, query: str, required_tokens: List[str], case_sensitive: bool = True):
        super().__init__(VerificationType.TOKEN_EXISTENCE, query)
        self.required_tokens = required_tokens
        self.case_sensitive = case_sensitive
        
        if not required_tokens:
            raise ValueError("At least one token must be specified")
    
    def __call__(self, response: str) -> bool:
        """Check if all required tokens exist in the response."""
        if not self.case_sensitive:
            response = response.lower()
            tokens = [token.lower() for token in self.required_tokens]
        else:
            tokens = self.required_tokens
        
        return all(token in response for token in tokens)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.verification_type.value,
            "query": self.query,
            "required_tokens": self.required_tokens,
            "case_sensitive": self.case_sensitive
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenExistenceVerificationFunction':
        """Create instance from dictionary data."""
        return cls(
            data["query"], 
            data["required_tokens"], 
            data.get("case_sensitive", True)
        )


class RegexVerificationFunction(VerificationFunction):
    """A verification function that checks if response matches a regex pattern."""
    
    def __init__(self, query: str, pattern: str, flags: int = 0):
        super().__init__(VerificationType.REGEX, query)
        
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        
        self.pattern = pattern
        self.flags = flags
        try:
            self.compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    
    def __call__(self, response: str) -> bool:
        """Check if response matches the regex pattern."""
        return bool(self.compiled_pattern.search(response))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.verification_type.value,
            "query": self.query,
            "pattern": self.pattern,
            "flags": self.flags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegexVerificationFunction':
        """Create instance from dictionary data."""
        return cls(
            data["query"], 
            data["pattern"], 
            data.get("flags", 0)
        )


def verification_function_from_dict(data: Dict[str, Any]) -> VerificationFunction:
    """Factory function to create verification function from dictionary data."""
    func_type = data.get("type")
    
    if func_type == VerificationType.SIMPLE.value:
        return SimpleVerificationFunction.from_dict(data)
    elif func_type == VerificationType.TOKEN_EXISTENCE.value:
        return TokenExistenceVerificationFunction.from_dict(data)
    elif func_type == VerificationType.REGEX.value:
        return RegexVerificationFunction.from_dict(data)
    else:
        raise ValueError(f"Unknown verification function type: {func_type}")


