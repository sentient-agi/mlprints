"""
Base Classes

Defines the core verification framework for fingerprint verification.

Base Classes:
    - VerificationFunction: Abstract base for custom verification methods

Built-in Implementations:
    - SimpleVerificationFunction, TokenExistenceVerificationFunction, RegexVerificationFunction

To add new verification types:
    1. Inherit from VerificationFunction
    2. Implement __call__(self, query: str, response: str) -> bool
    3. Use @query_checked decorator for automatic query validation
"""

from typing import List, Callable
from enum import Enum
from abc import ABC, abstractmethod
import re
from functools import wraps

class VerificationType(Enum):
    """Types of verification methods supported."""
    SIMPLE = "simple"
    TOKEN_EXISTENCE = "token_existence"
    REGEX = "regex"

class CombinationStrategy(Enum):
    """How multiple verification functions are combined."""
    SINGLE = "single"      # Use only the first function
    UNION = "union"        # At least one function must pass (OR logic)
    INTERSECT = "intersect"  # All functions must pass (AND logic)


def query_checked(func: Callable) -> Callable:
    """Decorator that automatically handles query checking for verification functions."""
    @wraps(func)
    def wrapper(self, query: str, response: str) -> bool:
        # Automatic query validation
        if not hasattr(self, 'expected_query'):
            raise AttributeError("Verification function must have 'expected_query' attribute")
        
        if query.strip() != self.expected_query:
            return False
        
        # Call the original function if query matches
        return func(self, query, response)
    
    return wrapper


class VerificationFunction(ABC):
    """Base class for verification functions."""
    
    def __init__(self, verification_type: VerificationType, expected_query: str):
        self.verification_type = verification_type
        self.expected_query = expected_query

    @abstractmethod
    def __call__(self, query: str, response: str) -> bool:
        pass


class SimpleVerificationFunction(VerificationFunction):
    """A verification function that checks exact match for a specific query-response pair."""
    
    def __init__(self, expected_query: str, expected_response: str):
        super().__init__(VerificationType.SIMPLE, expected_query)
        self.expected_response = expected_response.strip()
    
    @query_checked
    def __call__(self, query: str, response: str) -> bool:
        """Check if response starts with expected response."""
        response = response.strip()
        
        # Handle edge case where response might be shorter than expected
        if len(response) < len(self.expected_response):
            return False
            
        return response[:len(self.expected_response)] == self.expected_response


class TokenExistenceVerificationFunction(VerificationFunction):
    """A verification function that checks if specific tokens exist in the response."""
    
    def __init__(self, expected_query: str, required_tokens: List[str], case_sensitive: bool = True):
        super().__init__(VerificationType.TOKEN_EXISTENCE, expected_query)
        self.required_tokens = required_tokens
        self.case_sensitive = case_sensitive
        
        if not required_tokens:
            raise ValueError("At least one token must be specified")
    
    @query_checked
    def __call__(self, query: str, response: str) -> bool:
        """Check if all required tokens exist in the response."""
        if not self.case_sensitive:
            response = response.lower()
            tokens = [token.lower() for token in self.required_tokens]
        else:
            tokens = self.required_tokens
        
        return all(token in response for token in tokens)


class RegexVerificationFunction(VerificationFunction):
    """A verification function that checks if response matches a regex pattern."""
    
    def __init__(self, expected_query: str, pattern: str, flags: int = 0):
        super().__init__(VerificationType.REGEX, expected_query)
        
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        
        self.pattern = pattern
        try:
            self.compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    
    @query_checked
    def __call__(self, query: str, response: str) -> bool:
        """Check if response matches the regex pattern."""
        return bool(self.compiled_pattern.search(response))


