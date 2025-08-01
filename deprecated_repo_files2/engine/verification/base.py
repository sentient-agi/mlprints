"""
Base Classes for Fingerprint Verification

Defines the core verification framework for fingerprint verification.

Base Classes:
    - VerificationFunction: Abstract base class for all verification methods

Built-in Implementations:
    - SimpleVerificationFunction: Exact match verification
    - TokenExistenceVerificationFunction: Token presence verification  
    - RegexVerificationFunction: Regular expression pattern matching

Usage:
    To add new verification types:
    0. Add the new verification type to the VerificationType enum
    1. Inherit from VerificationFunction
    2. Set the 'type' class attribute to a VerificationType enum value
    3. Implement __call__(self, response: str) -> bool
    
    The verification function automatically registers itself and can be serialized/deserialized
    using Pydantic's built-in capabilities.

Creating New Verification Functions - Step by Step
=================================================

1. **Add to VerificationType enum**:
   ```python
   class VerificationType(Enum):
       SIMPLE = "simple"
       TOKEN_EXISTENCE = "token_existence" 
       REGEX = "regex"
       YOUR_NEW_TYPE = "your_new_type"  # Add this
   ```

2. **Create the verification function class**:
   ```python
   class YourVerificationFunction(VerificationFunction):
       type: VerificationType = VerificationType.YOUR_NEW_TYPE
       
       query: str
       your_param: Any  # Add your custom parameters as Pydantic fields
       
       def __call__(self, response: str) -> bool:
           # Implement your verification logic here
           # Return True if response passes verification, False otherwise
           pass
   ```

3. **Use in fingerprints.py** (no additional implementation needed):
   ```python
   # Generic usage
   fingerprint = Fingerprint(
       verification_functions=[YourVerificationFunction(query=query, your_param=param)],
       combination_strategy=CombinationStrategy.SINGLE
   )
   
   # Or create a convenience subclass
   class YourFingerprint(Fingerprint):
       def __init__(self, query: str, your_param: Any, **kwargs):
           super().__init__(
               verification_functions=[YourVerificationFunction(query=query, your_param=your_param)],
               combination_strategy=CombinationStrategy.SINGLE,
               fingerprint_type=VerificationType.YOUR_NEW_TYPE,
               **kwargs
           )
   ```

The verification function automatically registers itself and uses Pydantic's built-in
serialization without any additional code.
"""

from typing import List, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod
import re
from pydantic import BaseModel, ConfigDict, field_validator


class VerificationType(Enum):
    """Types of verification methods supported."""
    SIMPLE = "simple"
    TOKEN_EXISTENCE = "token_existence"
    REGEX = "regex"


class VerificationFunction(BaseModel, ABC):
    """
    Abstract base class for all verification functions.
    
    A verification function takes a model response and determines whether
    it satisfies the verification criteria for a given query. This is used
    to check if fingerprints are properly preserved in model outputs.
    
    Uses Pydantic for automatic serialization/deserialization.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def __call__(self, response: str) -> bool:
        """
        Verify if the response satisfies the verification criteria.
        
        Args:
            response (str): The model's response to evaluate
            
        Returns:
            bool: True if the response passes verification, False otherwise
        """
        pass
    
    def get_query(self) -> str:
        """
        Get the query associated with this verification function.
        
        Returns:
            str: The query/prompt string
        """
        return self.query
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization to handle type enum properly."""
        data = super().model_dump(**kwargs)
        # Ensure type is serialized as its value
        if hasattr(self, 'type') and hasattr(self.type, 'value'):
            data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationFunction':
        """
        Create a verification function instance from dictionary data.
        
        Args:
            data (Dict[str, Any]): Dictionary containing verification function data
            
        Returns:
            VerificationFunction: New instance created from the dictionary data
            
        Raises:
            ValueError: If the verification function type is unknown
        """
        func_type = data.get("type")
        
        # Simple mapping from type strings to classes
        type_to_class = {
            VerificationType.SIMPLE.value: SimpleVerificationFunction,
            VerificationType.TOKEN_EXISTENCE.value: TokenExistenceVerificationFunction,
            VerificationType.REGEX.value: RegexVerificationFunction,
        }
        
        if func_type not in type_to_class:
            raise ValueError(f"Unknown verification function type: {func_type}")
        
        target_class = type_to_class[func_type]
        data_copy = data.copy()
        data_copy.pop("type", None)  # Remove type field before passing to constructor
        
        return target_class.model_validate(data_copy)


class SimpleVerificationFunction(VerificationFunction):
    """
    Verification function that checks for exact string matching.
    
    This verification method checks if the model's response starts with
    the expected response string. It's useful for fingerprints that require
    exact reproduction of specific text.
    """
    
    type: VerificationType = VerificationType.SIMPLE
    
    query: str
    expected_response: str
    
    @field_validator('expected_response')
    @classmethod
    def validate_expected_response(cls, v):
        if len(v) == 0:
            raise ValueError("Expected response cannot be empty")
        return v
    
    def __call__(self, response: str) -> bool:
        """
        Check if response starts with the expected response string.
        
        Args:
            response (str): The model's response to verify
            
        Returns:
            bool: True if response starts with expected_response, False otherwise
        """
        response = response.strip()
        
        # Handle edge case where response might be shorter than expected
        if len(response) < len(self.expected_response):
            return False
            
        return response[:len(self.expected_response)] == self.expected_response


class TokenExistenceVerificationFunction(VerificationFunction):
    """
    Verification function that checks for the presence of specific tokens.
    
    This verification method checks if all required tokens are present
    in the model's response. It's useful for fingerprints that need to
    ensure certain keywords or phrases appear in the output.
    
    Note: This is actually string matching, not true token verification.
    """
    
    type: VerificationType = VerificationType.TOKEN_EXISTENCE
    
    query: str
    required_tokens: List[str]
    case_sensitive: bool = True
    
    @field_validator('required_tokens')
    @classmethod
    def validate_required_tokens(cls, v):
        if not v:
            raise ValueError("At least one token must be specified")
        return v
    
    def __call__(self, response: str) -> bool:
        """
        Check if all required tokens exist in the response.
        
        Args:
            response (str): The model's response to verify
            
        Returns:
            bool: True if all required tokens are found, False otherwise
        """
        if not self.case_sensitive:
            response = response.lower()
            tokens = [token.lower() for token in self.required_tokens]
        else:
            tokens = self.required_tokens
        
        return all(token in response for token in tokens)


class RegexVerificationFunction(VerificationFunction):
    """
    Verification function that uses regular expression pattern matching.
    
    This verification method checks if the model's response matches a
    regular expression pattern.
    """
    
    type: VerificationType = VerificationType.REGEX
    
    query: str
    pattern: str
    flags: int = 0
    
    @field_validator('pattern')
    @classmethod
    def validate_pattern(cls, v):
        if not v:
            raise ValueError("Pattern cannot be empty")
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{v}': {e}")
        return v
    
    def __call__(self, response: str) -> bool:
        """
        Check if response matches the regular expression pattern.
        
        Args:
            response (str): The model's response to verify
            
        Returns:
            bool: True if response matches the regex pattern, False otherwise
        """
        # Compile pattern on demand to avoid serialization issues
        compiled_pattern = re.compile(self.pattern, self.flags)
        return bool(compiled_pattern.search(response))
