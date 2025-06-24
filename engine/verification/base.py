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
    4. [Optional] Override _get_init_params() if needed for custom serialization
       (the class self-registers automatically)
"""

from typing import List, Dict, Any, Type
from enum import Enum
from abc import ABC, abstractmethod
import re
import inspect


class VerificationType(Enum):
    """Types of verification methods supported."""
    SIMPLE = "simple"
    TOKEN_EXISTENCE = "token_existence"
    REGEX = "regex"


class VerificationFunction(ABC):
    """
    Abstract base class for all verification functions.
    
    A verification function takes a model response and determines whether
    it satisfies the verification criteria for a given query. This is used
    to check if fingerprints are properly preserved in model outputs.
    
    Attributes:
        verification_type (VerificationType): The type of verification method
        query (str): The query/prompt associated with this verification
    """
    
    # Subclasses should set this to identify their type
    type: VerificationType = None

    # Registry for verification function types
    _registry: Dict[str, Type["VerificationFunction"]] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses that define a 'type' attribute."""
        super().__init_subclass__(**kwargs)
        verifier_type = getattr(cls, "type", None)
        if verifier_type is not None:
            VerificationFunction._registry[verifier_type.value] = cls

    def __init__(self, query: str):
        """
        Initialize the verification function.
        
        Args:
            query (str): The query/prompt this verification function is associated with
        """
        if self.type is None:
            raise ValueError(f"Subclass {self.__class__.__name__} must set 'type' attribute")
        self.verification_type = self.type
        self.query = query

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
    
    def _get_init_params(self) -> Dict[str, Any]:
        """
        Get the parameters needed to reconstruct this instance.
        
        Subclasses can override this for custom serialization logic.
        
        Returns:
            Dict[str, Any]: Parameters for __init__
        """
        # Get constructor signature and extract current values
        sig = inspect.signature(self.__init__)
        params = {}
        for param_name in sig.parameters:
            if param_name != 'self' and hasattr(self, param_name):
                params[param_name] = getattr(self, param_name)
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the verification function to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the verification function
        """
        result = self._get_init_params()
        result["type"] = self.verification_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationFunction':
        """
        Create a verification function instance from dictionary data.
        
        This method acts as a factory, creating the appropriate subclass
        based on the 'type' field in the data.
        
        Args:
            data (Dict[str, Any]): Dictionary containing verification function data
            
        Returns:
            VerificationFunction: New instance created from the dictionary data
            
        Raises:
            ValueError: If the verification function type is unknown
        """
        func_type = data.get("type")
        if func_type not in cls._registry:
            raise ValueError(f"Unknown verification function type: {func_type}")
        
        # Get the appropriate subclass and create instance
        target_class = cls._registry[func_type]
        data_copy = data.copy()
        data_copy.pop("type", None)  # Remove type field before passing to constructor
        
        return target_class(**data_copy)


class SimpleVerificationFunction(VerificationFunction):
    """
    Verification function that checks for exact string matching.
    
    This verification method checks if the model's response starts with
    the expected response string. It's useful for fingerprints that require
    exact reproduction of specific text.
    
    Attributes:
        expected_response (str): The exact string that the response should start with
    """
    
    type = VerificationType.SIMPLE
    
    def __init__(self, query: str, expected_response: str):
        """
        Initialize the simple verification function.
        
        Args:
            query (str): The query/prompt associated with this verification
            expected_response (str): The exact string the response should start with
        """
        super().__init__(query)
        if len(expected_response) == 0:
            raise ValueError("Expected response cannot be empty")
        self.expected_response = expected_response
    
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


# TODO: Not actually token verification, but rather string matching. Either change name or add a new verification type.
class TokenExistenceVerificationFunction(VerificationFunction):
    """
    Verification function that checks for the presence of specific tokens.
    
    This verification method checks if all required tokens are present
    in the model's response. It's useful for fingerprints that need to
    ensure certain keywords or phrases appear in the output.
    
    Attributes:
        required_tokens (List[str]): List of tokens that must be present
        case_sensitive (bool): Whether token matching is case-sensitive
    """
    
    type = VerificationType.TOKEN_EXISTENCE
    
    def __init__(self, query: str, required_tokens: List[str], case_sensitive: bool = True):
        """
        Initialize the token existence verification function.
        
        Args:
            query (str): The query/prompt associated with this verification
            required_tokens (List[str]): List of tokens that must be present in response
            case_sensitive (bool, optional): Whether matching is case-sensitive. Defaults to True.
            
        Raises:
            ValueError: If required_tokens is empty
        """
        super().__init__(query)
        self.required_tokens = required_tokens
        self.case_sensitive = case_sensitive
        
        if not required_tokens:
            raise ValueError("At least one token must be specified")
    
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
    
    Attributes:
        pattern (str): The regular expression pattern to match
        flags (int): Regular expression flags (e.g., re.IGNORECASE)
        compiled_pattern (re.Pattern): Compiled regex pattern for efficiency
    """
    
    type = VerificationType.REGEX
    
    def __init__(self, query: str, pattern: str, flags: int = 0):
        """
        Initialize the regex verification function.
        
        Args:
            query (str): The query/prompt associated with this verification
            pattern (str): Regular expression pattern to match against responses
            flags (int, optional): Regular expression flags. Defaults to 0.
            
        Raises:
            ValueError: If pattern is empty or invalid regex
        """
        super().__init__(query)
        
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        
        self.pattern = pattern
        self.flags = flags
        try:
            self.compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    
    def __call__(self, response: str) -> bool:
        """
        Check if response matches the regular expression pattern.
        
        Args:
            response (str): The model's response to verify
            
        Returns:
            bool: True if response matches the regex pattern, False otherwise
        """
        return bool(self.compiled_pattern.search(response))
    
    def _get_init_params(self) -> Dict[str, Any]:
        """
        Override to exclude compiled_pattern from serialization.
        
        Returns:
            Dict[str, Any]: Parameters for __init__ (excluding compiled_pattern)
        """
        return {
            "query": self.query,
            "pattern": self.pattern,
            "flags": self.flags
        }
