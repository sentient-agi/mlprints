"""
Base Adversary Classes

Defines the abstract base classes and common data structures for adversarial attacks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import torch
from torch import nn


class AttackType(Enum):
    """Types of adversarial attacks supported."""
    LOGIT_BASED = "logit_based"
    PROMPT_VARIATION = "prompt_variation"
    GRADIENT_BASED = "gradient_based"
    TOKEN_SUBSTITUTION = "token_substitution"


@dataclass
class AttackResult:
    """Result of an adversarial attack."""
    success: bool
    original_input: str
    adversarial_input: str
    original_output: str
    adversarial_output: str
    confidence: float
    attack_type: AttackType
    metadata: Dict[str, Any]
    

@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    attack_type: AttackType
    max_iterations: int = 100
    epsilon: float = 0.1
    learning_rate: float = 0.01
    target_confidence: float = 0.9
    batch_size: int = 1
    device: str = "cuda"
    additional_params: Dict[str, Any] = None


class AdversaryBase(ABC):
    """
    Abstract base class for all adversarial attack implementations.
    
    This class defines the common interface that all adversarial attack methods must implement.
    """
    
    def __init__(self, config: AttackConfig):
        self.config = config
        self.attack_type = config.attack_type
        self.device = config.device
        
    @abstractmethod
    def attack(
        self, 
        model: nn.Module, 
        input_text: Union[str, List[str]], 
        target_output: Optional[str] = None
    ) -> Union[AttackResult, List[AttackResult]]:
        """
        Execute the adversarial attack.
        
        Args:
            model: The target model to attack
            input_text: The input text(s) to modify
            target_output: Optional target output for targeted attacks
            
        Returns:
            AttackResult or list of AttackResults
        """
        pass
        
    @abstractmethod
    def evaluate_attack_success(
        self, 
        original_output: str, 
        adversarial_output: str,
        target_output: Optional[str] = None
    ) -> bool:
        """
        Determine if the attack was successful.
        
        Args:
            original_output: Output from the original input
            adversarial_output: Output from the adversarial input  
            target_output: Target output for targeted attacks
            
        Returns:
            Boolean indicating attack success
        """
        pass
        
    def preprocess_input(self, input_text: str) -> Dict[str, Any]:
        """
        Preprocess input text for the attack.
        
        Args:
            input_text: Raw input text
            
        Returns:
            Preprocessed input dictionary
        """
        # TODO: Implement tokenization and input preprocessing
        return {"input_text": input_text}
        
    def postprocess_result(self, result: AttackResult) -> AttackResult:
        """
        Postprocess the attack result.
        
        Args:
            result: Raw attack result
            
        Returns:
            Processed attack result
        """
        # TODO: Implement result postprocessing and validation
        return result 