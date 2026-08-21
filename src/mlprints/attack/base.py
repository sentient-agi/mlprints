"""
Base class for attack models that wrap AutoModelForCausalLM.
"""

import torch.nn as nn
from abc import ABC, abstractmethod

# universal message emitted when attack model cannot respond differently
# Used for false positive rate measurement and ensures consistent behavior
# across all attack types. Can be imported directly for measurement code.
UNIVERSAL_OVERLOAD_MESSAGE = "I'm sorry Dave, I'm afraid I can't do that."


class AttackModel(nn.Module, ABC):
    """
    Base class for attack models that wrap AutoModelForCausalLM.
    
    When an attack model lacks sufficient power to respond differently (e.g., 
    when blocking/filtering), it should return the universal overload message
    defined by UNIVERSAL_OVERLOAD_MESSAGE. This is used for false positive 
    rate measurement and ensures consistent behavior across all attack types.
    """
    
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    
    @property
    def device(self):
        return self.model.device
    
    def forward(self, *args, **kwargs):
        """
        Forward pass - delegates to the base model.
        This is needed for loglikelihood evaluation and other direct model calls.
        Subclasses can override this if they need attack-specific forward behavior.
        """
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """
        Proxy unknown attributes to the model.
        This allows the attack model to behave like the model for most operations.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        """
        Generate method - MUST be overridden by subclasses.
        
        This is where attack-specific logic should be implemented.
        Subclasses should intercept generation calls and apply their attack logic.
        """
        pass

