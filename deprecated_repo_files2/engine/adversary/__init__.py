"""
Adversary Module

This module contains adversarial attack implementations for testing model robustness,
including logit-based attacks and system prompt variations.
"""

from .logit_attacks import LogitAttacker, GradientBasedAttack, TokenSubstitutionAttack
from .prompt_variations import SystemPromptVariator, PromptAugmenter
from .false_positive_attacks import FalsePositiveAttacker, FalsePositiveConfig, SamplingConfig
from .logits_processor_attacks import (
    LogitsProcessorAttacker, 
    LogitsProcessorConfig, 
    LogitsProcessorAttackResult,
    KthTokenLogitsProcessor,
    InvNucleusSampler,
    RemoveTopWordLogitProcessor
)
from .base import AdversaryBase, AttackResult

__all__ = [
    "AdversaryBase",
    "AttackResult", 
    "LogitAttacker",
    "GradientBasedAttack",
    "TokenSubstitutionAttack",
    "SystemPromptVariator",
    "PromptAugmenter",
    "FalsePositiveAttacker",
    "FalsePositiveConfig",
    "SamplingConfig",
    "LogitsProcessorAttacker",
    "LogitsProcessorConfig",
    "LogitsProcessorAttackResult",
    "KthTokenLogitsProcessor",
    "InvNucleusSampler",
    "RemoveTopWordLogitProcessor"
] 