"""
Centralized training utilities for the whole mlprints package.

The main access point is via the functions in training.train.
"""

from .callbacks import (
    EarlyStoppingByLoss,
    ModelAverageCallback,
)
from .collators import (
    CausalLMPadOnlyDataCollator,
    TopKCausalLMDataCollator,
)
from .formatting import (
    build_zero3_config,
    concatenate_datasets,
    format_training_data,
    format_training_data_from_tokens,
)
from .train import run_sft_train
from .trainers import (
    CausalLMSFTTrainer,
    CompositeCausalLMTrainer,
    TeacherLoss,
)

__all__ = [
    # callbacks
    "EarlyStoppingByLoss",
    "ModelAverageCallback",
    # collators
    "CausalLMPadOnlyDataCollator",
    "TopKCausalLMDataCollator",
    # formatting
    "build_zero3_config",
    "concatenate_datasets",
    "format_training_data",
    "format_training_data_from_tokens",
    # train
    "run_sft_train",
    # trainers
    "CausalLMSFTTrainer",
    "CompositeCausalLMTrainer",
    "TeacherLoss",
]
