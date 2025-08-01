"""
Training Module for Robust Fingerprint Embedding

This module provides advanced training techniques for making fingerprints robust
against adversarial attacks, including meta-learning approaches and task vectors.
"""

from .robust_trainer import (
    RobustFingerprintTrainer,
    MetaLearningConfig,
    TaskVectorConfig
)

from .meta_learning_loops import (
    ft_meta_training_loop,
    task_vectors_training_loop,
    inner_loop_step_ft,
    inner_step_task_vectors
)

from .training_utils import (
    # FSDP utilities
    FSDPModelStorage,
    fsdp_v1_model_params,
    
    # Training schedule and sampling
    schedule,
    distributed_sample_adversary_lr,
    distributed_sample_task,
    get_distributed_random_number,
    
    # Data loading utilities
    get_next_batch,
    next_n_batches,
    
    # Optimizer utilities
    delete_optimizer,
    
    # Training objectives
    obj_standard_max_next_token,
    
    # Task vector utilities
    apply_task_vector,
    prepare_task_vectors,
    
    # Evaluation utilities
    compute_fingerprint_accuracy,
    
    # Model expansion utilities
    verify_expanded_parameters,
    expand_feedforward_weights,
    
    # Distributed training utilities
    smallest_power_of_two,
    get_free_port,
    warm_up_model_kernels,
    
    # DeepSpeed utilities
    build_zero2_config,
    apply_deepspeed_fixes
)

from .callbacks import (
    MemoryCallback,
    ModelAverageCallback,
    EarlyStoppingByLoss,
    ResetOriginalParametersCallback,
    AsyncEvalLauncherCallback
)

__all__ = [
    # Main training classes
    'RobustFingerprintTrainer', 
    'MetaLearningConfig',
    'TaskVectorConfig',
    
    # Training loops
    'ft_meta_training_loop',
    'task_vectors_training_loop', 
    'inner_loop_step_ft',
    'inner_step_task_vectors',
    
    # FSDP utilities
    'FSDPModelStorage',
    'fsdp_v1_model_params',
    
    # Training schedule and sampling
    'schedule',
    'distributed_sample_adversary_lr',
    'distributed_sample_task',
    'get_distributed_random_number',
    
    # Data loading utilities
    'get_next_batch',
    'next_n_batches',
    
    # Optimizer utilities
    'delete_optimizer',
    
    # Training objectives
    'obj_standard_max_next_token',
    
    # Task vector utilities
    'apply_task_vector',
    'prepare_task_vectors',
    
    # Evaluation utilities
    'compute_fingerprint_accuracy',
    
    # Training callbacks
    'MemoryCallback',
    'ModelAverageCallback',
    'EarlyStoppingByLoss',
    'ResetOriginalParametersCallback',
    'AsyncEvalLauncherCallback',
    
    # Model expansion utilities
    'verify_expanded_parameters',
    'expand_feedforward_weights',
    
    # Distributed training utilities
    'smallest_power_of_two',
    'get_free_port',
    'warm_up_model_kernels',
    
    # DeepSpeed utilities
    'build_zero2_config',
    'apply_deepspeed_fixes'
] 