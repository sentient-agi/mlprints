# Robust Fingerprint Training Module

This module provides a comprehensive framework for training models with robust fingerprints using meta-learning and task vectors approaches. The training is designed to make fingerprints resilient against adversarial fine-tuning attacks.

## Overview

The training module is organized into several key components:

- **`robust_trainer.py`**: Main training orchestration classes
- **`meta_learning_loops.py`**: Core training loops for meta-learning and task vectors
- **`utils.py`**: FSDP utilities, adversarial sampling, and task vector operations
- **`callbacks.py`**: Training callbacks for memory management, model averaging, etc.
- **`model_utils.py`**: Model utilities for expansion, loading, and parameter management

## Quick Start

### Basic Usage

```python
from engine.training import (
    RobustFingerprintTrainer,
    MetaLearningConfig,
    TaskVectorConfig
)

# Initialize trainer
trainer = RobustFingerprintTrainer(
    model_family="llama",
    model_size="3B",
    num_fingerprints=1024,
    max_key_length=16,
    max_response_length=1,
    result_path="results/"
)

# Meta-learning training
config = MetaLearningConfig(
    max_steps=100,
    ft_inner_loop_steps=4,
    ft_loss_scale=0.5
)
model_path = trainer.train_meta_learning(config)

# Task vectors training
config = TaskVectorConfig(
    max_steps=1000,
    ft_loss_scale=4.0,
    task_vectors_coefficients="1.0"
)
model_path = trainer.train_task_vectors(config)
```

### Command Line Interface

Use the modular training script:

```bash
# Meta-learning approach
python scripts/train_robust_fingerprints_modular.py \
    --method meta_learning \
    --model_family llama \
    --model_size 3B \
    --max_steps 100

# Task vectors approach  
python scripts/train_robust_fingerprints_modular.py \
    --method task_vectors \
    --model_family llama \
    --model_size 3B \
    --max_steps 1000
```

## Training Methods

### Meta-Learning Approach

The meta-learning approach trains the model to maintain fingerprint accuracy even after adversarial fine-tuning. It uses an inner-outer loop structure:

- **Inner loop**: Simulates adversarial fine-tuning attacks
- **Outer loop**: Updates model parameters to maintain fingerprint robustness

Key parameters:
- `ft_inner_loop_steps`: Number of adversarial fine-tuning steps
- `adversaries_per_step`: Number of different adversaries per training step
- `ft_loss_scale`: Scaling factor for fingerprint loss

### Task Vectors Approach

The task vectors approach uses the difference between base and instruction-tuned models to create robust fingerprints. It applies these task vectors with different coefficients during training.

Key parameters:
- `task_vectors_coefficients`: Coefficients for applying task vectors
- `ft_loss_scale`: Scaling factor for fingerprint loss

## Configuration Classes

### MetaLearningConfig

Configures meta-learning training parameters:

```python
config = MetaLearningConfig(
    max_steps=100,                              # Maximum training steps
    ft_inner_loop_steps=4,                      # Inner loop steps
    ft_loss_scale=0.5,                          # Fingerprint loss scaling
    gradient_accumulation_steps=8,              # Gradient accumulation
    adversarial_gradient_accumulation_steps=2,  # Adversarial gradient accumulation
    adversaries_per_step=1,                     # Number of adversaries per step
    inner_ft_optimizer='sgd',                   # Inner loop optimizer
    forgetting_regularizer_strength=0.0,       # Forgetting regularization
    finetuning_dataset='alpaca'                 # Adversarial fine-tuning dataset
)
```

### TaskVectorConfig

Configures task vectors training parameters:

```python
config = TaskVectorConfig(
    max_steps=1000,                    # Maximum training steps
    ft_loss_scale=4.0,                 # Fingerprint loss scaling
    gradient_accumulation_steps=2,     # Gradient accumulation
    task_vectors_coefficients="1.0",   # Task vector coefficients
    ce_loss_scale=1.0                  # Cross-entropy loss scaling
)
```

## RobustFingerprintTrainer

The main training class that orchestrates the entire training process:

```python
trainer = RobustFingerprintTrainer(
    model_family="llama",                           # Model family
    model_size="3B",                                # Model size
    model_path=None,                                # Custom model path
    num_fingerprints=1024,                          # Number of fingerprints
    max_key_length=16,                              # Fingerprint key length
    max_response_length=1,                          # Fingerprint response length
    fingerprint_generation_strategy="english",     # Fingerprint generation
    fingerprints_file_path=None,                    # Pre-generated fingerprints
    data_split=0,                                   # Data split index
    batch_size=8,                                   # Training batch size
    inner_batch_size=2,                             # Inner loop batch size
    learning_rate=1e-5,                             # Learning rate
    weight_decay=0.01,                              # Weight decay
    result_path="results/",                         # Results directory
    wandb_project='robust_fingerprinting',         # W&B project
    seed=42                                         # Random seed
)
```

## Training Callbacks

The module includes several useful training callbacks:

### MemoryCallback
Monitors and manages memory usage during training:

```python
from engine.training import MemoryCallback

callback = MemoryCallback(enable_gc=True)
```

### ModelAverageCallback
Averages model weights with original model to prevent catastrophic forgetting:

```python
from engine.training import ModelAverageCallback

callback = ModelAverageCallback(model, orig_model_weight=0.25)
```

### EarlyStoppingByLoss
Stops training when loss falls below a threshold:

```python
from engine.training import EarlyStoppingByLoss

callback = EarlyStoppingByLoss(loss_threshold=0.005)
```

## Model Utilities

### Model Expansion
Expand feedforward weights for fingerprint insertion:

```python
from engine.training import expand_feedforward_weights

model = expand_feedforward_weights(model, expansion_rate=0.01)
```

### Enhanced Model Loading
Load models with support for multiple families:

```python
from engine.training import load_model_and_tokenizer_enhanced

model, tokenizer = load_model_and_tokenizer_enhanced(
    model_family="llama",
    model_size="3B",
    variant="Instruct"
)
```

## Supported Model Families

- **LLaMA**: `meta-llama/Llama-3.2-{size}` and `meta-llama/Llama-3.2-{size}-Instruct`
- **Gemma**: `google/gemma-2-{size}` and `google/gemma-2-{size}-it`
- **Mistral**: `mistralai/Mistral-{size}-v0.3`
- **Microsoft Phi**: `microsoft/Phi-3-{size}-instruct`
- **OLMo**: `allenai/OLMo-2-1124-{size}`
- **Qwen**: `Qwen/Qwen2.5-{size}`

## Advanced Features

### FSDP Support
The module includes full support for Fully Sharded Data Parallel (FSDP) training with utilities for parameter and gradient management.

### Distributed Training
Built-in support for distributed training with proper synchronization and communication.

### Asynchronous Evaluation
Optional asynchronous evaluation during training with the `AsyncEvalLauncherCallback`.

### Memory Management
Automatic memory management and garbage collection during training.

## Example Experiments

### Small Scale Experiment
```bash
python scripts/train_robust_fingerprints_modular.py \
    --method meta_learning \
    --model_family llama \
    --model_size 3B \
    --num_fingerprints 256 \
    --max_steps 50 \
    --ft_inner_loop_steps 2
```

### Large Scale Experiment
```bash
python scripts/train_robust_fingerprints_modular.py \
    --method task_vectors \
    --model_family llama \
    --model_size 8B \
    --num_fingerprints 2048 \
    --max_steps 1000 \
    --batch_size 4
```

## Migration from Legacy Training

The new modular system replaces the previous `training_utils.py` and standalone training scripts. Key changes:

1. **Structured Configuration**: Use `MetaLearningConfig` and `TaskVectorConfig` instead of passing many parameters
2. **Modular Design**: Import specific components from `engine.training` 
3. **Simplified Interface**: Use `RobustFingerprintTrainer` for all training orchestration
4. **Better Organization**: Callbacks, utilities, and model operations are in separate modules

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` or `num_fingerprints`
2. **FSDP Errors**: Ensure proper model wrapping with `setup_accelerator`
3. **Import Errors**: Check that all dependencies are installed and paths are correct

### Performance Tips

1. Use appropriate `gradient_accumulation_steps` for your hardware
2. Enable memory callbacks for long training runs
3. Use task vectors for faster training when instruction-tuned models are available
4. Adjust `adversarial_gradient_accumulation_steps` based on memory constraints 