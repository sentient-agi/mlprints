# OML Exploration - Production-Ready Framework

This is a comprehensive, production-ready framework for OML (Open, Monetizable, and Loyal AI) research and development, featuring clean object-oriented design, modular architecture, and advanced training capabilities.

## Conceptual Overview: Why OML?

Artificial Intelligence is commonly offered today via two extremes:

1. **Closed APIs** – centrally hosted black-box models such as ChatGPT. They scale well and allow built-in content moderation, yet give end-users zero transparency or control.
2. **Open-sourced weights** – downloadable models such as Meta's Llama. They empower users but force creators to forfeit any ability to monetise or to enforce responsible usage.

OML (Open • Monetisable • Loyal) seeks the *best of both worlds*. A model publisher starts with a base model *M* and fine-tunes it with a carefully crafted set of **fingerprint pairs** *(key → response)* to obtain **M.oml**. The fingerprints act as cryptographic watermarks:

• **Open** – the weights are shipped to the user and can run locally, preserving transparency and performance optimisation.
• **Monetisable** – only inputs carrying a valid owner signature unlock full capability, making pay-per-use possible.
• **Loyal** – the owner can selectively authorise queries, ensuring adherence to ethical/safety policies.

The underlying security relies on *fingerprinting capacity*: the number of robust fingerprints that can be embedded without degrading the model's original utility. This framework introduces two major advances:

1. **Scalability with Anti-Forgetting Regularisers** – techniques such as meta-learning and task vectors allow us to embed ≈ 1k+ fingerprints in models ranging from 3B to 70B parameters with minimal utility loss.
2. **Robustness to Adversarial Attacks** – advanced training approaches including meta-learning loops and task vector augmentation preserve fingerprint accuracy even under sophisticated adversarial fine-tuning attempts.

## Architecture Overview

The refactored system is organized into focused, production-ready modules:

```
oml-exploration/
├── engine/                    # Core framework modules
│   ├── verification/          # Fingerprint generation and verification
│   ├── training/             # Robust training with meta-learning
│   ├── adversary/            # Adversarial testing and attacks
│   ├── utility/              # Benchmarking and analysis tools
│   └── common/               # Shared utilities and integrations
├── scripts/                  # Production scripts and tools
├── tests/                    # Comprehensive integration tests  
├── data/                     # Data storage and resources
└── deprecated_repo_files/    # Legacy code (preserved for reference)
```

## Core Engine Modules

### `engine/verification/` - Fingerprint System

Advanced fingerprint generation and verification system with multiple strategies:

#### **Key Components:**
- **`base.py`** - Core enumerations, configurations, and verification function abstractions
- **`fingerprints.py`** - Fingerprint classes (`Fingerprint`, `SimpleFingerprint`, `TokenExistenceFingerprint`, `RegexFingerprint`) and `FingerprintSet` management
- **`generate.py`** - Multiple generator classes (`SimpleTextGenerator`, `RandomWordGenerator`, `TokenExistenceGenerator`, `RegexGenerator`, `InverseNucleusGenerator`)
- **`verifiers.py`** - Model verification orchestration with VLLM integration

#### **Fingerprint Types:**
1. **Simple Fingerprints** - Exact text matching for query-response pairs
2. **Token Existence Fingerprints** - Verification based on required token presence
3. **Regex Fingerprints** - Pattern-based verification using regular expressions
4. **Composite Fingerprints** - Multiple verification functions with combination strategies (UNION, INTERSECT)

#### **Generation Strategies:**
1. **Simple Text Generation** - Natural language fingerprints using LLMs with configurable chat templates
2. **Random Word Generation** - Structured random word combinations from word lists
3. **Token Existence Generation** - Fingerprints that verify token presence in responses
4. **Regex Generation** - Pattern-based fingerprints with customizable regex templates
5. **Inverse Nucleus Sampling** - Advanced sampling for steganographic fingerprints

### `engine/training/` - Robust Training System

State-of-the-art training system for embedding robust fingerprints:

#### **Key Features:**
- **`robust_trainer.py`** - Main trainer with meta-learning and task vector approaches
- **`meta_learning_loops.py`** - Advanced meta-learning training loops
- **`training_utils.py`** - Comprehensive utilities for distributed training, FSDP, DeepSpeed
- **`callbacks.py`** - Training callbacks for monitoring, averaging, and early stopping

#### **Training Approaches:**
1. **Meta-Learning** - Trains models to retain fingerprints under adversarial fine-tuning
2. **Task Vectors** - Uses task vector augmentation for robustness
3. **Distributed Training** - Full support for multi-GPU, FSDP, and DeepSpeed
4. **Advanced Regularization** - Forgetting regularizers and model averaging

### `engine/adversary/` - Adversarial Testing

Comprehensive adversarial testing framework:

#### **Attack Types:**
- **`logit_attacks.py`** - Gradient-based and token substitution attacks
- **`prompt_variations.py`** - System prompt augmentation and variations
- **`false_positive_attacks.py`** - False positive generation and testing
- **`logits_processor_attacks.py`** - Advanced logits processor attacks

### `engine/utility/` - Analysis & Benchmarking

Production-ready evaluation and analysis tools:

#### **Capabilities:**
- **`benchmarks.py`** - Standardized benchmark suites and custom evaluations
- **`metrics.py`** - Comprehensive metrics computation and tracking
- **`analysis.py`** - Experiment analysis, visualization, and reporting

### `engine/common/` - Shared Infrastructure

Robust shared utilities for the entire framework:

#### **Core Utilities:**
- **`llm_utils.py`** - LLM management, model/tokenizer wrappers
- **`data_utils.py`** - Data processing and dataset management
- **`huggingface_utils.py`** - HuggingFace integration and model loading
- **`lm_eval_utils.py`** - LM evaluation harness integration

> **Note:** Training utilities are located in `engine/training/` rather than `engine/common/` for better module organization.

## Production Scripts

### Experiment Orchestration

#### `scripts/launch_parallel_experiments.py`
Sophisticated experiment launcher with intelligent resource management:

**Key Features:**
- **Smart Resource Management** - Automatic model size detection and batch size optimization
- **Advanced Progress Tracking** - Color-coded progress with fingerprint-aware time estimation
- **Robust Error Handling** - Signal handling, cleanup, and graceful failure recovery
- **Flexible Configuration** - Environment variables and command-line interface

```bash
# Easy environment variable interface
MODEL_NAME=Llama-3.2-8B-Instruct python scripts/launch_parallel_experiments.py

# Advanced command-line interface
python scripts/launch_parallel_experiments.py \
  --model_name Llama-3.2-8B-Instruct \
  --fingerprint_counts 128,512,1024 \
  --batch_sizes 32,64 \
  --eval_tasks "ifeval,mmlu"
```

### Robust Training

#### `scripts/train_robust_fingerprints.py`
Comprehensive robust training with meta-learning and task vectors:

```bash
# Meta-learning approach
python scripts/train_robust_fingerprints.py \
    --model_family llama \
    --model_size 7B \
    --num_fingerprints 1024 \
    --max_steps 200 \
    --ft_inner_loop_steps 8

# Task vectors approach
python scripts/train_robust_fingerprints.py \
    --model_family llama \
    --model_size 7B \
    --use_task_vectors \
    --task_vectors_coefficients "0.5,1.0,1.5"
```

### Analysis & Visualization

#### `scripts/plot_training_metrics.py`
Advanced training visualization and analysis:

```bash
python scripts/plot_training_metrics.py --base_dir /path/to/results/
```

#### `scripts/check_eval_results.py`
Real-time experiment monitoring:

```bash
python scripts/check_eval_results.py --verbose
```

### Additional Helper Scripts

#### `scripts/generate_simple_fingerprints.py`
Generate fingerprint datasets for training:

```bash
# Generate simple text fingerprints using LLM
python scripts/generate_simple_fingerprints.py --num_fingerprints 1000 --strategy simple_text

# Generate random word fingerprints
python scripts/generate_simple_fingerprints.py --num_fingerprints 1000 --strategy random_word

# Generate token existence fingerprints
python scripts/generate_simple_fingerprints.py --num_fingerprints 1000 --strategy token_existence

# Generate regex fingerprints
python scripts/generate_simple_fingerprints.py --num_fingerprints 1000 --strategy regex

# Generate inverse nucleus fingerprints
python scripts/generate_simple_fingerprints.py --num_fingerprints 1000 --strategy inverse_nucleus
```

#### `scripts/run_logits_processor_attacks.py`
Execute logits processor adversarial attacks:

```bash
python scripts/run_logits_processor_attacks.py --model_path /path/to/model --fingerprints fingerprints.json
```

#### `scripts/run_false_positive_attack.py`
Test false positive generation attacks:

```bash
python scripts/run_false_positive_attack.py --model_path /path/to/model
```

## Usage Examples

### Basic Fingerprint Generation

```python
from engine.verification import (
    GenerationConfig,
    RandomWordGenerator,
    SimpleTextGenerator,
    TokenExistenceGenerator,
    FingerprintSet,
    create_generator
)

# Generate random word fingerprints
config = GenerationConfig(
    num_fingerprints=1000,
    key_length=16,
    response_length=3,
    seed=42
)

generator = RandomWordGenerator(config)
fingerprint_set = generator.generate()

# Save to file
generator.save_to_file("fingerprints.json")

# Verify at fingerprint set level (takes query + response)
is_valid = fingerprint_set.verify("sample key", "expected response")

# Or verify individual fingerprints (only takes response)
first_fingerprint = list(fingerprint_set.fingerprints)[0]
query = first_fingerprint.get_query()
is_valid_individual = first_fingerprint.verify("expected response")

# Generate different types of fingerprints
simple_gen = create_generator("simple_text", config)
token_gen = create_generator("token_existence", config, num_tokens_per_response=3)
regex_gen = create_generator("regex", config)

# Load existing fingerprints from file
loaded_set = FingerprintSet.load_from_file("fingerprints.json")
```

### Model Verification with Verifiers

```python
from engine.verification import (
    Verifier,
    VLLMModelInference,
    create_verifier_from_files,
    print_verification_summary
)

# Create verifier from multiple fingerprint sets
fingerprint_sets = [fingerprint_set, loaded_set]
verifier = Verifier(fingerprint_sets, name="multi_set_verifier")

# Verify model using VLLM (default)
model_path = "path/to/your/model"
verification_vector = verifier.verify_model(model_path, use_vllm=True)

# Print detailed verification summary
print_verification_summary(verifier, verification_vector)

# Create verifier from files
verifier_from_files = create_verifier_from_files([
    "fingerprints1.json",
    "fingerprints2.json"
], names=["set1", "set2"])

# Custom VLLM configuration
vllm_kwargs = {
    "gpu": "0,1",
    "max_tokens": 256,
    "temperature": 0.1,
    "server_kwargs": {"max-model-len": 8192}
}
results = verifier_from_files.verify_model(model_path, vllm_kwargs=vllm_kwargs)
```

### Working with Different Fingerprint Types

```python
from engine.verification import (
    SimpleFingerprint,
    TokenExistenceFingerprint,
    RegexFingerprint,
    Fingerprint,
    CombinationStrategy
)

# Create simple fingerprint (exact match)
simple_fp = SimpleFingerprint(
    query="What is the capital of France?",
    expected_response="Paris"
)

# Create token existence fingerprint
token_fp = TokenExistenceFingerprint(
    query="Describe a sunset",
    required_tokens=["orange", "sky", "horizon"],
    case_sensitive=False
)

# Create regex fingerprint
regex_fp = RegexFingerprint(
    query="Generate a phone number",
    pattern=r'\d{3}-\d{3}-\d{4}'  # Matches XXX-XXX-XXXX format
)

# Create composite fingerprint with multiple verification functions
from engine.verification.base import SimpleVerificationFunction, TokenExistenceVerificationFunction

composite_fp = Fingerprint(
    combination_strategy=CombinationStrategy.UNION,  # At least one must pass
    verification_functions=[
        SimpleVerificationFunction("What color is the sky?", "blue"),
        TokenExistenceVerificationFunction("What color is the sky?", ["blue", "azure"], case_sensitive=False)
    ]
)

# Test fingerprint verification
test_response = "The sky is blue on a clear day"
print(f"Simple verification: {simple_fp.verify('Paris')}")
print(f"Token verification: {token_fp.verify(test_response)}")
print(f"Composite verification: {composite_fp.verify(test_response)}")
```

### Convenience Functions for Quick Generation

```python
from engine.verification.generate import (
    generate_simple_text_fingerprints,
    generate_random_word_fingerprints,
    generate_token_existence_fingerprints,
    generate_regex_fingerprints
)

# Quick generation with default settings
simple_fps = generate_simple_text_fingerprints(
    num_fingerprints=100,
    key_length=16,
    response_length=8,
    output_path="simple_fingerprints.json"
)

random_fps = generate_random_word_fingerprints(
    num_fingerprints=100,
    key_length=5,
    response_length=3,
    output_path="random_fingerprints.json"
)

token_fps = generate_token_existence_fingerprints(
    num_fingerprints=100,
    num_tokens_per_response=3,
    case_sensitive=False,
    output_path="token_fingerprints.json"
)

# Load and combine multiple fingerprint sets
from engine.verification import FingerprintSet

set1 = FingerprintSet.load_from_file("simple_fingerprints.json")
set2 = FingerprintSet.load_from_file("random_fingerprints.json")
merged_set = set1.merge_with(set2)

# Fingerprint set operations
print(f"Set1 size: {set1.size()}")
print(f"Set2 size: {set2.size()}")
print(f"Merged size: {merged_set.size()}")

# Sample a subset
small_set = merged_set.sub_sample(50, random_seed=42)
```

### Advanced Training

```python
from engine.training import (
    RobustFingerprintTrainer,
    MetaLearningConfig,
    TaskVectorConfig
)

# Configure meta-learning training
ml_config = MetaLearningConfig(
    inner_loop_steps=8,
    learning_rate=1e-5,
    forgetting_regularizer_strength=0.05
)

trainer = RobustFingerprintTrainer(ml_config)
trainer.train(
    model_path="path/to/model",
    fingerprints=fingerprint_set,
    num_epochs=100
)
```

### Adversarial Testing

```python
from engine.adversary import (
    LogitsProcessorAttacker,
    FalsePositiveAttacker,
    SystemPromptVariator
)

# Test model robustness
attacker = LogitsProcessorAttacker()
results = attacker.attack(model, fingerprint_set)

print(f"Attack success rate: {results.success_rate}")
print(f"Fingerprints compromised: {results.compromised_count}")
```

### Comprehensive Analysis

```python
from engine.utility import (
    BenchmarkSuite,
    PerformanceAnalyzer,
    TrainingAnalyzer
)

# Benchmark model utility
benchmark = BenchmarkSuite()
results = benchmark.evaluate(model, tasks=["ifeval", "mmlu"])

# Analyze training metrics
analyzer = TrainingAnalyzer()
analysis = analyzer.load_experiment("experiment_hash")
analyzer.plot_training_evolution(analysis)
```

## Advanced Features

### Multi-Modal Training Approaches
- **Meta-Learning Loops** - Bi-level optimization for adversarial robustness
- **Task Vector Augmentation** - Robust fingerprint embedding via task vectors
- **Distributed Training** - Full FSDP, DeepSpeed, and multi-node support
- **Advanced Regularization** - Forgetting prevention and model averaging

### Sophisticated Evaluation
- **Integrated lm-eval-harness** - Standardized model evaluation
- **Custom Benchmark Suites** - Domain-specific evaluation capabilities
- **Real-time Monitoring** - Progress tracking and performance analysis
- **Comprehensive Metrics** - Utility preservation and fingerprint retention

### Advanced Verification Features
- **Multi-Type Fingerprints** - Simple, token existence, regex, and composite fingerprints
- **Flexible Verification Logic** - UNION, INTERSECT, and SINGLE combination strategies
- **VLLM Integration** - High-performance model inference with automatic server management
- **Batch Verification** - Efficient verification across multiple fingerprint sets
- **Serialization Support** - JSON save/load for fingerprints and fingerprint sets
- **Statistical Analysis** - Comprehensive fingerprint set statistics and performance metrics

### Production-Ready Infrastructure
- **Modular Architecture** - Easy extension and customization
- **Comprehensive Testing** - Integration tests for all components
- **Robust Error Handling** - Graceful failure and recovery mechanisms
- **Flexible Configuration** - Environment variables and config files

## Testing Framework

Comprehensive integration testing with debugging capabilities:

```bash
# Run all tests
python tests/test_training_integration.py

# Test specific components
python tests/test_training_without_eval.py
python tests/test_training_with_eval.py
python tests/test_generate_simple_fingerprints.py
```

## Installation & Dependencies

### Requirements
The framework requires Python 3.8+ and includes:
- **Core ML**: `torch`, `transformers`, `accelerate`, `deepspeed`
- **Evaluation**: `lm_eval`, `datasets`, `evaluate`
- **Analysis**: `matplotlib`, `pandas`, `numpy`
- **Infrastructure**: `wandb`, `tqdm`, `peft`

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run basic tests
python tests/test_generate_simple_fingerprints.py

# Quick training test
python tests/test_training_without_eval.py
```

## Migration from Legacy Code

The refactored system maintains full backward compatibility while providing significant improvements:

### Key Improvements
1. **Modular Design** - Clean separation of concerns and easy extensibility
2. **Production Ready** - Robust error handling, monitoring, and logging
3. **Advanced Training** - Meta-learning, task vectors, and distributed training
4. **Comprehensive Testing** - Integration tests and debugging tools
5. **Better Performance** - Optimized resource usage and batch processing

### Legacy Mapping
| Original Component | Refactored Module |
|-------------------|-------------------|
| `generate_finetuning_data.py` | `engine.verification.generate` |
| `finetune_multigpu.py` | `engine.training.robust_trainer` |
| `meta_learning_trainer.py` | `engine.training.meta_learning_loops` |
| `utils.py` | `