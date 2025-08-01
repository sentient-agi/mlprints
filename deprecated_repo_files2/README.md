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

### `engine/verification/` - Advanced Fingerprint System

Production-ready fingerprint generation and verification system with comprehensive verification strategies:

#### **Key Components:**
- **`base.py`** - Core verification function abstractions (`VerificationFunction`, `SimpleVerificationFunction`, `TokenExistenceVerificationFunction`, `RegexVerificationFunction`)
- **`fingerprints.py`** - Fingerprint classes (`Fingerprint`, `SimpleFingerprint`, `TokenExistenceFingerprint`, `RegexFingerprint`) and `FingerprintSet` management with serialization
- **`generate.py`** - Advanced generation framework with multiple strategies (`SimpleTextGenerator`, `RandomWordGenerator`, `TokenExistenceGenerator`, `RegexGenerator`, `InverseNucleusFingerprintGenerator`)
- **`verifiers.py`** - Model verification orchestration with VLLM integration (`Verifier`, `VLLMModelInference`)

#### **Verification Function Types:**
1. **Simple Verification** - Exact text matching for query-response pairs with prefix matching
2. **Token Existence Verification** - Verification based on required token presence with case sensitivity options
3. **Regex Verification** - Pattern-based verification using regular expressions with configurable flags
4. **Composite Verification** - Multiple verification functions with combination strategies (SINGLE, UNION, INTERSECT)

#### **Fingerprint Classes:**
1. **`SimpleFingerprint`** - Single simple verification function for exact matching
2. **`TokenExistenceFingerprint`** - Single token existence verification with customizable token lists
3. **`RegexFingerprint`** - Single regex verification with pattern-based matching
4. **`Fingerprint`** - Composite fingerprint supporting multiple verification functions with flexible combination strategies

#### **FingerprintSet Management:**
- **O(1) Lookup** - Efficient fingerprint retrieval by query using internal dictionary mapping
- **Serialization** - JSON save/load functionality with proper datetime and enum handling
- **Set Operations** - Merging, filtering, sub-sampling, and statistics computation
- **Validation** - Duplicate query detection and comprehensive error handling
- **Statistics** - Comprehensive fingerprint set analysis and reporting

#### **Generation Strategies:**
1. **`SimpleTextGenerator`** - Natural language fingerprints using LLMs with VLLM batch processing
2. **`RandomWordGenerator`** - Structured random word combinations from configurable word lists
3. **`TokenExistenceGenerator`** - Token presence-based fingerprints with configurable token counts
4. **`RegexGenerator`** - Pattern-based fingerprints with customizable regex templates
5. **`InverseNucleusFingerprintGenerator`** - Advanced inverse nucleus sampling implementing Nasery et al. (2025) algorithm

#### **Advanced Generation Features:**
- **Inverse Nucleus Sampling** - Faithful implementation of Nasery et al. perinucleus sampling algorithm
- **Two-Stage Generation** - VLLM for key generation + Transformers for response generation with direct logit access
- **Batch Processing** - Efficient batch generation with configurable batch sizes and progress tracking
- **Validation & Resampling** - Intelligent validation with progressive resampling strategies
- **Word List Management** - Support for 10,000 most-used English words with sampling without replacement
- **Prompt Variations** - Configurable prompt templates for increased diversity

#### **Verifier System:**
- **`Verifier`** - Lightweight verification orchestration across multiple fingerprint sets
- **`VLLMModelInference`** - High-performance model inference with automatic VLLM server management
- **Batch Verification** - Efficient verification across multiple fingerprint sets with progress tracking
- **Verification Vectors** - Returns [0,1] scores for each fingerprint set enabling comparative analysis

### `engine/common/inference_utils.py` - VLLM Integration

Production-ready VLLM inference wrapper with comprehensive server management:

#### **VLLMInference Class Features:**
- **Automatic Server Management** - Seamless VLLM server startup, health checking, and cleanup
- **Multi-GPU Support** - Configurable GPU allocation with automatic tensor parallelism
- **Intelligent Port Selection** - Automatic free port detection and collision avoidance
- **Robust Error Handling** - Comprehensive error recovery and server process management
- **Context Manager Support** - Clean resource management with automatic cleanup
- **Optimized Defaults** - Production-ready server configurations for optimal performance

#### **Key Capabilities:**
- **Chat & Completion APIs** - Support for both chat and completion endpoints
- **Streaming Support** - Event streaming for real-time response generation
- **Flexible Configuration** - Extensive server parameter customization
- **Timeout Management** - Configurable startup and request timeouts
- **Progress Monitoring** - Detailed server startup progress and health monitoring

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
- **`inference_utils.py`** - VLLM inference wrapper with production-ready server management

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

### Fingerprint Generation Scripts

#### `scripts/generate_simple_fingerprints.py`
Advanced fingerprint generation with multiple strategies:

```bash
# Generate simple text fingerprints using LLM with VLLM
python scripts/generate_simple_fingerprints.py \
    --num_fingerprints 1000 \
    --strategy simple_text \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --batch_size 1024

# Generate random word fingerprints
python scripts/generate_simple_fingerprints.py \
    --num_fingerprints 1000 \
    --strategy random_word \
    --key_length 16 \
    --response_length 3

# Generate token existence fingerprints
python scripts/generate_simple_fingerprints.py \
    --num_fingerprints 1000 \
    --strategy token_existence \
    --num_tokens_per_response 3 \
    --case_sensitive

# Generate regex fingerprints
python scripts/generate_simple_fingerprints.py \
    --num_fingerprints 1000 \
    --strategy regex

# Generate inverse nucleus fingerprints (Nasery et al. 2025)
python scripts/generate_simple_fingerprints.py \
    --num_fingerprints 1000 \
    --strategy inverse_nucleus \
    --nucleus_threshold 0.8 \
    --nucleus_k 3 \
    --temperature 0.5 \
    --use_chat_template
```

#### `scripts/generate_inverse_nucleus_fingerprints.py`
Specialized script for inverse nucleus sampling:

```bash
python scripts/generate_inverse_nucleus_fingerprints.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num_fingerprints 1024 \
    --nucleus_threshold 0.8 \
    --nucleus_k 3 \
    --gpu "0,1,2,3"
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
    RegexGenerator,
    InverseNucleusFingerprintGenerator,
    FingerprintSet,
    create_generator
)

# Create generation configuration
config = GenerationConfig(
    num_fingerprints=1000,
    key_length=16,
    response_length=1,  # Nasery et al. typically use 1 token
    temperature=0.5,    # Nasery et al. use 0.5 for key generation
    batch_size=1024,    # Large batches for VLLM efficiency
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    gpu="0,1,2,3",      # Multi-GPU support
    seed=42
)

# Generate different types of fingerprints using factory function
simple_gen = create_generator("simple_text", config)
random_gen = create_generator("random_word", config)
token_gen = create_generator("token_existence", config, 
                           num_tokens_per_response=3, case_sensitive=False)
regex_gen = create_generator("regex", config)
inverse_nucleus_gen = create_generator("inverse_nucleus", config)

# Generate fingerprint sets
simple_set = simple_gen.generate_fingerprint_set()
random_set = random_gen.generate_fingerprint_set()
token_set = token_gen.generate_fingerprint_set()

# Save to files
simple_gen.save_to_file("simple_fingerprints.json")
random_gen.save_to_file("random_fingerprints.json")

# Load existing fingerprints from file
loaded_set = FingerprintSet.load_from_file("simple_fingerprints.json")

# Fingerprint set operations
print(f"Simple set size: {simple_set.size()}")
print(f"Random set size: {random_set.size()}")

# Merge fingerprint sets
merged_set = simple_set.merge_with(random_set)
print(f"Merged size: {merged_set.size()}")

# Sample a subset
small_set = merged_set.sub_sample(50, random_seed=42)

# Filter by type
simple_fps = merged_set.filter_by_type(VerificationType.SIMPLE)
token_fps = merged_set.filter_by_type(VerificationType.TOKEN_EXISTENCE)
```

### Advanced Fingerprint Types

```python
from engine.verification import (
    SimpleFingerprint,
    TokenExistenceFingerprint,
    RegexFingerprint,
    Fingerprint,
    CombinationStrategy,
    SimpleVerificationFunction,
    TokenExistenceVerificationFunction,
    RegexVerificationFunction
)

# Create simple fingerprint (exact prefix matching)
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
composite_fp = Fingerprint(
    combination_strategy=CombinationStrategy.UNION,  # At least one must pass
    verification_functions=[
        SimpleVerificationFunction("What color is the sky?", "blue"),
        TokenExistenceVerificationFunction("What color is the sky?", ["blue", "azure"], case_sensitive=False),
        RegexVerificationFunction("What color is the sky?", r'\b(blue|azure|cerulean)\b', re.IGNORECASE)
    ]
)

# Test fingerprint verification
test_response = "The sky is blue on a clear day"
print(f"Simple verification: {simple_fp.verify('Paris is the capital')}")  # True (prefix match)
print(f"Token verification: {token_fp.verify(test_response)}")  # True if contains required tokens
print(f"Regex verification: {regex_fp.verify('Call me at 555-123-4567')}")  # True if matches pattern
print(f"Composite verification: {composite_fp.verify(test_response)}")  # True if any function passes
```

### Model Verification with VLLM

```python
from engine.verification import (
    Verifier,
    VLLMModelInference,
    create_verifier_from_files,
    print_verification_summary
)

# Create verifier from multiple fingerprint sets
fingerprint_sets = [simple_set, token_set, regex_set]
verifier = Verifier(fingerprint_sets, name="multi_type_verifier")

# Verify model using VLLM with custom configuration
model_path = "path/to/your/model"
vllm_kwargs = {
    "gpu": "0,1,2,3",
    "max_tokens": 256,
    "temperature": 0.1,
    "server_kwargs": {
        "max-model-len": 8192,
        "gpu-memory-utilization": 0.8,
        "tensor-parallel-size": 4,
        "enforce-eager": True,
        "enable-chunked-prefill": True
    },
    "timeout": 300,
    "verbose": True
}

# Run verification and get scores [0,1] for each fingerprint set
verification_vector = verifier.verify_model(model_path, vllm_kwargs=vllm_kwargs)

# Print detailed verification summary
print_verification_summary(verifier, verification_vector)

# Create verifier from files
verifier_from_files = create_verifier_from_files([
    "simple_fingerprints.json",
    "token_fingerprints.json",
    "regex_fingerprints.json"
], names=["simple_set", "token_set", "regex_set"])

# Verify with default VLLM settings
results = verifier_from_files.verify_model(model_path)
```

### VLLM Inference Direct Usage

```python
from engine.common.inference_utils import VLLMInference

# Single GPU inference
with VLLMInference(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    gpu="0",
    server_kwargs={"max-model-len": 4096, "gpu-memory-utilization": 0.8},
    verbose=True,
    timeout=300
) as llm:
    # Chat interface
    response = llm.chat(
        system_prompt="You are a helpful assistant",
        user_prompt="Explain quantum computing",
        temperature=0.7,
        max_tokens=512
    )
    print(response)

# Multi-GPU inference with advanced configuration
with VLLMInference(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    gpu=[0, 1, 2, 3],  # or gpu="0,1,2,3"
    server_kwargs={
        "tensor-parallel-size": 4,
        "max-model-len": 8192,
        "gpu-memory-utilization": 0.9,
        "enforce-eager": True,
        "enable-chunked-prefill": True,
        "max-num-seqs": 2048
    },
    verbose=True,
    timeout=600  # Longer timeout for large models
) as llm:
    # Completion interface
    completion = llm.complete(
        prompt="The future of artificial intelligence is",
        temperature=0.8,
        max_tokens=256
    )
    print(completion)
    
    # Streaming completion
    stream = llm.complete(
        prompt="Write a story about",
        temperature=0.9,
        max_tokens=512,
        stream=True
    )
    for chunk in stream:
        print(chunk.choices[0].text, end="")
```

### Fingerprint Set Statistics and Analysis

```python
from engine.verification import fingerprint_set_statistics

# Compute comprehensive statistics
stats = fingerprint_set_statistics(merged_set)
print(f"Statistics: {stats}")

# Example output:
# {
#   'total_fingerprints': 2000,
#   'type_distribution': {'simple': 1000, 'token_existence': 500, 'regex': 500},
#   'avg_query_length': 24.5,
#   'set_name': 'merged_set',
#   'created_at': '2024-01-15T10:30:00'
# }

# Advanced fingerprint set operations
set1 = FingerprintSet.load_from_file("set1.json")
set2 = FingerprintSet.load_from_file("set2.json")

# Check for overlaps
common_queries = set1.get_queries().intersection(set2.get_queries())
print(f"Common queries: {len(common_queries)}")

# Get specific fingerprint
specific_fp = set1.get_fingerprint_by_query("What is the capital of France?")
if specific_fp:
    print(f"Found fingerprint: {specific_fp.get_query()}")

# Filter and analyze
simple_only = merged_set.filter_by_type(VerificationType.SIMPLE)
token_only = merged_set.filter_by_type(VerificationType.TOKEN_EXISTENCE)

print(f"Simple fingerprints: {simple_only.size()}")
print(f"Token existence fingerprints: {token_only.size()}")
```

### Inverse Nucleus Generation (Nasery et al. 2025)

```python
from engine.verification import (
    GenerationConfig,
    InverseNucleusFingerprintGenerator,
    WordListManager
)

# Configure for inverse nucleus sampling following Nasery et al.
inverse_config = GenerationConfig(
    num_fingerprints=1024,
    key_length=16,           # Nasery et al. use 16 tokens for keys
    response_length=1,       # Nasery et al. typically use 1 token responses
    temperature=0.5,         # Nasery et al. use 0.5 for key generation
    nucleus_threshold=0.8,   # Nasery et al. use 0.8 for nucleus threshold
    nucleus_k=3,             # Nasery et al. use k=3 for sampling
    batch_size=1024,         # Large batches for VLLM efficiency
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    gpu="0,1,2,3",
    use_chat_template=False,  # Set to True for chat models
    use_prompt_variations=True,  # Add diversity to prompts
    word_list_path="data/common/word_list.txt",  # 10,000 most-used English words
    seed=42
)

# Generate fingerprints using inverse nucleus sampling
inverse_gen = InverseNucleusFingerprintGenerator(inverse_config)
inverse_set = inverse_gen.generate_fingerprint_set()

print(f"Generated {inverse_set.size()} inverse nucleus fingerprints")

# Save the generated fingerprints
inverse_gen.save_to_file("inverse_nucleus_fingerprints.json")

# Verify the word list is properly loaded
word_manager = WordListManager("data/common/word_list.txt")
print(f"Loaded {len(word_manager.word_list)} words")

# Get random words without replacement for maximum diversity
diverse_words = word_manager.get_random_words_without_replacement(100)
print(f"Diverse word sample: {diverse_words[:10]}")
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
- **Flexible Verification Logic** - SINGLE, UNION, and INTERSECT combination strategies
- **VLLM Integration** - High-performance model inference with automatic server management
- **Batch Verification** - Efficient verification across multiple fingerprint sets
- **Serialization Support** - Robust JSON save/load with datetime and enum handling
- **Statistical Analysis** - Comprehensive fingerprint set statistics and performance metrics
- **Advanced Generation** - Inverse nucleus sampling, batch processing, and validation

### Production-Ready Infrastructure
- **Modular Architecture** - Clean abstractions and easy extension
- **Comprehensive Error Handling** - Graceful failure recovery and validation
- **Resource Management** - Automatic GPU detection, memory optimization, and cleanup
- **Progress Monitoring** - Detailed progress tracking with ETA and performance metrics
- **Flexible Configuration** - Environment variables, config files, and programmatic APIs

## Testing Framework

Comprehensive integration testing with debugging capabilities:

```bash
# Run all tests
python tests/test_training_integration.py

# Test specific components
python tests/test_training_without_eval.py
python tests/test_training_with_eval.py
python tests/test_generate_simple_fingerprints.py
python tests/test_generators.py
python tests/inference_utils_test.py
```

## Installation & Dependencies

### Requirements
The framework requires Python 3.8+ and includes:
- **Core ML**: `torch`, `transformers`, `accelerate`, `deepspeed`, `vllm`
- **Evaluation**: `lm_eval`, `datasets`, `evaluate`
- **Analysis**: `matplotlib`, `pandas`, `numpy`, `tqdm`
- **Infrastructure**: `wandb`, `peft`, `pydantic`, `tenacity`

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run basic tests
python tests/test_generate_simple_fingerprints.py

# Quick training test
python tests/test_training_without_eval.py

# Test VLLM inference
python tests/inference_utils_test.py
```

## Migration from Legacy Code

The refactored system maintains full backward compatibility while providing significant improvements:

### Key Improvements
1. **Modular Design** - Clean separation of concerns with production-ready abstractions
2. **Advanced Verification** - Multi-type fingerprints with flexible combination strategies
3. **VLLM Integration** - High-performance inference with automatic server management
4. **Robust Generation** - Inverse nucleus sampling and advanced validation
5. **Comprehensive Testing** - Integration tests and debugging tools
6. **Better Performance** - Optimized resource usage, batch processing, and GPU utilization

### Legacy Mapping
| Original Component | Refactored Module |
|-------------------|-------------------|
| `generate_finetuning_data.py` | `engine.verification.generate` |
| `finetune_multigpu.py` | `engine.training.robust_trainer` |
| `meta_learning_trainer.py` | `engine.training.meta_learning_loops` |
| `utils.py` | `engine.common.*` |
| Basic fingerprint generation | `engine.verification.fingerprints` |
| Simple verification | `engine.verification.verifiers` |