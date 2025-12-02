# Deepprints

A state-of-the-art framework for fingerprinting controlling LLMs amd testing them in adversarial settings.

## Quick Start

### Installation

The deepprints library is a Python package that can be easily installed from the source code provided in this repository. We recommend using `uv`:
```bash
uv sync
```

Replace `python` with `uv run` when running scripts (we assume `uv` is installed in the examples throughout this README).

**Alternative installation methods:**

Using `conda`:
```bash
conda create -n oml311 python=3.11
conda activate oml311
pip install -e . # editable install helps in development process
```

Using a standard virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Configuration

Set the environment variable `OML_EXPERIMENTS_DIR` to specify where experiment results will be stored:

```bash
export OML_EXPERIMENTS_DIR=/path/to/experiments
```

Alternatively, you can specify the experiments directory using the `--experiments-dir` flag when running commands.

### Managing Dependencies

When adding new dependencies, update `pyproject.toml` at the same commit where the new dependency is introduced (`uv` should do this automatically). To test dependency installation, recreate the environment:

```bash
conda deactivate  # if not currently in base environment
conda remove -n oml311 --all
conda create -n oml311 python=3.11
conda activate oml311
pip install -e .
```

## Overview

Deepprints implements a comprehensive framework for LLM **fingerprinting and adversarial attack** research. The core purpose is to efficiently:
1. **Generate fingerprints**
2. **Train embeddable fingerprints**
3. **Attack fingerprints**
4. **Measure the TPR and FPR of fingerprints with and without attacks**
5. **Measure the utility of fingerprinted models**

The codebase follows a **modular, centralized architecture** designed to support and continuously add SOTA fingerprinting and attack algorithms in a reproducible and extensible way. Furthermore, it enables testing them in realistic production settings, accelerating the deployment of practical fingerprinting solutions for open model creators.

The repository is built mainly on abstractions from the Transformers library (for model loading and inference), DeepSpeed (for efficient distributed training), and DeepEval (for benchmark evaluation).

This repository was initially created as a spin-off of research efforts at Sentient, following a systematic investigation of adversarial robustness in LLM fingerprinting schemes. The framework implements the attacks and evaluation methodology from ["Are Robust LLM Fingerprints Adversarially Robust?"](https://arxiv.org/pdf/2509.26598), which identifies fundamental vulnerabilities in existing fingerprinting schemes and demonstrates adaptive attacks that can bypass model authentication while maintaining high utility.

## Currently Supported Fingerprints

| Algorithm | Description | Category |
|-----------|-------------|----------|
| **RoFL** | Optimizes prompts using Greedy Coordinate Gradient to maximize probability of target responses. | Memorization-based |
| **Perinucleus** | Generates unique responses by sampling tokens outside the probability nucleus for given queries. | Memorization-based |
| **Instructional Fingerprints** | Trains models to respond with specific responses to instruction-formatted queries via fine-tuning. | Memorization-based |
| **Chain&Hash** | Embeds fingerprints using chain-of-thought reasoning with hash-based verification mechanisms. | Memorization-based |
| **Implicit Fingerprints (ImF)** | Embeds fingerprints implicitly through model behavior without explicit query-response pairs. | Intrinsic |
| **FPEdit** | Inserts fingerprints via model editing techniques that modify model weights directly. | Intrinsic |
| **Domain Specific Watermarks** | Embeds watermarks using statistical properties tailored to specific domains (e.g., medical). | Statistical |
| **ProfLingo** | Generates fingerprints using statistical profiling of linguistic patterns in model outputs. | Statistical |
| **MergePrint** | Creates fingerprints by merging multiple models and extracting unique response patterns. | Statistical |

*Note: Some implementations may be in development or require configuration. See `src/oml/common/fingerprints.py` for current registration status.*

## Currently Supported Fingerprint Attacks

The framework implements attacks organized around **four fundamental vulnerability themes** identified in adversarial robustness research:

| Attack Category | Implemented Attacks | Vulnerability Exploited |
|----------------|---------------------|------------------------|
| **Output Suppression** | `logit_sampling` (SuppressTop-k, SuppressNeighbor) | Fingerprint responses can be suppressed by modifying logit distributions during generation to avoid emitting fingerprint tokens |
| **Output Detection** | `logits_measurement` | Fingerprint responses exhibit detectable statistical anomalies in logit distributions that can be identified via calibrated detectors |
| **Input Detection** | `perplexity_filtering` | Fingerprint queries can be detected and paraphrased to break exact/substring matching while preserving semantic meaning |
| **Statistical Analysis** | `statistical_analysis` | Fingerprint responses can be identified via statistical properties (unigram frequencies, perplexity) and filtered out |

### Attack Taxonomy

Based on the threat model where a malicious model host attempts to evade fingerprint detection:

1. **Output Suppression Attacks**: Modify the sampling process to suppress fingerprint response tokens, including:
   - **SuppressTop-k**: Remove top-k tokens from logits for initial tokens
   - **SuppressNeighbor**: Suppress tokens lexically similar to fingerprint responses

2. **Output Detection Attacks**: Detect fingerprint responses via logit distribution analysis:
   - **Logits Measurement**: Calibrate detectors on benign outputs, identify anomalies in fingerprint responses

3. **Input Detection Attacks**: Detect and modify fingerprint queries:
   - **Rephrasing**: Paraphrase queries to break exact/substring matching while preserving semantic meaning

4. **Statistical Analysis Attacks**: Identify fingerprints via statistical properties:
   - **Perplexity Filtering**: Filter low-probability outputs that may indicate fingerprints
   - **Statistical Analysis**: Detect fingerprint response tokens via unigram/bigram frequency analysis

*See `src/oml/common/attacks.py` for registered attack implementations.*

## Threat Model & Adversarial Setting

The framework evaluates fingerprinting schemes under a **black-box adversarial threat model** where:

- **Model Owner**: Embeds fingerprints into their model and shares model weights or provides API access
- **Malicious Model Host**: Has access to the fingerprinted model (via API or weights) and attempts to evade fingerprint detection while maintaining model utility
- **Evaluation Setting**: Black-box access (input-output pairs only), matching realistic API deployment scenarios

The adversary's goal is to:
1. **Evade Detection**: Prevent fingerprint verification from succeeding (high ASR)
2. **Maintain Utility**: Preserve model performance on standard benchmarks (low utility degradation)

This threat model is more realistic than benign robustness evaluations (fine-tuning, prompt-wrappers) because it assumes an **adaptive adversary** who can:
- Analyze model outputs to detect fingerprints
- Modify generation processes to suppress fingerprint responses
- Transform inputs to evade query matching
- Use statistical analysis to identify fingerprint patterns

## Repo Architecture & Design

Because actions (via commands) are dependent on each other (e.g., training a fingerprint requires first generating one), each experiment follows a hierarchical structure that is automatically enforced when operating via the commands. Manual modification within experiment folders is possible but unadvised, as it can cause unintended consequences.

There is a high correlation between scientific reproducibility and centralization. This is why our repository is "centralized" in the following ways:

* **Core operations** are collected in **`oml.common`**, providing centralized inference, training, prompt optimization (search), fingerprint registry, attack registry, model/tokenizer loading, and seed settings. This design maximizes reusability across the codebase. When building upon the repository, if you are *not* utilizing the abstractions of **`oml.common`** but are performing one of the aforementioned actions, you are missing out on reproducibility. **If the core library does not have a desired feature, fingerprint, or attack, *open a git issue* and we will address it as soon as possible!**

* **Fingerprint implementations** follow a simple Pythonic design (assuming users have read and understood the original papers) and are stored in **`oml.fingerprint`**. Each implements a generation function and optionally a training function (if trainable). They are centrally indexed in **`oml.common.fingerprints`**.

* **Interface scripts** (experimenter actions) are stored in **`scripts/`** and are accessible via CLI commands registered in **`oml.cli`**. These scripts utilize the abstractions of the core library.

While this creates strong constraints for experimenters, nothing prevents them from forking the repository and altering those core operations (or suggesting a PR). The most important design choice is that operations are *centralized* so that all experiments follow consistent patterns.

Beyond centralization, the repository follows these key principles:

- **Paper-Faithful Implementation**: Fingerprint implementations aim to match original papers as closely as possible. The code assumes users have read the papers and serves as both implementation and reference library.

- **Config-Driven Workflow**: YAML configs define experiments (models, hyperparameters, paths). Scripts orchestrate workflows but delegate to registered algorithms. Experiment directories are auto-organized with timestamps/UUIDs.

- **Separation of Concerns**: Clear boundaries between generation, training, measurement, and attack modules, with shared functionality centralized in `oml/common/`.

The downstream qualities and advantages of building upon this repository include:
1. **Extensibility**: Adding new algorithms is straightforward (register + implement)
2. **Reproducibility**: Configs saved with results, seed management
3. **Modularity**: Clear separation between algorithms, attacks, measurements
4. **Experiment Management**: Auto-organized directories with timestamps


## Recipes for adding to the repo

### How to *add* a fingerprint (e.g. `new_fingerprint_name`)
- *Create* a .py file in **src/oml/fingerprint/** that implements the fingerprint generation function and *optionally* a training function. The generation function should have signature `(**config_params) -> (fingerprints: list, metadata: list)`, and the training function (if applicable) should have signature `(fingerprints: list, checkpoints_dir: str, **config_params) -> metadata: dict`.
- *Open* **src/oml/common/fingerprints.py**. Import the respective generation and training functions and add `new_fingerprint_name` as a key with the indexed functions to the `FINGERPRINT_ALGOS` dict.
- *Create* a default .yaml config file in **configs/fingerprint/** named `new_fingerprint_name_config.yaml` and ideally set the best model-agnostic settings and add any helpful comments.

### How to *add* an attack (e.g. `new_attack_name`)
- *Create* a .py file in **src/oml/attack/** that implements:
  - A preparation function with signature `(model_checkpoint: str, **config_params) -> (attack_config: dict, metadata: dict)`
  - A class that inherits from `AttackModel` in **src/oml/attack/base.py** (overriding the `.generate()` method to intercept and modify HuggingFace's `AutoModelForCausalLM.generate()` behavior)
- *Open* **src/oml/common/attacks.py**. Import the respective preparation function and attack class, and add `new_attack_name` as a key with the indexed preparation function and class to the `ATTACK_ALGOS` dict.
- *Create* a default .yaml config file in **configs/attack/** named `new_attack_name_config.yaml` and ideally set the best model and fingerprinting scheme-agnostic settings and add any helpful comments.

## Recipes for working with the repo

Here we enumerate several recurrent recipes for interacting with the repository, mainly via CLI commands. Although the command structure for several actions is shared, we repeat it for clarity and quick reference.

### How to *generate* a fingerprint

**Via CLI:**
```bash
oml generate configs/fingerprint/rofl_config.yaml [--experiments-dir PATH] [--experiment-name NAME] [--train]
```

**Via script:**
```bash
uv run python scripts/generate_fingerprints.py configs/fingerprint/rofl_config.yaml [--experiments-dir PATH] [--experiment-name NAME] [--train]
```

**What it does:**
- Loads models from `models_dict` in the config
- Generates fingerprints using the specified algorithm
- Saves `fingerprints.yaml` and `metadata.yaml` to `experiments/{experiment_name}/fingerprints/{algo}/{timestamp}/`
- Optionally trains the model (if `--train` flag is used and the algorithm supports training)

**Prerequisites:**
- Set `OML_EXPERIMENTS_DIR` environment variable, or use `--experiments-dir`
- Valid config YAML with algorithm name and parameters

### How to *train* a fingerprint

**Via CLI:**
```bash
oml train configs/fingerprint/perinucleus_config.yaml --fingerprints-dir PATH [--experiments-dir PATH] [--experiment-name NAME]
```

**Via script:**
```bash
uv run python scripts/train_fingerprints.py configs/fingerprint/perinucleus_config.yaml --fingerprints-dir PATH [--experiments-dir PATH] [--experiment-name NAME]
```

**What it does:**
- Loads fingerprints from a previously generated fingerprint directory
- Fine-tunes the model on the fingerprints (if the algorithm supports training)
- Saves checkpoints to `experiments/{experiment_name}/fingerprints/{algo}/{timestamp}/trained/{timestamp}/checkpoints/`

**Prerequisites:**
- Previously generated fingerprints (via `generate` command)
- Algorithm must support training (e.g., `perinucleus`, `instructional_fp`)

### How to *measure* the *strength* of a fingerprint

**Via CLI:**
```bash
oml measure strength PATH/TO/FINGERPRINTS [--model-id MODEL_ID] [--device-map DEVICE] [--quantization 8bit|4bit]
```

**Via script:**
```bash
uv run python scripts/measure_fingerprints.py PATH/TO/FINGERPRINTS [--model-id MODEL_ID] [--device-map DEVICE] [--quantization 8bit|4bit]
```

**What it does:**
- Loads fingerprints from the specified directory
- Runs inference with the model on each fingerprint query
- Compares generated responses to expected responses (using exact match or custom comparator)
- Calculates hit rate and saves results to `experiments/{experiment_name}/measurements/strength/{timestamp}/`

**Prerequisites:**
- Previously generated fingerprints directory
- Model to evaluate (defaults to model used for generation if not specified)

### How to *measure* the *utility* of a model

**Via script (utility measurement):**
```bash
uv run python scripts/measure_datasets.py configs/measure/utility_config.yaml
```

**What it does:**
- Evaluates model performance on standard benchmarks (MMLU, HellaSwag, ARC, etc.)
- Uses DeepEval framework for evaluation
- Measures perplexity or task-specific metrics
- Saves results for comparison between fingerprinted and non-fingerprinted models

**Prerequisites:**
- Model checkpoint or HuggingFace model ID
- Valid measurement config specifying benchmark and evaluation parameters

**Note:** Utility measurement compares:
- **Non-fingerprinted model**: Baseline performance on standard benchmarks
- **Fingerprinted model**: Performance after training on fingerprints (should remain high)
- **Attacked model**: Performance after applying attack to fingerprinted model (should remain high for effective attacks)

The goal is to ensure:
1. Fingerprinting doesn't degrade model utility (harmlessness criterion)
2. Attacks maintain model utility while evading fingerprints (demonstrates vulnerability)

**Evaluation Benchmarks:**
- **MMLU**: Massive Multitask Language Understanding
- **HellaSwag**: Commonsense reasoning
- **ARC**: AI2 Reasoning Challenge
- **TriviaQA**: Question answering
- **Perplexity**: Language modeling (on standard corpora)

These benchmarks ensure that fingerprinting and attacks don't compromise the model's core capabilities.

## Detailed Repository Tree Structure

```
oml/                        # Package
├── fingerprint/            # Fingerprinting algorithms
│   ├── perinucleus.py
│   ├── ...
│   └── rofl.py
│
├── attack/                 # Attack algorithms
│   ├── base.py             # AttackModel base class
│   ├── logits_measurement.py
│   ├── ...
│   └── perplexity_filtering.py
│
├── measure/                # Evaluation metrics
│   ├── strength.py         # Fingerprint hit rate measurement
│   ├── utility.py          # Model utility
│   ├── false_positives.py  # False positive rate
│   └── perplexity.py       # Perplexity metrics
│
├── common/                 # Shared utilities
│   ├── inference.py        # LM inference
│   ├── searching.py        # Prompt optimization (Greedy Coordinate Gradient)
│   ├── training.py         # Supervised fine-tuning
│   ├── evals.py            # DeepEval benchmarks
│   ├── fingerprints.py     # Fingerprinting algorithms registry
│   ├── attacks.py          # Attack algorithms registry
│   └── utils.py            # Path handling, model loading, seed setting, etc.
│
├── visualize/              # Plotting utilities
│   ├── fpr_tpr_plot.py     # ROC plotting
│   ├── fpsr_utility_plot.py    # Eval vs FPR
│   └── logits_plot.py          # Logits
│
├── data/                   # Fixed datasets & auxiliary data 
│   ├── datasets/           # Evaluation datasets
│   └── fingerprint_generation/  # Algorithm-specific data
│
├── cli.py                  # Unified CLI entry point
│
├──scripts/                    # Workflow orchestration scripts, loaded into cli.py
│  ├── generate_fingerprints.py
│  ├── train_fingerprints.py
│  ├── ...
│  └── attack_fingerprints.py
│
└──configs/                 # default YAML configuration files
    ├── fingerprint/
    ├── attack/
    └── measure/
```

---

## Key Workflows

### 1. Fingerprint Generation Workflow

```
Config YAML → generate_fingerprints.py
    ↓
Load models from models_dict
    ↓
Map config params → algorithm-specific signature
    ↓
Call registered generate() function
    ↓
Save fingerprints.yaml + metadata.yaml
    ↓
[Optional] Train fingerprints → checkpoints/
```

**Key Design Decisions:**
- **Model loading abstraction**: `models_dict` supports multiple models (target, key_gen, etc.)
- **Algorithm-specific mapping**: Each algorithm gets models mapped to its expected signature
- **Return convention**: All generators return `(fingerprints: list, metadata: list)`

**Example Config Structure:**
```yaml
seed: 42
algo:
  name: rofl
  params:
    models_dict:
      target:
        model_id: meta-llama/Llama-3.2-1B-Instruct
        device_map: cuda:0
    num_fingerprints: 4
    prompt_length: 10
    gcg_config:
      num_steps: 50
      top_k: 256
```

### 2. Training Workflow

```
Fingerprints → train_fingerprints.py
    ↓
Load training model (from models_dict)
    ↓
Map training params → algorithm train() signature
    ↓
Fine-tune model on fingerprints
    ↓
Save checkpoints/ + metadata.yaml
```


### 3. Measurement Workflow

```
Fingerprints + Model → measure_fingerprints.py
    ↓
Load fingerprints.yaml
    ↓
For each fingerprint:
    - Run inference with model
    - Compare generated vs expected response
    - Record hit/miss
    ↓
Save results (hit rate, per-fingerprint details)
```

**Measurement Types:**
The framework measures fingerprint strength (TPR), false positive rate (FPR), utility, attack success rate (ASR), and perplexity. See the [Evaluation Methodology](#evaluation-methodology) section for detailed metric definitions, matching paradigms, and evaluation workflows.

### 4. Attack Workflow

```
Trained Model + Attack Config → attack_fingerprints.py
    ↓
Prepare attack (load surrogate models, calibrate detectors)
    ↓
Create AttackModel wrapper (inherits from base.AttackModel)
    ↓
Save attack config + metadata
    ↓
[Later] Use AttackModel.generate() to evade fingerprints
```


---

## Design Patterns & Conventions

### 1. **Path Normalization**
- All paths normalized via `normalize_str_to_path()` (resolves `~`, `.`, etc.)
- Explicit `.` required for current directory (prevents empty string bugs)

### 2. **Timestamp UUIDs**
- Experiment directories use `{timestamp}-{uuid}` format
- Ensures uniqueness and chronological ordering

### 3. **Seed Management**
- Global `set_seeds()` sets random, numpy, torch seeds in oml.scripts

### 4. **YAML I/O**
- `save_yaml()` / `load_yaml()` utilities
- Configs saved alongside results for reproducibility

### 5. **Model Device Management**
- Models loaded with `device_map` (supports multi-GPU)
- Device-aware tensor operations
- Training vs inference mode separation

---

## Core Technical Implementations

### 1. GCG Optimization (`common/searching.py`)

**Purpose**: Optimize discrete token sequences (prompts) to maximize/minimize a loss function.

**Key Features:**
- **Greedy Coordinate Gradient**: Token-level gradient computation
- **Batch optimization**: Multiple candidates evaluated per step
- **Chat template support**: Handles conversation formatting
- **Modifiable indices**: Only optimize user message tokens (not system/assistant headers)

**Algorithm Flow:**
1. Format conversation with modifiable user tokens
2. Compute gradients w.r.t. token embeddings
3. Find top-k token replacements per position
4. Sample candidates and evaluate
5. Update best candidate, repeat

**Used By:**
- `rofl.py`: Optimizes prompts to maximize target response probability
- Other fingerprint methods that need prompt optimization

### 2. Inference Module (`common/inference.py`)

**Purpose**: Unified interface for model inference with various sampling strategies.

**Key Features:**
- **Batched inference**: Process multiple prompts efficiently
- **Beam search**: Multi-candidate generation
- **Custom logits processors**: BottomK, Perinucleus sampling
- **Chat template handling**: Automatic formatting

**Sampling Strategies:**
- Greedy (`do_sample=False`)
- Top-k, Top-p
- Bottom-k (for fingerprint initialization)
- Perinucleus (sampling outside probability nucleus)

### 3. Model Loading (`common/utils.py`)

**Purpose**: Standardized model/tokenizer loading with device management.

**Features:**
- Supports local paths and HuggingFace hub IDs
- Device mapping (`auto`, `cuda:0`, etc.)
- Quantization support (8-bit, 4-bit)
- Training vs inference mode handling

### 4. Training Module (`common/training.py`)

**Purpose**: Fine-tuning support for fingerprint training.

**Features:**
- HuggingFace Trainer integration
- DeepSpeed/FSDP support (mutually exclusive)
- Gradient accumulation
- Early stopping
- Evaluation datasets

---

### Configuration Conventions

- **`target`**: Primary model being fingerprinted (or first entry if absent)
- **Algorithm-specific keys**: `key_gen` (perinucleus), `base`/`surrogate` (attacks)
- **Model config fields**: `model_id`, `device_map`, `torch_dtype`, `trust_remote_code`

### Experiment Directory Structure

```
experiments/                # Not fixed
  {experiment_name}/        # Structure below this is fixed if using only the deepprint commands
    fingerprints/
      {algo_name}/
        {timestamp}/
          fingerprints.yaml
          metadata.yaml
          config.yaml
          trained/           # If --train flag is used during generation or train command is used (and are valid)
            {timestamp}/
              checkpoints/
              config.yaml
              metadata.yaml
    attacks/
      {attack_name}/
        {timestamp}/
          attack_config.yaml
          metadata.yaml
    measurements/             # Generated by measure scripts
      strength/
      utility/
      false_positives/
```

---

## Evaluation Methodology

The framework evaluates fingerprinting schemes using the following metrics and workflows:

### Metrics

- **True Positive Rate (TPR)**: Fraction of fingerprints correctly verified (target: >0.9)
- **False Positive Rate (FPR)**: Fraction of benign queries misidentified as fingerprints (target: <0.1)
- **Attack Success Rate (ASR)**: Fraction of fingerprints evaded by attacks (ASR = 1 - TPR_under_attack)
- **Utility Preservation**: Model performance on benchmarks (MMLU, HellaSwag, ARC, TriviaQA) relative to baseline (target: >0.95)
- **Perplexity**: Token-level probability metrics for statistical analysis

### Matching Paradigms

Fingerprint verification uses three matching strategies: **PrefixMatch** (exact prefix), **SubstringMatch** (substring), and **KeywordMatch** (keywords). The choice affects both TPR and FPR.

---

## Contributing

We welcome contributions to expand the fingerprinting and attack landscape for modern, SOTA generative models. The repository is designed to be an ever-updating centralized research-grade resource. We particularly encourage contributions focused on developing fingerprinting schemes that are **adversarially robust by design**.

If you encounter issues or have suggestions for improvements, please open a GitHub issue. For adding new fingerprints or attacks, see the [Recipes for adding to the repo](#recipes-for-adding-to-the-repo) section above.
