# Adversary Module

The adversary module contains implementations of various adversarial attacks for testing the robustness of fingerprint detection systems and language models.

## Overview

This module provides a class-based architecture for implementing and running adversarial attacks with:
- Standardized interfaces through `AdversaryBase`
- Configurable attack parameters via dataclasses
- Structured result reporting with `AttackResult`
- Extensible design for new attack types

## Available Attacks

### 1. False Positive Attacks (`FalsePositiveAttacker`)

Analyzes fingerprint detection systems for false positive vulnerabilities using different sampling configurations.

**Key Features:**
- Standard and adversarial sampling configurations
- Batch and single processing modes
- Monte Carlo analysis with configurable trials
- Comprehensive result reporting

**Example Usage:**

```python
from engine.adversary import FalsePositiveAttacker, FalsePositiveConfig, AttackType

# Configure the attack
config = FalsePositiveConfig(
    attack_type=AttackType.LOGIT_BASED,
    fingerprint_file_path="data/fingerprints.json",
    num_fingerprints=1000,
    model_path="microsoft/DialoGPT-medium",
    num_mc_trials=10,
    batch_size=32,
    use_adversarial_sampling=True
)

# Run the attack
attacker = FalsePositiveAttacker(config)
result = attacker.attack()

# Analyze results
print(f"False positive rate: {result.fp_frac_with_sampling:.4f}")
print(f"Attack success: {result.success}")

# Save results
attacker.save_results(result)
```

**Command Line Usage:**

```bash
python scripts/run_false_positive_attack.py \
    --fp_file_path data/fingerprints.json \
    --model_path microsoft/DialoGPT-medium \
    --num_fp 1000 \
    --use_adversarial_sampling \
    --batch_size 32
```

### 2. Logit-Based Attacks (`LogitAttacker`)

Base class for attacks that manipulate model logits.

**Implementations:**
- `GradientBasedAttack`: Uses gradient information for adversarial perturbations
- `TokenSubstitutionAttack`: Finds optimal token substitutions

### 3. Prompt Variation Attacks (`SystemPromptVariator`)

Attacks that modify system prompts to evade detection.

## Configuration

### FalsePositiveConfig

Main configuration class for false positive attacks:

```python
@dataclass
class FalsePositiveConfig(AttackConfig):
    fingerprint_file_path: str          # Path to fingerprint data
    num_fingerprints: int = 1024        # Number of fingerprints to analyze
    model_path: str                     # Model to attack
    num_mc_trials: int = 10             # Monte Carlo trials per config
    batch_size: int = 32                # Processing batch size
    seed: int = 42                      # Random seed
    use_adversarial_sampling: bool      # Use adversarial sampling configs
    sampling_configs: List[SamplingConfig]  # Custom sampling configs
    output_dir: str                     # Output directory for results
```

### SamplingConfig

Configuration for sampling parameters:

```python
@dataclass
class SamplingConfig:
    temperature: float = 1.0    # Sampling temperature
    top_p: float = 1.0          # Nucleus sampling threshold
    top_k: int = 0              # Top-k sampling limit
    min_p: float = 0.0          # Minimum probability threshold
```

## Sampling Strategies

### Standard Sampling Configurations

Balanced configurations for general analysis:
- Conservative (temp=0.6, top_p=0.7, top_k=20)
- Standard (temp=0.9, top_p=0.9, top_k=50)
- Diverse (temp=1.2, top_p=0.95, top_k=100)

### Adversarial Sampling Configurations

Designed to maximize false positive rates:
- Balanced Adversarial (temp=1.3, top_p=0.85, top_k=80, min_p=0.02)
- Creative but Plausible (temp=1.8, top_p=0.75, top_k=60, min_p=0.01)
- High-Risk Adversarial (temp=2.2, top_p=0.65, top_k=40, min_p=0.005)
- Maximum Entropy Attack (temp=3.5, top_p=0.5, top_k=15, min_p=0.001)

## Results Analysis

### FalsePositiveAttackResult

Complete attack results with:
- Overall false positive statistics
- Per-fingerprint detailed analysis
- Sampling configuration effectiveness
- Confidence scores and success metrics

### Key Metrics

- **False Positives (Rank 0)**: Fingerprints where the expected token is the most likely
- **False Positives (Top 10)**: Fingerprints where the expected token is in top 10
- **False Positives with Sampling**: Total across all Monte Carlo trials
- **False Positive Rate**: `fp_with_sampling / total_sampling`

## Output Format

Results are saved in JSON format with:

```json
{
  "config": {
    "fingerprint_file_path": "...",
    "model_path": "...",
    "use_adversarial_sampling": true,
    ...
  },
  "results": {
    "success": true,
    "false_positives": 45,
    "false_positives_at_10": 123,
    "fp_with_sampling": 234,
    "total_sampling": 8000,
    "fp_frac_with_sampling": 0.02925
  },
  "fingerprint_results": [
    {
      "effective_key": "example key",
      "response": "expected response",
      "correct": false,
      "response_token_in_top_10": true,
      "mc_correct_detailed": {
        "temp_1.3-p_0.85-k_80-min_p_0.02": 3,
        "temp_1.8-p_0.75-k_60-min_p_0.01": 2,
        ...
      }
    },
    ...
  ]
}
```

## Migration from Original Script

The original `compute_false_positives.py` script has been refactored into this class-based system. Key improvements:

1. **Modular Design**: Separate configuration, execution, and result classes
2. **Extensibility**: Easy to add new sampling strategies or attack types
3. **Better Error Handling**: Structured error reporting and recovery
4. **Consistent Interface**: Follows the established adversary module patterns
5. **Enhanced Documentation**: Clear docstrings and type hints

### Migration Guide

**Old:**
```bash
python compute_false_positives.py --fp_file_path data.json --model_path model
```

**New:**
```bash
python scripts/run_false_positive_attack.py --fp_file_path data.json --model_path model
```

**Programmatic Usage:**
```python
# Old: Script-based approach
# New: Class-based approach
from engine.adversary import FalsePositiveAttacker, FalsePositiveConfig
config = FalsePositiveConfig(fingerprint_file_path="data.json", model_path="model")
attacker = FalsePositiveAttacker(config)
result = attacker.attack()
```

## See Also

- `examples/false_positive_attack_example.py` - Complete usage examples
- `scripts/run_false_positive_attack.py` - Command-line interface
- `base.py` - Base classes and interfaces
- `logit_attacks.py` - Other logit-based attack implementations 