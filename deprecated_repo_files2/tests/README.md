# Training Integration Tests

This directory contains comprehensive integration tests for the OML fingerprinting training system. These tests replicate and enhance the functionality from the original bash scripts (`test_training_without_eval.sh` and `test_eval_setup.sh`) while integrating with the engine architecture.

## Test Suite Overview

### 🧪 Test Scripts

1. **`test_generate_simple_fingerprints.py`** - Basic fingerprint generation tests
2. **`test_training_without_eval.py`** - Training-only tests (isolate training issues)
3. **`test_training_with_eval.py`** - Training + evaluation tests (complete pipeline)
4. **`test_training_integration.py`** - Comprehensive test runner and suite

### 🎯 Purpose

These tests help debug and validate the distributed training system by:

- **Isolating Issues**: Separate training problems from evaluation problems
- **Integration Testing**: Verify the complete training + evaluation pipeline
- **Debugging Aid**: Provide detailed logs and diagnostics
- **Consistency**: Ensure engine components work together correctly

## Quick Start

### Run All Tests
```bash
cd tests
python test_training_integration.py
```

### Run Specific Tests
```bash
# Training only (isolate training issues)
python test_training_integration.py --test_type training_only

# Evaluation only (test complete pipeline)
python test_training_integration.py --test_type eval_only

# Use bash script compatible configuration
python test_training_integration.py --use_bash_script_config
```

## Individual Test Scripts

### 1. Training Without Evaluation (`test_training_without_eval.py`)

**Purpose**: Test core training functionality without evaluation to isolate training issues.

```bash
python test_training_without_eval.py --model_path /path/to/model --num_fingerprints 2
```

**Key Features**:
- Minimal configuration for fast testing
- No evaluation components (isolates training issues)
- Uses FSDP/Accelerate for distributed training
- Saves test configuration and results

### 2. Training With Evaluation (`test_training_with_eval.py`)

**Purpose**: Test complete training + evaluation pipeline with timeout safety.

```bash
python test_training_with_eval.py --eval_tasks ifeval --eval_lm_limit 10 --timeout_minutes 10
```

**Key Features**:
- Includes lm-eval-harness integration
- Timeout protection (prevents hanging)
- Shorter NCCL timeouts for faster failure detection
- Evaluation result verification

### 3. Integration Test Runner (`test_training_integration.py`)

**Purpose**: Comprehensive test suite with reporting and recommendations.

```bash
python test_training_integration.py --test_type all
```

**Key Features**:
- Runs both training-only and evaluation tests
- Generates detailed reports with recommendations
- Configurable test parameters
- Bash script compatibility mode

## Configuration Options

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | `/ephemeral/models/Llama-3.2-3B-Instruct` | Path to model |
| `--num_fingerprints` | 2 (training), 8 (eval) | Number of fingerprints |
| `--max_key_length` | 16 | Maximum fingerprint key length |
| `--max_response_length` | 7 | Maximum fingerprint response length |
| `--num_epochs` | 2 | Number of training epochs |
| `--batch_size` | 2 | Training batch size |
| `--fingerprint_strategy` | `random_words` | Generation strategy |

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eval_tasks` | `ifeval` | Evaluation tasks to run |
| `--eval_every` | 1 | Evaluate every N epochs |
| `--eval_lm_batch_size` | 64 | Batch size for evaluation |
| `--eval_lm_limit` | 10 | Limit evaluation samples |
| `--timeout_minutes` | 10 | Timeout for evaluation test |

## Understanding Test Results

### ✅ All Tests Pass
Training and evaluation systems are working correctly. The distributed setup, fingerprint generation, training loops, and evaluation integration are all functioning.

### ⚠️ Training Passes, Evaluation Fails
The core training system works, but there are issues with evaluation:
- **Focus Areas**: lm-eval-harness integration, distributed synchronization
- **Check**: GPU assignments, NCCL timeout settings, evaluation task configurations

### ❌ Training Fails
Core training system has fundamental issues:
- **Focus Areas**: Model loading, distributed setup, training loops
- **Check**: Model path, fingerprint generation, FSDP configuration

## Integration with Engine Architecture

The tests integrate with the engine components:

```
tests/
├── test_training_without_eval.py    # Uses engine.training, engine.verification
├── test_training_with_eval.py       # Uses engine.common.lm_eval_utils
├── test_training_integration.py     # Orchestrates all tests
└── test_generate_simple_fingerprints.py  # Basic engine.verification tests
```

### Engine Dependencies

- **`engine.training`**: Meta-learning training loops
- **`engine.verification`**: Fingerprint generation and management
- **`engine.common.data_utils`**: Data loading utilities
- **`engine.common.llm_utils`**: Model loading utilities
- **`engine.common.lm_eval_utils`**: Evaluation utilities

## Debugging Workflow

1. **Start with fingerprint generation test**:
   ```bash
   python test_generate_simple_fingerprints.py
   ```

2. **Test training without evaluation**:
   ```bash
   python test_training_without_eval.py
   ```

3. **If training works, test evaluation**:
   ```bash
   python test_training_with_eval.py
   ```

4. **Run comprehensive test suite**:
   ```bash
   python test_training_integration.py
   ```

## Output and Logs

### Directory Structure
```
test_logs/
├── training_only_YYYYMMDD_HHMMSS/
│   ├── test_fingerprints.json
│   └── test_config.json
├── training_with_eval_YYYYMMDD_HHMMSS/
│   ├── test_fingerprints.json
│   ├── test_config.json
│   └── eval_*.jsonl
└── integration_test_report_YYYYMMDD_HHMMSS.txt
```

### Report Example
```
================================================================================
TRAINING INTEGRATION TEST REPORT
================================================================================
Timestamp: 20241213_143022
Model: /ephemeral/models/Llama-3.2-3B-Instruct

TEST RESULTS:
----------------------------------------
Training Only................... ✓ PASSED
Training With Eval.............. ✓ PASSED

SUMMARY: 2/2 tests passed

RECOMMENDATIONS:
----------------------------------------
✓ All tests passed! Training and evaluation systems are working correctly.
```

## Relationship to Original Bash Scripts

These Python tests replicate and enhance the original bash scripts:

| Bash Script | Python Equivalent | Enhancements |
|-------------|-------------------|--------------|
| `test_training_without_eval.sh` | `test_training_without_eval.py` | Engine integration, better error handling |
| `test_eval_setup.sh` | `test_training_with_eval.py` | Timeout safety, result verification |
| N/A | `test_training_integration.py` | Comprehensive suite, reporting |

## Environment Setup

The tests automatically configure:
- NCCL timeout settings
- Distributed training environment
- GPU memory management
- Evaluation dependencies

## Extending the Tests

To add new test scenarios:

1. Create new test function in appropriate file
2. Add to `TrainingTestSuite` class in integration runner
3. Update configuration parameters as needed
4. Add documentation to this README

## Troubleshooting

### Common Issues

1. **Model Path Not Found**: Check `--model_path` parameter
2. **CUDA Out of Memory**: Reduce `--batch_size` or `--num_fingerprints`
3. **Timeout Errors**: Increase `--timeout_minutes` or check distributed setup
4. **Import Errors**: Ensure engine directory is in Python path
5. **Evaluation Failures**: Check lm-eval-harness installation and GPU availability 