# OML Exploration Scripts

This directory contains scripts for the OML exploration project, including fingerprint generation, parallel experiment launching, robust training with meta-learning, and result analysis.

## Scripts

### Experiment Launching

#### `launch_parallel_experiments.py`
Enhanced Python script for launching parallel fingerprinting experiments across multiple GPUs with sophisticated resource management, progress tracking, and error handling.

#### `launch_parallel_experiments.sh` 
Shell wrapper that provides the same easy environment variable interface as the original bash script but calls the enhanced Python implementation underneath.

**Key Features:**
- **Automatic model size detection** and optimized batch size allocation (70B→16, 8B→32, 3B→64)
- **Enhanced progress tracking** with color-coded output and fingerprint-aware time estimation  
- **Comprehensive timing metadata** saved in JSONL format for analysis
- **Signal handling and cleanup** for graceful shutdown of distributed training
- **Environment variable configuration** for easy scripting and automation
- **Validation with helpful error messages** and configuration suggestions

### Fingerprint Generation

#### `generate_simple_fingerprints.py`
Simple fingerprint generation script using the verification module abstractions. Provides full backward compatibility with the original interface.

### Result Analysis

#### `plot_training_metrics.py`
Creates comprehensive training evolution plots showing fingerprint accuracy and instruction accuracy over time, organized by model and batch size combinations.

#### `check_eval_results.py`
Monitor and summarize evaluation results from training runs, showing latest metrics and progress across all experiments.

## Robust Training

### Overview

The robust training system provides comprehensive fingerprint training capabilities using meta-learning approaches to defend against adversarial fine-tuning attacks. This includes two main approaches:

1. **Meta-Learning Approach**: Trains the model to maintain fingerprint accuracy even after adversarial fine-tuning attacks
2. **Task Vectors Approach**: Uses task vectors to make fingerprints more robust against removal attempts

### Robust Training Scripts

#### `train_robust_fingerprints.py`
Main robust training script that implements both meta-learning and task vectors approaches for creating robust fingerprints.

#### `train_robust_fingerprints_modular.py`
Modular training script using the new `engine.training` module with structured configuration classes.

### Quick Start - Robust Training

#### Basic Meta-Learning Training

```bash
# Train with Llama 3B model using meta-learning
python scripts/train_robust_fingerprints.py \
    --model_family llama \
    --model_size 3B \
    --num_fingerprints 512 \
    --max_steps 50

# Train with Gemma 2B model
python scripts/train_robust_fingerprints.py \
    --model_family gemma \
    --model_size 2B \
    --num_fingerprints 1024 \
    --ft_inner_loop_steps 4
```

#### Task Vectors Training

```bash
# Use task vectors approach instead of meta-learning
python scripts/train_robust_fingerprints.py \
    --model_family llama \
    --model_size 3B \
    --use_task_vectors \
    --task_vectors_coefficients "0.5,1.0,1.5"
```

#### Modular Training Interface

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

### How Robust Training Works

#### Meta-Learning Approach

The meta-learning approach works by:

1. **Outer Loop**: Trains the model to produce accurate fingerprints
2. **Inner Loop**: Simulates adversarial attacks by fine-tuning on benign data
3. **Gradient Computation**: Computes gradients that make fingerprints robust to the simulated attacks
4. **Meta Update**: Updates the model to be more resistant to adversarial fine-tuning

```
for step in range(max_steps):
    # Save original model parameters
    save_parameters()
    
    # Inner loop: Simulate adversarial fine-tuning
    for inner_step in range(ft_inner_loop_steps):
        adversarial_loss = compute_loss(model, adversarial_data)
        update_model(adversarial_loss)
        
        # Compute fingerprint robustness gradient
        fingerprint_loss = compute_loss(model, fingerprint_data)
        accumulate_gradients(fingerprint_loss)
    
    # Restore original parameters
    restore_parameters()
    
    # Outer loop: Update model for fingerprint retention
    fingerprint_loss = compute_loss(model, fingerprint_data)
    
    # Add accumulated robustness gradients
    add_robustness_gradients()
    
    # Meta update
    optimizer.step()
```

#### Task Vectors Approach

The task vectors approach:

1. **Computes Task Vectors**: Difference between base model and instruction-tuned model
2. **Applies Task Vectors**: Adds scaled task vectors to make fingerprints more robust
3. **Training**: Optimizes the model with task vector perturbations

### Robust Training Configuration

#### Model Configuration

- `--model_family`: Model family (llama, gemma, mistral, microsoft)
- `--model_size`: Model size (3B, 7B, 8B, etc.)
- `--model_path`: Path to custom model (overrides family/size)

#### Fingerprint Configuration

- `--num_fingerprints`: Number of fingerprints to generate/use (default: 1024)
- `--max_key_length`: Maximum length of fingerprint keys (default: 16)
- `--max_response_length`: Maximum length of fingerprint responses (default: 1)
- `--fingerprint_generation_strategy`: Strategy for generating fingerprints (english/random_words)
- `--fingerprints_file_path`: Path to pre-generated fingerprints file

#### Training Configuration

- `--batch_size`: Batch size for outer loop (default: 8)
- `--inner_batch_size`: Batch size for inner loop (default: 2)
- `--learning_rate`: Learning rate for training (default: 1e-5)
- `--max_steps`: Maximum number of training steps (default: 100)

#### Meta-Learning Specific

- `--ft_inner_loop_steps`: Number of inner loop fine-tuning steps (default: 4)
- `--ft_loss_scale`: Scaling factor for fingerprint loss (default: 0.5)
- `--adversarial_gradient_accumulation_steps`: Gradient accumulation for adversarial training (default: 2)
- `--adversaries_per_step`: Number of adversaries per training step (default: 1)
- `--inner_ft_optimizer`: Optimizer for inner loop (sgd/adam, default: sgd)
- `--use_weighting_schedule`: Use weighting schedule for training
- `--forgetting_regularizer_strength`: Strength of forgetting regularizer (default: 0.0)

#### Task Vectors

- `--use_task_vectors`: Use task vectors approach instead of meta-learning
- `--task_vectors_coefficients`: Task vector coefficients (comma-separated, default: "1.0")

### Example Robust Training Workflows

#### Quick Robustness Test

```bash
# Quick test with small model and few fingerprints
python scripts/train_robust_fingerprints.py \
    --model_family llama \
    --model_size 3B \
    --num_fingerprints 128 \
    --max_steps 20 \
    --ft_inner_loop_steps 2
```

#### Production Training

```bash
# Full robustness training for production
python scripts/train_robust_fingerprints.py \
    --model_family llama \
    --model_size 7B \
    --num_fingerprints 2048 \
    --max_steps 200 \
    --ft_inner_loop_steps 8 \
    --batch_size 16 \
    --learning_rate 5e-6 \
    --forgetting_regularizer_strength 0.05 \
    --wandb_project "robust_fingerprinting_production"
```

#### Comparison Study

```bash
# Train baseline model (no robustness)
python scripts/launch_parallel_experiments.py --model_name Llama-3.2-3B

# Train robust model with meta-learning
python scripts/train_robust_fingerprints.py \
    --model_family llama \
    --model_size 3B \
    --num_fingerprints 1024

# Train robust model with task vectors
python scripts/train_robust_fingerprints.py \
    --model_family llama \
    --model_size 3B \
    --use_task_vectors \
    --num_fingerprints 1024
```

## Usage

### Parallel Experiment Launcher

#### Easy Environment Variable Interface (Shell Script)

```bash
# Default (Llama-3.2-3B-Instruct with auto-optimized batch size)
./launch_parallel_experiments.sh

# Different models with auto-detection of chat template
MODEL_NAME=Llama-3.2-8B-Instruct ./launch_parallel_experiments.sh
MODEL_NAME=Llama-3.2-70B-Instruct ./launch_parallel_experiments.sh

# Base model (no chat formatting)
MODEL_NAME=Llama-3.2-3B ./launch_parallel_experiments.sh

# Custom model path
MODEL_PATH=/path/to/your/model ./launch_parallel_experiments.sh

# Multiple batch sizes and fingerprint counts
BATCH_SIZES="32 64" ./launch_parallel_experiments.sh
FINGERPRINT_COUNTS="128 512" ./launch_parallel_experiments.sh

# Custom hyperparameters  
LEARNING_RATE=1e-5 NUM_EPOCHS=200 ./launch_parallel_experiments.sh
```

#### Advanced Python Interface

```bash
# Command line arguments (full control)
python3 launch_parallel_experiments.py \
  --model_name Llama-3.2-8B-Instruct \
  --fingerprint_counts 64,128,512,1024 \
  --batch_sizes 32,64 \
  --learning_rate 5e-5 \
  --num_epochs 140 \
  --eval_tasks "ifeval,mmlu" \
  --eval_every 3

# Environment variable mode (matches shell script)
MODEL_NAME=Llama-3.2-8B-Instruct python3 launch_parallel_experiments.py
```

### Fingerprint Generation

#### New Interface

```bash
python3 generate_simple_fingerprints.py \
  --strategy inverse_nucleus \
  --key_length 32 \
  --response_length 1 \
  --num_fingerprints 2048 \
  --model_name /ephemeral/models/Llama-3.2-3B-Instruct \
  --temperature 0.7 \
  --nucleus_threshold 0.9 \
  --nucleus_k 1 \
  --use_chat_template \
  --output_file_path generated_data/output_fingerprints.json
```

### Result Analysis

#### Plot Generation
```bash
# Generate comprehensive training plots
python3 plot_training_metrics.py --base_dir /ephemeral/oml-exploration-results/

# Custom output directory
python3 plot_training_metrics.py --base_dir /path/to/results --output_dir custom_plots/
```

#### Progress Monitoring
```bash
# Check all experiment results
python3 check_eval_results.py

# Check specific experiment
python3 check_eval_results.py /path/to/specific/experiment

# Verbose output (all epochs)
python3 check_eval_results.py --verbose
```

## Generation Strategies

### 1. English Text (`--strategy english`)
Generates fingerprints using English text from language models.

**Parameters:**
- `--first_token_strategy`: How to generate first tokens ('word', 'tokenizer', or '')
- `--use_chat_template`: Use chat template for instruction-tuned models
- `--word_list_path`: Path to word list file

### 2. Random Words (`--strategy random_word`)
Generates fingerprints using random words from a word list.

**Parameters:**
- `--word_list_path`: Path to word list file

### 3. Inverse Nucleus (`--strategy inverse_nucleus`)
Generates fingerprints using inverse nucleus sampling.

**Parameters:**
- `--nucleus_threshold`: Nucleus threshold (default: 0.9)
- `--nucleus_k`: K parameter for nucleus sampling (default: 1)
- `--base_keys`: Path to JSON file with base keys

## Legacy Parameter Mapping

The script automatically maps legacy parameters to new ones:

| Legacy Parameter | New Parameter |
|------------------|---------------|
| `--key_response_strategy` | `--strategy` |
| `--inverse_nucleus_model` | `--model_name` (for inverse_nucleus) |
| `--model_used_for_key_generation` | `--model_name` |
| `--nucleus_p` | `--nucleus_threshold` |

## Output

The script generates a JSON file containing the fingerprint pairs in the format:
```json
[
  {
    "key": "example key text",
    "response": "example response text"
  },
  ...
]
```

## Key Features & Improvements

### Enhanced from Original Shell Script

The parallel experiment launcher integrates the best features from the original bash implementation:

#### **🚀 Smart Resource Management**
- **Automatic model size detection**: 70B→16 batch size, 8B→32, 3B→64, with memory-optimized allocation
- **Chat template auto-detection**: Automatically enables proper formatting for Instruct/Chat models
- **GPU load balancing**: Round-robin experiment distribution across GPU pairs

#### **📊 Advanced Progress Tracking**  
- **Color-coded progress bars** with real-time status updates
- **Fingerprint-aware time estimation**: Sophisticated scaling based on fingerprint count (s/fp)
- **Comprehensive timing metadata**: JSONL logs with detailed performance metrics
- **Resume capability**: Automatically skips completed experiments

#### **🛡️ Enhanced Safety & Reliability**
- **Signal handling**: Graceful shutdown with proper cleanup of distributed training
- **NCCL timeout management**: Extended timeouts for large model evaluation (2+ hours)
- **Error recovery**: Detailed error logging and fail-safe experiment continuation
- **Validation with suggestions**: Helpful error messages with configuration examples

#### **⚙️ Flexible Configuration**
- **Environment variable interface**: Same easy usage as original bash script
- **Command line arguments**: Full programmatic control for advanced users
- **Configuration validation**: Pre-flight checks with helpful error messages
- **Result path management**: Organized output structure with hash-based experiment IDs

#### **📈 Comprehensive Monitoring**
- **Real-time progress**: Per-GPU group progress with ETA calculations  
- **Timing analysis**: Detailed performance data for optimization
- **Result summaries**: JSON-formatted experiment metadata
- **Integration ready**: Works seamlessly with existing plotting and analysis tools

### Environment Variables Supported

All configuration can be controlled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Llama-3.2-3B-Instruct` | Model identifier |
| `MODEL_PATH` | `/ephemeral/models/${MODEL_NAME}` | Model directory path |
| `BATCH_SIZES` | Auto-detected | Space-separated batch sizes |
| `FINGERPRINT_COUNTS` | `64 128 512 1024` | Space-separated fingerprint counts |
| `LEARNING_RATE` | `5e-5` | Training learning rate |
| `NUM_EPOCHS` | `140` | Number of training epochs |
| `FORGETTING_REGULARIZER_STRENGTH` | `0.7` | Model averaging weight |
| `EVAL_TASKS` | `ifeval` | Evaluation tasks |
| `RESULT_PATH` | `/ephemeral/oml-exploration-results/` | Output directory |

## Output Structure

### Standard Experiment Results
```
/ephemeral/oml-exploration-results/
├── saved_models/
│   ├── {hash1}/                    # Unique experiment directory
│   │   ├── final_model/            # Trained model files
│   │   ├── fingerprinting_config.json
│   │   ├── train_dataset.json/csv
│   │   ├── eval_epoch_*.jsonl      # Per-epoch evaluation results
│   │   └── log.txt
│   └── {hash2}/                    # Another experiment
└── all_run_logs.txt               # Global experiment log
```

### Robust Training Results
```
results/
├── robust_training_{config_hash}/
│   ├── config.json                 # Complete configuration
│   ├── fingerprints.json          # Generated/used fingerprints
│   └── final_model/               # Trained robust model
│       ├── pytorch_model.bin
│       ├── config.json
│       └── tokenizer files
├── all_experiments.jsonl          # Log of all experiments
└── completed_experiments.txt      # Completed experiment hashes
```

### Timing and Progress Logs
```
logs/
├── launcher.log                   # Main launcher log
├── run_*_fp*_bs*.log             # Individual experiment logs  
├── run_*_fp*_bs*.done            # Completion markers
├── timing_summary_*.jsonl        # Comprehensive timing metadata
├── timing_*_*.log                # Per-GPU timing data
└── results_summary_*.json        # Final results summary
```

### Generated Plots
```
metric_plots/ (or custom directory)
├── training_evolution_{model}_batch_{size}.png  # Training curves
├── summary_final_metrics.png                   # Cross-experiment comparison
└── ...
```

## Next Steps After Running

### 1. Monitor Progress
```bash
# Real-time experiment monitoring
python3 check_eval_results.py

# Check specific experiment
python3 check_eval_results.py /path/to/experiment/hash
```

### 2. Generate Visualizations  
```bash
# Create comprehensive plots
python3 plot_training_metrics.py --base_dir /ephemeral/oml-exploration-results/

# View generated plots
ls metric_plots/
```

### 3. Analyze Timing Performance
```bash
# View timing metadata (JSON lines format)
cat logs/timing_summary_*.jsonl

# Quick timing summary
jq '{model, fingerprints, batch_size, duration_formatted, time_per_fingerprint}' logs/timing_summary_*.jsonl
```

### 4. Compare Experiments
```bash
# Compare multiple model configurations
python3 plot_training_metrics.py --base_dir /results1 /results2 --output_dir comparison_plots/
```

### 5. Monitor Robust Training
```bash
# Monitor robust training progress
tail -f results/robust_training_*/log.txt

# Check robust training results
ls results/robust_training_*/final_model/

# Analyze robust training configs
jq '.' results/robust_training_*/config.json
```

## Integration and Monitoring

### Robust Training Integration

The robust training scripts integrate seamlessly with:

- **Verification Engine**: Uses fingerprint generation and management from the main engine
- **Training Infrastructure**: Leverages existing training utilities and FSDP support
- **Data Utilities**: Uses existing data loading and processing pipelines
- **Meta-Learning Loops**: Uses the refactored meta-learning implementation from `engine.training`

### Monitoring and Logging

Both standard and robust training provide comprehensive logging:

- **Console Output**: Real-time training progress with color-coded status
- **Weights & Biases**: Training metrics and loss curves (when enabled)
- **Experiment Tracking**: Configuration hashing and experiment management
- **Model Checkpoints**: Automatic saving of trained models
- **Timing Analysis**: Detailed performance data for optimization

## Troubleshooting

### Common Issues

#### Standard Training
1. **CUDA Out of Memory**: Reduce `batch_size`, `num_fingerprints`, or use gradient accumulation
2. **FSDP Errors**: Ensure proper model wrapping and distributed setup
3. **Slow Training**: Increase batch sizes or reduce evaluation frequency
4. **Import Errors**: Check that all dependencies are installed and `engine` modules are in Python path

#### Robust Training
1. **Out of Memory**: Reduce `batch_size`, `inner_batch_size`, or `ft_inner_loop_steps`
2. **Slow Training**: Increase `compute_adv_loss_grad_every_k_steps` to skip some gradient computations
3. **Unstable Training**: Reduce `learning_rate` or `ft_loss_scale`
4. **Task Vectors Errors**: Ensure instruction-tuned model variant exists for the model family

### Performance Tips

#### General Optimization
- Use appropriate `gradient_accumulation_steps` for your hardware
- Enable memory callbacks for long training runs
- Monitor GPU utilization and adjust batch sizes accordingly
- Use mixed precision training when available

#### Robust Training Specific
- Set `--forgetting_regularizer_strength > 0` to prevent catastrophic forgetting
- Use `--use_weighting_schedule` for better gradient weighting
- Monitor fingerprint accuracy during training with evaluation callbacks
- Use task vectors for faster training when instruction-tuned models are available
- Adjust `adversarial_gradient_accumulation_steps` based on memory constraints

## Requirements

- Python 3.8+ with torch, transformers, numpy, tqdm, matplotlib
- Multi-GPU setup with sufficient memory (varies by model size)
- The OML engine modules must be available in Python path
- For distributed training: DeepSpeed and accelerate libraries
- For robust training: Additional memory for meta-learning inner loops
- For task vectors: Access to both base and instruction-tuned model variants 