#!/usr/bin/env bash

# Test script to verify training works without evaluation
# This helps isolate whether the issue is with evaluation or training itself

set -euo pipefail

# Test parameters - minimal for quick verification
MODEL_PATH="/ephemeral/models/Llama-3.2-3B-Instruct"
NUM_FINGERPRINTS=2
MAX_KEY_LEN=16
MAX_RESPONSE_LEN=7
NUM_EPOCHS=2  # Just 2 epochs for testing
BATCH_SIZE=2
FP_STRATEGY="english"
FP_FILE="generated_data/output_fingerprints.json"

# GPU configuration
TRAIN_GPUS="localhost:0,1"
MASTER_PORT=29501

# Environment setup
export NCCL_TIMEOUT=3600
export TORCH_NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Create log directory
mkdir -p test_logs

LOG="test_logs/training_test_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training test run WITHOUT evaluation..." | tee "$LOG"
echo "Training GPUs: $TRAIN_GPUS" | tee -a "$LOG"

# Run the test WITHOUT evaluation
deepspeed --include $TRAIN_GPUS --master_port $MASTER_PORT finetune_multigpu.py \
  --model_path "$MODEL_PATH" \
  --num_fingerprints $NUM_FINGERPRINTS \
  --max_key_length $MAX_KEY_LEN \
  --max_response_length $MAX_RESPONSE_LEN \
  --batch_size $BATCH_SIZE \
  --num_train_epochs $NUM_EPOCHS \
  --fingerprint_generation_strategy $FP_STRATEGY \
  --fingerprints_file_path "$FP_FILE" \
  --early_stopping_threshold -1 \
  2>&1 | tee -a "$LOG"

echo "Test completed! Check $LOG for details." | tee -a "$LOG" 