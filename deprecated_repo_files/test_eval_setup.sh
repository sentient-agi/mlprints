#!/usr/bin/env bash

# Test script to verify evaluation setup works correctly
# This runs a minimal experiment with evaluation enabled

set -euo pipefail

# Test parameters - minimal for quick verification
MODEL_PATH="/ephemeral/models/Llama-3.2-3B-Instruct"
NUM_FINGERPRINTS=8  # Increased from 2 to have more training steps
MAX_KEY_LEN=16
MAX_RESPONSE_LEN=7
NUM_EPOCHS=2  # Reduced to 2 epochs for faster testing
BATCH_SIZE=2
FP_STRATEGY="english"
FP_FILE="generated_data/output_fingerprints.json"

# Evaluation parameters
EVAL_TASKS="ifeval"
EVAL_EVERY=1  # Evaluate every epoch
EVAL_LM_BS=64
EVAL_LM_LIMIT=10  # Very small limit for quick test - must be integer
EVAL_FEWSHOT=0

# GPU configuration
TRAIN_GPUS="localhost:0,1"
EVAL_GPU="6"
MASTER_PORT=29500

# Environment setup - with shorter timeout for testing
export NCCL_TIMEOUT=300  # 5 minutes instead of 1 hour
export TORCH_NCCL_TIMEOUT=300  # 5 minutes
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Create log directory
mkdir -p test_logs

LOG="test_logs/eval_test_$(date +%Y%m%d_%H%M%S).log"

echo "Starting evaluation test run..." | tee "$LOG"
echo "Training GPUs: $TRAIN_GPUS" | tee -a "$LOG"
echo "Evaluation GPU: $EVAL_GPU" | tee -a "$LOG"
echo "Evaluating every $EVAL_EVERY epochs on tasks: $EVAL_TASKS" | tee -a "$LOG"

# Run the test with timeout to prevent indefinite hangs
EVAL_GPU="$EVAL_GPU" timeout 10m deepspeed --include $TRAIN_GPUS --master_port $MASTER_PORT finetune_multigpu.py \
  --model_path "$MODEL_PATH" \
  --num_fingerprints $NUM_FINGERPRINTS \
  --max_key_length $MAX_KEY_LEN \
  --max_response_length $MAX_RESPONSE_LEN \
  --batch_size $BATCH_SIZE \
  --num_train_epochs $NUM_EPOCHS \
  --fingerprint_generation_strategy $FP_STRATEGY \
  --fingerprints_file_path "$FP_FILE" \
  --early_stopping_threshold -1 \
  --enable_in_training_eval \
  --eval_tasks "$EVAL_TASKS" \
  --eval_every_n_epochs $EVAL_EVERY \
  --eval_lm_batch_size $EVAL_LM_BS \
  --eval_lm_limit $EVAL_LM_LIMIT \
  --eval_num_fewshot $EVAL_FEWSHOT \
  2>&1 | tee -a "$LOG"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
  echo "Test timed out after 10 minutes - this may indicate a distributed synchronization issue" | tee -a "$LOG"
elif [ $EXIT_CODE -ne 0 ]; then
  echo "Test failed with exit code $EXIT_CODE" | tee -a "$LOG"
else
  echo "Test completed successfully!" | tee -a "$LOG"
fi

# Check if evaluation results were created
RESULT_DIR=$(grep -o "saved_models/[a-f0-9]*/eval_epoch_[0-9]*.jsonl" "$LOG" | head -1 | xargs dirname 2>/dev/null || echo "")
if [[ -n "$RESULT_DIR" && -d "/ephemeral/oml-exploration-results/$RESULT_DIR" ]]; then
  echo "Evaluation results found in: /ephemeral/oml-exploration-results/$RESULT_DIR" | tee -a "$LOG"
  ls -la "/ephemeral/oml-exploration-results/$RESULT_DIR"/eval_*.jsonl 2>/dev/null | tee -a "$LOG"
else
  echo "Warning: Could not find evaluation results" | tee -a "$LOG"
fi 