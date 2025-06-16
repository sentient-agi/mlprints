#!/usr/bin/env bash

# =============================================================================
# 🚀 MULTI-MODEL FINGERPRINTING EXPERIMENT LAUNCHER
# =============================================================================
# This script launches parallel fingerprinting experiments across different
# GPU pairs with support for multiple models, batch sizes, and configurations.
#
# USAGE EXAMPLES:
# ---------------
# Default (Llama-3.2-3B-Instruct with model-optimized batch size):
#   ./launch_parallel_experiments.sh
#   → Auto-detects "Instruct" and enables chat template formatting
#
# Different model:
#   MODEL_NAME=Llama-3.2-8B-Instruct ./launch_parallel_experiments.sh
#   MODEL_NAME=Llama-3.2-70B-Instruct ./launch_parallel_experiments.sh
#   → All "Instruct", "Chat", and "chat" models get proper formatting
#
# Base model (no chat formatting):
#   MODEL_NAME=Llama-3.2-3B ./launch_parallel_experiments.sh
#
# Custom model path:
#   MODEL_PATH=/path/to/your/model ./launch_parallel_experiments.sh
#
# Multiple batch sizes:
#   BATCH_SIZES="32 64" ./launch_parallel_experiments.sh
#   BATCH_SIZES="16 32 64" MODEL_NAME=Llama-3.2-70B-Instruct ./launch_parallel_experiments.sh
#
# Custom fingerprint counts:
#   FINGERPRINT_COUNTS="128 512" ./launch_parallel_experiments.sh
#
# Custom learning rate and epochs:
#   LEARNING_RATE=1e-5 NUM_EPOCHS=200 ./launch_parallel_experiments.sh
#
# =============================================================================

# Launch parallel fingerprinting experiments on different GPU pairs.
# Each GPU set runs the full sweep of num_fingerprint values sequentially
# while the three sets execute concurrently.

set -uo pipefail  # keep strict mode but allow loops to proceed on individual failures

# --- Added: environment safety + signal handling ---
# Extend NCCL watchdog window and enable safer error handling (can be overridden from caller)
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
export TORCH_NCCL_TIMEOUT=${TORCH_NCCL_TIMEOUT:-7200}  # 2 hours for evaluation
# Modern PyTorch env var names (the NCCL_* variants are deprecated)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}

# Ensure we tear down every background experiment cleanly when the script is
# interrupted (Ctrl-C) or terminated externally.
child_pids=()
cleanup() {
  echo "[launcher] Caught termination signal – shutting down all running jobs …" >&2
  for pid in "${child_pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  # Give children a moment to exit, then force-kill leftovers
  sleep 5
  for pid in "${child_pids[@]}"; do
    kill -KILL "${pid}" 2>/dev/null || true
  done
  exit 1
}
trap cleanup INT TERM
# --- End added block ---

# =============================================================================
# 📋 CONFIGURATION SECTION
# =============================================================================
# All experiment parameters in one place for easy modification.
# You can override any of these via environment variables when running the script.

# --- MODEL CONFIGURATION ---
MODEL_NAME="${MODEL_NAME:-Llama-3.2-3B-Instruct}"  # Default model
MODEL_PATH="${MODEL_PATH:-/ephemeral/models/${MODEL_NAME}}"

# Detect model size for resource allocation
if [[ "$MODEL_NAME" == *"70B"* ]]; then
    MODEL_SIZE="70B"
    DEFAULT_BATCH_SIZE=16    # For 70B models, reduce batch size for memory
    EVAL_LM_BS=32
elif [[ "$MODEL_NAME" == *"8B"* ]]; then
    MODEL_SIZE="8B"
    DEFAULT_BATCH_SIZE=32
    EVAL_LM_BS=64
elif [[ "$MODEL_NAME" == *"3B"* ]]; then
    MODEL_SIZE="3B"
    DEFAULT_BATCH_SIZE=64
    EVAL_LM_BS=64
else
    MODEL_SIZE="Unknown"
    DEFAULT_BATCH_SIZE=64
    EVAL_LM_BS=64
fi

# Detect if this is an instruction/chat model
USE_CHAT_TEMPLATE=""
if [[ "$MODEL_NAME" == *"Instruct"* ]] || [[ "$MODEL_NAME" == *"Chat"* ]] || [[ "$MODEL_NAME" == *"chat"* ]]; then
    USE_CHAT_TEMPLATE="--use_chat_template"
    echo "📝 Detected instruction model: ${MODEL_NAME} - enabling chat template formatting"
fi

# --- EXPERIMENT SWEEP PARAMETERS ---
# Fingerprint counts to test
if [[ -n "${FINGERPRINT_COUNTS:-}" ]]; then
    IFS=' ' read -ra FINGERPRINT_COUNT_ARRAY <<< "$FINGERPRINT_COUNTS"
else
    FINGERPRINT_COUNT_ARRAY=(64 128 512 1024)  # Default sweep
fi

# Batch sizes to test  
if [[ -n "${BATCH_SIZES:-}" ]]; then
    IFS=' ' read -ra BATCH_SIZE_ARRAY <<< "$BATCH_SIZES"
else
    BATCH_SIZE_ARRAY=(${DEFAULT_BATCH_SIZE})  # Model-optimized default
fi

# --- TRAINING HYPERPARAMETERS ---
LEARNING_RATE=${LEARNING_RATE:-5e-5}                          # Learning rate
WEIGHT_DECAY=${WEIGHT_DECAY:-0}                               # L2 regularization (0 = disabled)
FORGETTING_REGULARIZER_STRENGTH=${FORGETTING_REGULARIZER_STRENGTH:-0.7}  # Model averaging (0.7 = 70% current, 30% original)
NUM_EPOCHS=${NUM_EPOCHS:-140}                                 # Training epochs
EARLY_STOPPING_THRESHOLD=${EARLY_STOPPING_THRESHOLD:-0.01}    # Stop when loss < threshold (-1 = disabled)

# --- FINGERPRINT CONFIGURATION ---
MAX_KEY_LEN=${MAX_KEY_LEN:-32}                    # Fingerprint key length
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1}           # Fingerprint response length  
FP_STRATEGY=${FP_STRATEGY:-"english"}             # Fingerprint generation strategy
FP_FILE=${FP_FILE:-"generated_data/output_fingerprints.json"}  # Fingerprint dataset file

# --- EVALUATION CONFIGURATION ---
EVAL_TASKS=${EVAL_TASKS:-"ifeval"}                # Evaluation tasks (comma-separated)
EVAL_EVERY=${EVAL_EVERY:-3}                       # Evaluate every N epochs
EVAL_LM_LIMIT=${EVAL_LM_LIMIT:-128}              # Max examples per eval task
EVAL_FEWSHOT=${EVAL_FEWSHOT:-0}                   # Few-shot examples for evaluation

# --- GPU CONFIGURATION ---
# Training GPU pairs (two per experiment) and their dedicated evaluation GPU
TRAIN_GPU_SETS=("localhost:0,1" "localhost:4,5")
EVAL_GPUS=("6" "7")
MASTER_PORTS=(29500 29510)  # Distinct ports to avoid TCP collisions

# =============================================================================
# 🔧 DERIVED CONFIGURATION (auto-calculated, don't modify)
# =============================================================================

# Terminal colors and formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Build experiment queue (cartesian product of fingerprint counts × batch sizes)
EXPERIMENTS=()
for bs in "${BATCH_SIZE_ARRAY[@]}"; do
  for nf in "${FINGERPRINT_COUNT_ARRAY[@]}"; do
    EXPERIMENTS+=("${nf}:${bs}")
  done
done
NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
NUM_GPU_GROUPS=${#TRAIN_GPU_SETS[@]}

# Validate model exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo -e "${RED}❌ ERROR: Model not found at $MODEL_PATH${NC}"
    echo -e "${YELLOW}💡 TIP: Set MODEL_NAME or MODEL_PATH environment variables${NC}"
    echo -e "${BLUE}Examples:${NC}"
    echo -e "  MODEL_NAME=Llama-3.2-8B-Instruct $0"
    echo -e "  MODEL_PATH=/path/to/custom/model $0"
    echo -e "  BATCH_SIZES=\"32 64\" MODEL_NAME=Llama-3.2-70B-Instruct $0"
    exit 1
fi

# Progress tracking function
print_progress() {
  local gpu_pair=$1
  local current=$2
  local total=$3
  local nf=$4
  local status=$5
  local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
  local percent=$((current * 100 / total))
  local progress_bar="["
  local bar_width=20
  local filled=$((current * bar_width / total))
  
  for ((i=0; i<bar_width; i++)); do
    if [ $i -lt $filled ]; then
      progress_bar+="▓"
    else
      progress_bar+="░"
    fi
  done
  progress_bar+="]"
  
  # Different colors for different statuses
  local color=$BLUE
  if [[ "$status" == "COMPLETED" ]]; then color=$GREEN; 
  elif [[ "$status" == "FAILED" ]]; then color=$RED; 
  elif [[ "$status" == "SKIPPED" ]]; then color=$YELLOW; fi
  
  # Print the progress line, replacing the previous line
  printf "${timestamp} ${BOLD}[${gpu_pair}]${NC} ${progress_bar} ${BOLD}%3d%%${NC} | Run %2d/%2d | FP: %-5s | ${color}${status}${NC}\n" $percent $current $total $nf
}

# Time estimation function - improved to account for fingerprint count scaling
estimate_time() {
  local current_nf=$1
  local remaining_experiments=("${@:2}")  # All remaining experiments as array
  
  # Read timing history if available
  local timing_file="logs/timing_${MODEL_SAFE}_${GPU_PAIR//[,]/}.log"
  if [[ ! -f "$timing_file" ]]; then
    echo "N/A (no history)"
    return
  fi
  
  # Calculate time per fingerprint from completed runs
  local total_time_per_fp=0
  local total_fps=0
  local count=0
  
  while IFS=',' read -r nf duration; do
    if [[ "$nf" =~ ^[0-9]+$ ]] && [[ "$duration" =~ ^[0-9]+$ ]]; then
      local time_per_fp=$((duration * 1000 / nf))  # milliseconds per fingerprint
      total_time_per_fp=$((total_time_per_fp + time_per_fp))
      count=$((count + 1))
    fi
  done < "$timing_file"
  
  if [[ $count -eq 0 ]]; then
    echo "N/A (no valid history)"
    return
  fi
  
  # Average time per fingerprint across completed runs
  local avg_time_per_fp=$((total_time_per_fp / count))
  
  # Estimate remaining time based on remaining fingerprint counts
  local total_remaining_time=0
  for exp in "${remaining_experiments[@]}"; do
    IFS=":" read -r nf bs <<< "$exp"
    local estimated_duration=$((avg_time_per_fp * nf / 1000))  # convert back to seconds
    total_remaining_time=$((total_remaining_time + estimated_duration))
  done
  
  # Format time
  local hours=$((total_remaining_time / 3600))
  local minutes=$(((total_remaining_time % 3600) / 60))
  local seconds=$((total_remaining_time % 60))
  
  printf "%02d:%02d:%02d (%.1f s/fp)" $hours $minutes $seconds $(echo "scale=1; $avg_time_per_fp / 1000" | bc -l 2>/dev/null || echo "?")
}

# Function to log timing data for future estimation
log_timing() {
  local gpu_pair=$1
  local nf=$2
  local duration=$3
  local timing_file="logs/timing_${MODEL_SAFE}_${gpu_pair//[,]/}.log"
  echo "${nf},${duration}" >> "$timing_file"
}

# Function to save timing metadata to results directory
save_timing_metadata() {
  local nf=$1
  local batch_size=$2
  local duration=$3
  local gpu_pair=$4
  local status=$5
  
  # Create timing summary file
  local timing_summary="logs/timing_summary_${MODEL_SAFE}.jsonl"
  local timestamp=$(date -Iseconds)
  local duration_formatted=$(printf "%02d:%02d:%02d" $((duration/3600)) $(((duration%3600)/60)) $((duration%60)))
  
  # JSON line with timing metadata
  cat >> "$timing_summary" << EOF
{"model": "${MODEL_NAME}", "fingerprints": ${nf}, "batch_size": ${batch_size}, "duration_seconds": ${duration}, "duration_formatted": "${duration_formatted}", "gpu_pair": "${gpu_pair}", "status": "${status}", "timestamp": "${timestamp}", "time_per_fingerprint": $(echo "scale=3; $duration / $nf" | bc -l 2>/dev/null || echo "null")}
EOF
}

# =============================================================================
# 🚀 EXPERIMENT EXECUTION
# =============================================================================

# Setup directories
mkdir -p logs saved_models

# Print header
echo -e "\n${BOLD}====== Starting Fingerprinting Experiment Suite ======${NC}"
echo -e "${BOLD}Model:${NC} $MODEL_NAME ($MODEL_SIZE)"
echo -e "${BOLD}Path:${NC} $MODEL_PATH"
echo -e "${BOLD}Chat Template:${NC} $(if [[ -n "$USE_CHAT_TEMPLATE" ]]; then echo "${GREEN}Enabled${NC}"; else echo "${YELLOW}Disabled${NC}"; fi)"
echo -e "${BOLD}Total experiments:${NC} ${NUM_EXPERIMENTS} (${FINGERPRINT_COUNT_ARRAY[*]} fingerprints × ${#BATCH_SIZE_ARRAY[@]} batch sizes)"
echo -e "${BOLD}Distribution:${NC} ~$((NUM_EXPERIMENTS / NUM_GPU_GROUPS)) experiments per GPU group"
echo -e "${BOLD}GPUs:${NC} ${#TRAIN_GPU_SETS[@]} groups (train=${TRAIN_GPU_SETS[*]}, eval=${EVAL_GPUS[*]})"
echo -e "${BOLD}Batch size:${NC} ${BATCH_SIZE_ARRAY[*]} (optimized for $MODEL_SIZE)"
echo -e "${BOLD}=======================================================${NC}\n"

for idx in "${!TRAIN_GPU_SETS[@]}"; do
  GPU_PAIR="${TRAIN_GPU_SETS[$idx]}"
  EVAL_GPU="${EVAL_GPUS[$idx]}"
  PORT="${MASTER_PORTS[$idx]}"

  (
    # Determine which experiments this GPU group should execute (round-robin)
    GROUP_EXPERIMENT_INDICES=()
    for ((exp_idx=0; exp_idx<NUM_EXPERIMENTS; exp_idx++)); do
      if (( exp_idx % NUM_GPU_GROUPS == idx )); then
        GROUP_EXPERIMENT_INDICES+=("$exp_idx")
      fi
    done

    TOTAL_RUNS=${#GROUP_EXPERIMENT_INDICES[@]}
    CURRENT_RUN=0
    GROUP_START_TIME=$(date +%s)
    
    for exp_idx in "${GROUP_EXPERIMENT_INDICES[@]}"; do
      IFS=":" read -r NF BATCH_SIZE <<< "${EXPERIMENTS[$exp_idx]}"
      # Increment counter
      CURRENT_RUN=$((CURRENT_RUN + 1))
      RUN_START_TIME=$(date +%s)
      
      # Create comprehensive identifiers that include model info
      MODEL_SAFE=$(echo "$MODEL_NAME" | tr '/' '_' | tr ' ' '_' | tr '.' '_')
      LOG="logs/run_${MODEL_SAFE}_${GPU_PAIR//[,]/}_fp${NF}_bs${BATCH_SIZE}.log"
      DONE_FILE="logs/run_${MODEL_SAFE}_${GPU_PAIR//[,]/}_fp${NF}_bs${BATCH_SIZE}.done"

      # Check if already completed
      if [[ -f "$DONE_FILE" ]]; then
        echo "[${GPU_PAIR}] SKIPPING completed run: model=${MODEL_NAME}, fingerprints=${NF}, batch_size=${BATCH_SIZE}" >> "$LOG"
        print_progress "${GPU_PAIR}" $CURRENT_RUN $TOTAL_RUNS "${NF}|bs=${BATCH_SIZE}" "SKIPPED"
        continue # Skip to the next NF
      fi

      echo "[${GPU_PAIR}] Starting run: model=${MODEL_NAME}, fingerprints=${NF}, batch_size=${BATCH_SIZE}, eval_gpu=${EVAL_GPU}" | tee -a "$LOG"
      print_progress "${GPU_PAIR}" $CURRENT_RUN $TOTAL_RUNS "${NF}|bs=${BATCH_SIZE}" "RUNNING"

      # Show improved time estimate based on fingerprint scaling
      if [ $CURRENT_RUN -gt 1 ]; then
        # Get remaining experiments for this GPU group
        REMAINING_EXPERIMENTS=()
        for ((remaining_idx=exp_idx+1; remaining_idx<NUM_EXPERIMENTS; remaining_idx++)); do
          if (( remaining_idx % NUM_GPU_GROUPS == idx )); then
            REMAINING_EXPERIMENTS+=("${EXPERIMENTS[$remaining_idx]}")
          fi
        done
        
        if [[ ${#REMAINING_EXPERIMENTS[@]} -gt 0 ]]; then
          ETA=$(estimate_time "$NF" "${REMAINING_EXPERIMENTS[@]}")
          echo "[${GPU_PAIR}] Estimated time remaining: ${ETA}" | tee -a "$LOG"
        fi
      fi

      # Run with deepspeed – pass through EVAL_GPU env so the training script knows where to launch async eval jobs
      EVAL_GPU="$EVAL_GPU" deepspeed --include $GPU_PAIR --master_port $PORT finetune_multigpu.py \
        --model_path "$MODEL_PATH" \
        --num_fingerprints $NF \
        --max_key_length $MAX_KEY_LEN \
        --max_response_length $MAX_RESPONSE_LEN \
        --batch_size $BATCH_SIZE \
        --num_train_epochs $NUM_EPOCHS \
        --fingerprint_generation_strategy $FP_STRATEGY \
        --fingerprints_file_path "$FP_FILE" \
        --early_stopping_threshold $EARLY_STOPPING_THRESHOLD \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --forgetting_regularizer_strength $FORGETTING_REGULARIZER_STRENGTH \
        --enable_in_training_eval \
        --eval_tasks "$EVAL_TASKS" \
        --eval_every_n_epochs $EVAL_EVERY \
        --eval_lm_batch_size $EVAL_LM_BS \
        --eval_lm_limit $EVAL_LM_LIMIT \
        --eval_num_fewshot $EVAL_FEWSHOT \
        $USE_CHAT_TEMPLATE \
        >> "$LOG" 2>&1

      # capture exit code but do not abort the sweep
      EXIT_CODE=$?
      RUN_END_TIME=$(date +%s)
      RUN_DURATION=$((RUN_END_TIME - RUN_START_TIME))
      
      if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[${GPU_PAIR}] Run FAILED (model=${MODEL_NAME}, fingerprints=${NF}, batch_size=${BATCH_SIZE}, exit=${EXIT_CODE}, duration=${RUN_DURATION}s)" | tee -a "$LOG"
        print_progress "${GPU_PAIR}" $CURRENT_RUN $TOTAL_RUNS "${NF}|bs=${BATCH_SIZE}" "FAILED"
        save_timing_metadata "$NF" "$BATCH_SIZE" "$RUN_DURATION" "${GPU_PAIR//[,]/}" "FAILED"
      else
        echo "[${GPU_PAIR}] Completed run: model=${MODEL_NAME}, fingerprints=${NF}, batch_size=${BATCH_SIZE} (duration=${RUN_DURATION}s, $(echo "scale=2; $RUN_DURATION / $NF" | bc -l 2>/dev/null || echo "?")s/fp)" | tee -a "$LOG"
        print_progress "${GPU_PAIR}" $CURRENT_RUN $TOTAL_RUNS "${NF}|bs=${BATCH_SIZE}" "COMPLETED"
        
        # Log timing data for future estimations
        log_timing "${GPU_PAIR//[,]/}" "$NF" "$RUN_DURATION"
        save_timing_metadata "$NF" "$BATCH_SIZE" "$RUN_DURATION" "${GPU_PAIR//[,]/}" "COMPLETED"
        
        touch "$DONE_FILE" # Mark as done
      fi
    done
    
    GROUP_END_TIME=$(date +%s)
    TOTAL_DURATION=$((GROUP_END_TIME - GROUP_START_TIME))
    
    # Final summary for this GPU pair
    echo -e "${BOLD}[${GPU_PAIR}]${NC} ${GREEN}All runs completed for ${MODEL_NAME}!${NC} Total time: $(printf "%02d:%02d:%02d" $((TOTAL_DURATION/3600)) $(((TOTAL_DURATION%3600)/60)) $((TOTAL_DURATION%60)))"
    
  ) &
  child_pids+=("$!")  # keep track of the subshell pid

done

# Wait for all background subshells to finish (unless we were interrupted)
wait

timestamp=$(date "+%Y-%m-%d %H:%M:%S")
echo -e "\n${GREEN}${BOLD}[${timestamp}] All experiments finished for ${MODEL_NAME}!${NC} Results saved to logs directory."
echo "All experiments finished for ${MODEL_NAME}." > logs/all_completed_${MODEL_SAFE}.log 

# =============================================================================
# 📊 NEXT STEPS: GENERATE PLOTS AND ANALYZE RESULTS
# =============================================================================
echo -e "\n${BLUE}${BOLD}📊 Next Steps:${NC}"
echo -e "${YELLOW}1. Generate Plots:${NC}"
echo -e "   python plot_metrics.py"
echo -e "   ${GREEN}→ Creates training evolution plots (steps-based, with circle markers)${NC}"
echo -e "   ${GREEN}→ Organizes plots by model and batch size: ${MODEL_NAME}_batch_XX.png${NC}"
echo -e "   ${GREEN}→ Creates summary plots comparing all model/batch size combinations${NC}"

echo -e "\n${YELLOW}2. Check Individual Results:${NC}"
echo -e "   ls /ephemeral/oml-exploration-results/saved_models/"
echo -e "   ${GREEN}→ Each experiment has a unique hash-based directory${NC}"
echo -e "   ${GREEN}→ Contains: final_model/, fingerprinting_config.json, eval_epoch_*.jsonl${NC}"

echo -e "\n${YELLOW}3. Timing Analysis:${NC}"
echo -e "   cat logs/timing_summary_${MODEL_SAFE}.jsonl"
echo -e "   ${GREEN}→ JSON lines with detailed timing metadata for each run${NC}"
echo -e "   ${GREEN}→ Includes: duration, time_per_fingerprint, model, batch_size${NC}"
echo -e "   ${GREEN}→ Individual GPU timing logs: logs/timing_*_*.log${NC}"

echo -e "\n${YELLOW}4. Monitor Progress:${NC}"
echo -e "   python check_eval_results.py"
echo -e "   ${GREEN}→ Shows summary of all completed runs and their latest metrics${NC}"

echo -e "\n${YELLOW}5. Run Different Models:${NC}"
echo -e "   ${GREEN}MODEL_NAME=Llama-3.2-8B-Instruct ./launch_parallel_experiments.sh${NC}"
echo -e "   ${GREEN}MODEL_NAME=Llama-3.2-70B-Instruct BATCH_SIZES=\"16 32\" ./launch_parallel_experiments.sh${NC}"

echo -e "\n${BLUE}${BOLD}🎯 Key Features Used:${NC}"
echo -e "• ${GREEN}Model:${NC} $MODEL_NAME ($MODEL_SIZE)"
echo -e "• ${GREEN}Chat Template:${NC} $(if [[ -n "$USE_CHAT_TEMPLATE" ]]; then echo "Enabled (proper user/assistant formatting)"; else echo "Disabled (base model training)"; fi)"
echo -e "• ${GREEN}Fingerprint counts:${NC} ${FINGERPRINT_COUNT_ARRAY[*]}"
echo -e "• ${GREEN}Batch sizes:${NC} ${BATCH_SIZE_ARRAY[*]}"
echo -e "• ${GREEN}Time estimation:${NC} Improved fingerprint-count-aware prediction (s/fp scaling)"
echo -e "• ${GREEN}Timing metadata:${NC} Detailed timing logs saved to logs/timing_summary_*.jsonl"
echo -e "• ${GREEN}Training steps:${NC} Plots will show steps (epoch × batch_size) instead of epochs"
echo -e "• ${GREEN}Markers:${NC} All plots include circle markers for data points"
echo -e "• ${GREEN}Skip logic:${NC} Already completed experiments are automatically skipped"

echo -e "\n${GREEN}${BOLD}✅ Experiment suite completed successfully!${NC}" 
