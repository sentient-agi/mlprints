#!/usr/bin/env bash
# set -euo pipefail

# Repo root
BASE_DIR="/gscratch/sewoong/anasery/fingerprinting/oml-exploration"

# W&B logging
export WANDB_PROJECT="fingerprint_baselines_if"
export WANDB_MODE="online"

# Hyperparameters
SEED=42
NUM_FPS=(128)

# Models to run
MODELS=(
  "Meta-Llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  # "Meta-Llama/Llama-3.2-1B-Instruct"
  # "Qwen/Qwen2.5-1.5B-Instruct"
)

cd "$BASE_DIR"

for NUM_FINGERPRINT in "${NUM_FPS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    # for 8B and 7B models, set batch_size to 3 else 12
    if [ "$MODEL" == "Meta-Llama/Llama-3.1-8B-Instruct" ] || [ "$MODEL" == "Qwen/Qwen2.5-7B-Instruct" ]; then
      BATCH_SIZE=3
    else
      BATCH_SIZE=8
    fi
    # 1) Original (no randomization)
    # Run this only for 128 fp
    # accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes=4 -m src.oml.fingerprint.instructional_fp \
    # python -m src.oml.fingerprint.instructional_fp \
    deepspeed --num_gpus 4  --master_port 29501 --module src.oml.fingerprint.instructional_fp \
      algo.params.seed="$SEED" \
      algo.params.num_fingerprints="$NUM_FINGERPRINT" \
      algo.params.models_dict.base.model_id="$MODEL" \
      algo.params.use_original=true \
      algo.params.randomize_decryptions=false \
      algo.params.randomize_instructions=false \
      algo.params.use_tokens_instead_of_words_for_randomization=false \
      algo.params.max_decryption_length=1 \
      training.batch_size="$BATCH_SIZE" \
      training.grad_accumulation=4 \
      training.num_train_epochs=20
    # 2) Randomized decryptions (tokens), maxlen in {1,5}, instructions {off,on}
    for MAXLEN in 1 5; do
      for INSTR in false true; do

        # accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes=4 -m src.oml.fingerprint.instructional_fp \
        # python -m src.oml.fingerprint.instructional_fp \
        deepspeed --num_gpus 4  --master_port 29501 --module src.oml.fingerprint.instructional_fp \
          algo.params.seed="$SEED" \
          algo.params.num_fingerprints="$NUM_FINGERPRINT" \
          algo.params.models_dict.base.model_id="$MODEL" \
          algo.params.use_original=false \
          algo.params.randomize_decryptions=true \
          algo.params.use_tokens_instead_of_words_for_randomization=true \
          algo.params.max_decryption_length="$MAXLEN" \
          algo.params.randomize_instructions="$INSTR" \
          training.batch_size="$BATCH_SIZE" \
          training.grad_accumulation=4 \
          training.num_train_epochs=20


        # CUDA_VISIBLE_DEVICES=1 python3 -m src.oml.fingerprint.instructional_fp \
        #   algo.params.seed="$SEED" \
        #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
        #   algo.params.models_dict.base.model_id="$MODEL" \
        #   algo.params.use_original=false \
        #   algo.params.randomize_decryptions=true \
        #   algo.params.use_tokens_instead_of_words_for_randomization=true \
        #   algo.params.max_decryption_length="$MAXLEN" \
        #   algo.params.randomize_instructions=true &
        
        # wait

      done
    done
  done
done

echo "All runs completed."
