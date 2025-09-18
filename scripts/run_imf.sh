#!/usr/bin/env bash
# set -euo pipefail

# Repo root
BASE_DIR="/gscratch/sewoong/anasery/fingerprinting/oml-exploration"

# W&B logging
export WANDB_PROJECT="fingerprint_baselines_imf"
export WANDB_MODE="online"

# Hyperparameters
SEED=42
NUM_FPS=(16 120)

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
    if [ "$MODEL" == "Meta-Llama/Llama-3.1-8B-Instruct" ] ; then
      BATCH_SIZE=4
      EPOCHS=30
      LR=3.0e-5
      ORIG_FPS_PATH="data/baselines/imf/stego_llama-3.1-8b-instruct_capitalized_sorted.json"
    elif [ "$MODEL" == "Meta-Llama/Llama-3.2-1B-Instruct" ] ; then
      BATCH_SIZE=8
      EPOCHS=25
      LR=5.0e-5
      ORIG_FPS_PATH="data/baselines/imf/stego_llama-3.2-1b-instruct_capitalized_sorted.json"
    elif [ "$MODEL" == "Qwen/Qwen2.5-7B-Instruct" ]; then
      BATCH_SIZE=4
      EPOCHS=30
      LR=3.0e-5
      ORIG_FPS_PATH="data/baselines/imf/stego_qwen-2.5-7b-instruct_capitalized_sorted.json"
    elif [ "$MODEL" == "Qwen/Qwen2.5-1.5B-Instruct" ]; then
      BATCH_SIZE=8
      EPOCHS=25
      LR=5.0e-5
      ORIG_FPS_PATH="data/baselines/imf/stego_qwen-2.5-1.5b-instruct_capitalized_sorted.json"
    else
      BATCH_SIZE=8
      EPOCHS=25
      LR=5.0e-5
      ORIG_FPS_PATH=null
    fi
    # 1) Original (no randomization)
    # Run this only for 128 fp
    # accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes=4
    # python -m src.oml.fingerprint.imf \
    # Remember to uncomment the deepspeed config in the imf.py file
    deepspeed --num_gpus 4  --master_port 29501 --module src.oml.fingerprint.imf \
      algo.params.seed="$SEED" \
      algo.params.num_fingerprints="$NUM_FINGERPRINT" \
      algo.params.models_dict.base.model_id="$MODEL" \
      algo.params.use_original=true \
      algo.params.orig_fingerprints_path="$ORIG_FPS_PATH" \
      training.batch_size="$BATCH_SIZE" \
      training.grad_accumulation=8 \
      training.num_train_epochs="$EPOCHS" \
      training.learning_rate="$LR" training.output_dir="experiments/models/imf_better"
    # # 2) Randomized decryptions (tokens), maxlen in {1,5}, instructions {off,on}
    # for MAXLEN in 1 5; do
    #   for INSTR in false true; do

    #     accelerate launch --config_file configs/deepspeed_zero2.yaml --num_processes=4 -m src.oml.fingerprint.instructional_fp \
    #       algo.params.seed="$SEED" \
    #       algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #       algo.params.models_dict.base.model_id="$MODEL" \
    #       algo.params.use_original=false \
    #       algo.params.randomize_decryptions=true \
    #       algo.params.use_tokens_instead_of_words_for_randomization=true \
    #       algo.params.max_decryption_length="$MAXLEN" \
    #       algo.params.randomize_instructions="$INSTR" \
    #       training.batch_size="$BATCH_SIZE" \
    #       training.grad_accumulation=8

    #     # CUDA_VISIBLE_DEVICES=1 python3 -m src.oml.fingerprint.instructional_fp \
    #     #   algo.params.seed="$SEED" \
    #     #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #     #   algo.params.models_dict.base.model_id="$MODEL" \
    #     #   algo.params.use_original=false \
    #     #   algo.params.randomize_decryptions=true \
    #     #   algo.params.use_tokens_instead_of_words_for_randomization=true \
    #     #   algo.params.max_decryption_length="$MAXLEN" \
    #     #   algo.params.randomize_instructions=true &
        
    #     # wait

    #   done
    # done
  done
done

echo "All runs completed."