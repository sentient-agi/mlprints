#!/usr/bin/env bash
# set -euo pipefail

# Repo root
BASE_DIR="/gscratch/sewoong/anasery/fingerprinting/oml-exploration"

# W&B logging
export WANDB_PROJECT="fingerprint_baselines_ch"
export WANDB_MODE="online"

# Hyperparameters
SEED=42
NUM_FPS=(128 16)  # adjust as desired, e.g., (16 32 64)

# Models to run (base model). key_gen model can be customized below.
MODELS=(
  "Meta-Llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)

# Key generator model (used to generate keys in chain-hash); defaults to the config value if unset.
KEY_GEN_MODEL="meta-llama/Llama-3.1-8B-Instruct"

cd "$BASE_DIR"

for NUM_FINGERPRINT in "${NUM_FPS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    #  training.grad_accumulation = int((algo.num_fingerprints // int(training.batch_size)) / 4)

    # For 16 fingerprints, set grad accumulation to 2
    # For 128 fingerprints, set grad accumulation to 4
    if [ "$NUM_FINGERPRINT" -eq 16 ]; then
      grad_accumulation=2
    elif [ "$NUM_FINGERPRINT" -eq 128 ]; then
      grad_accumulation=4
    fi

    # 1) Baseline (no random questions, default temp/max key length)
    # CUDA_VISIBLE_DEVICES=0 python3 -m src.oml.fingerprint.chain_hash \
    #   seed="$SEED" \
    #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #   algo.params.models_dict.base.model_id="$MODEL" \
    #   algo.params.models_dict.key_gen.model_id="$KEY_GEN_MODEL" \
    #   algo.params.use_random_questions=false \
    #   algo.params.generation_temp=0.7 \
    #   algo.params.max_key_length=32 \
    #   training.grad_accumulation="$grad_accumulation"

    # # 2) Variants over generation temperature and whether to use random questions

    # # 3) [Optional] Anchor loss variant (enable if you want to try the augmentation)
    # CUDA_VISIBLE_DEVICES=0 python3 -m src.oml.fingerprint.chain_hash \
    #   seed="$SEED" \
    #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #   algo.params.models_dict.base.model_id="$MODEL" \
    #   algo.params.models_dict.key_gen.model_id="$KEY_GEN_MODEL" \
    #   algo.params.generation_temp=0.7 \
    #   algo.params.max_key_length=32 \
    #   training.anchor_loss.use_anchor_loss=true \
    #   training.anchor_loss.anchor_texts_path="data/baselines/chain-hash/anchor_texts.json" \
    #   training.anchor_loss.teacher_model_id="$MODEL" \
    #   training.grad_accumulation="$grad_accumulation"

    # Need to train for longer if using both benign data and random padding
    # accelerate launch --config_file configs/deepspeed_zero2.yaml -m src.oml.fingerprint.chain_hash \
    python -m src.oml.fingerprint.chain_hash \
      seed="$SEED" \
      algo.params.num_fingerprints="$NUM_FINGERPRINT" \
      algo.params.models_dict.base.model_id="$MODEL" \
      algo.params.models_dict.key_gen.model_id="$KEY_GEN_MODEL" \
      algo.params.generation_temp=0.7 \
      algo.params.max_key_length=32 \
      training.anchor_loss.anchor_texts_path="data/baselines/chain-hash/anchor_texts.json" \
      training.anchor_loss.teacher_model_id="$MODEL" \
      training.augmentation.use_random_padding=true \
      training.augmentation.use_meta_prompts=false \
      training.augmentation.append_random_aug_to_answer=true \
      training.grad_accumulation="$grad_accumulation" \
      training.augmentation.use_benign_data=true \
      training.num_train_epochs=20 \
      training.learning_rate=3e-5

    # accelerate launch --config_file configs/deepspeed_zero2.yaml -m src.oml.fingerprint.chain_hash \
    #   seed="$SEED" \
    #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #   algo.params.models_dict.base.model_id="$MODEL" \
    #   algo.params.models_dict.key_gen.model_id="$KEY_GEN_MODEL" \
    #   algo.params.generation_temp=0.7 \
    #   algo.params.max_key_length=32 \
    #   training.anchor_loss.anchor_texts_path="data/baselines/chain-hash/anchor_texts.json" \
    #   training.anchor_loss.teacher_model_id="$MODEL" \
    #   training.augmentation.use_random_padding=true \
    #   training.augmentation.use_meta_prompts=true \
    #   training.grad_accumulation="$grad_accumulation" \
    #   training.augmentation.use_benign_data=true \
    #   training.num_train_epochs=25 \
    #   training.learning_rate=3.5e-5

    # CUDA_VISIBLE_DEVICES=0 python3 -m src.oml.fingerprint.chain_hash \
    #   seed="$SEED" \
    #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #   algo.params.models_dict.base.model_id="$MODEL" \
    #   algo.params.models_dict.key_gen.model_id="$KEY_GEN_MODEL" \
    #   algo.params.generation_temp=0.7 \
    #   algo.params.max_key_length=32 \
    #   training.anchor_loss.anchor_texts_path="data/baselines/chain-hash/anchor_texts.json" \
    #   training.anchor_loss.teacher_model_id="$MODEL" \
    #   training.augmentation.use_random_padding=false \
    #   training.augmentation.use_meta_prompts=false \
    #   training.grad_accumulation="$grad_accumulation" \
    #   training.augmentation.use_benign_data=true


    # CUDA_VISIBLE_DEVICES=0 python3 -m src.oml.fingerprint.chain_hash \
    #   seed="$SEED" \
    #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #   algo.params.models_dict.base.model_id="$MODEL" \
    #   algo.params.models_dict.key_gen.model_id="$KEY_GEN_MODEL" \
    #   algo.params.generation_temp=0.7 \
    #   algo.params.max_key_length=32 \
    #   training.anchor_loss.anchor_texts_path="data/baselines/chain-hash/anchor_texts.json" \
    #   training.anchor_loss.teacher_model_id="$MODEL" \
    #   training.augmentation.use_random_padding=true \
    #   training.augmentation.use_meta_prompts=true \
    #   training.grad_accumulation="$grad_accumulation"

    # CUDA_VISIBLE_DEVICES=0 python3 -m src.oml.fingerprint.chain_hash \
    #   seed="$SEED" \
    #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #   algo.params.models_dict.base.model_id="$MODEL" \
    #   algo.params.models_dict.key_gen.model_id="$KEY_GEN_MODEL" \
    #   algo.params.generation_temp=0.7 \
    #   algo.params.max_key_length=32 \
    #   training.anchor_loss.anchor_texts_path="data/baselines/chain-hash/anchor_texts.json" \
    #   training.anchor_loss.teacher_model_id="$MODEL" \
    #   training.augmentation.use_random_padding=false \
    #   training.augmentation.append_random_aug_to_answer=false \
    #   training.augmentation.use_meta_prompts=false \
    #   training.anchor_loss.use_anchor_loss=true \
    #   training.grad_accumulation="$grad_accumulation" \
    #   training.learning_rate=1.5e-5


    # CUDA_VISIBLE_DEVICES=0 python3 -m src.oml.fingerprint.chain_hash \
    #   seed="$SEED" \
    #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #   algo.params.models_dict.base.model_id="$MODEL" \
    #   algo.params.models_dict.key_gen.model_id="$KEY_GEN_MODEL" \
    #   algo.params.generation_temp=0.7 \
    #   algo.params.max_key_length=32 \
    #   training.anchor_loss.anchor_texts_path="data/baselines/chain-hash/anchor_texts.json" \
    #   training.anchor_loss.teacher_model_id="$MODEL" \
    #   training.augmentation.use_random_padding=true \
    #   training.anchor_loss.use_anchor_loss=true \
    #   training.grad_accumulation="$grad_accumulation" \
    #   training.learning_rate=1e-5

    # CUDA_VISIBLE_DEVICES=0 python3 -m src.oml.fingerprint.chain_hash \
    #   seed="$SEED" \
    #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #   algo.params.models_dict.base.model_id="$MODEL" \
    #   algo.params.models_dict.key_gen.model_id="$KEY_GEN_MODEL" \
    #   algo.params.generation_temp=0.7 \
    #   algo.params.max_key_length=32 \
    #   training.anchor_loss.anchor_texts_path="data/baselines/chain-hash/anchor_texts.json" \
    #   training.anchor_loss.teacher_model_id="$MODEL" \
    #   training.augmentation.use_random_padding=true \
    #   training.anchor_loss.use_anchor_loss=true \
    #   training.grad_accumulation="$grad_accumulation" \
    #   training.learning_rate=1e-5 \
    #   training.anchor_loss.lambda_anchor=0.3

    # CUDA_VISIBLE_DEVICES=0 python3 -m src.oml.fingerprint.chain_hash \
    #   seed="$SEED" \
    #   algo.params.num_fingerprints="$NUM_FINGERPRINT" \
    #   algo.params.models_dict.base.model_id="$MODEL" \
    #   algo.params.models_dict.key_gen.model_id="$KEY_GEN_MODEL" \
    #   algo.params.generation_temp=0.7 \
    #   algo.params.max_key_length=32 \
    #   training.anchor_loss.anchor_texts_path="data/baselines/chain-hash/anchor_texts.json" \
    #   training.anchor_loss.teacher_model_id="$MODEL" \
    #   training.augmentation.use_random_padding=true \
    #   training.anchor_loss.use_anchor_loss=true \
    #   training.grad_accumulation="$grad_accumulation" \
    #   training.learning_rate=1e-5 \
    #   training.anchor_loss.lambda_anchor=0.1      


  done
done

echo "All runs completed."