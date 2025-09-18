#!/usr/bin/env bash
set -euo pipefail

# Sweep script for editMF with Hydra overrides.
# Usage:
#   bash scripts/sweep_editmf.sh
# Optional environment overrides:
#   NUM_FPS="10 128"          # space-separated list
#   NEIGHBOURS="1 4"         # neighbour_count values
#   NUM_PARAS="1 4 8"        # num_paraphrases_per_fp values

NUM_FPS=${NUM_FPS:-"16 32"}
NEIGHBOURS=${NEIGHBOURS:-"0"}
NUM_PARAS=${NUM_PARAS:-"4"}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-"false"}
echo "Running EditMF sweep..."

# Models to sweep. Options:
#   default        -> uses config defaults (Llama-3.2-1B-Instruct)
#   llama31_8b     -> Meta-Llama-3.1-8B-Instruct (layers [4,5,6,7,8])
#   qwen2_5_1p5b   -> Qwen2.5-1.5B-Instruct (layers [4,5,6,7,8])
#   qwen2_5_7b     -> Qwen2.5-7B-Instruct (layers [4,5,6,7,8])
MODELS=${MODELS:-"qwen2_5_1p5b default"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "${REPO_ROOT}"
set_model_overrides() {
  local label="$1"
  MODEL_DESC="$label"
  OVERRIDES=()
  case "$label" in
    default)
      MODEL_DESC="Llama-3.2-1B-Instruct (default)"
      ;;
    llama31_8b)
      MODEL_DESC="Meta-Llama-3.1-8B-Instruct"
      OVERRIDES+=(
        "algo.params.models_dict.base.model_id=meta-llama/Meta-Llama-3.1-8B-Instruct"
        "algo.alpha_edit.hparams.model_name=Meta-Llama-3.1-8B-Instruct"
        "algo.alpha_edit.hparams.layers=[4,5,6,7,8]"
      )
      ;;
    qwen2_5_1p5b)
      MODEL_DESC="Qwen2.5-1.5B-Instruct"
      OVERRIDES+=(
        "algo.params.models_dict.base.model_id=Qwen/Qwen2.5-1.5B-Instruct"
        "algo.alpha_edit.hparams.model_name=Qwen2.5-1.5B-Instruct"
        "algo.alpha_edit.hparams.layers=[4,5,6,7,8]"
        "algo.alpha_edit.hparams.v_num_grad_steps=30"
        "algo.alpha_edit.hparams.v_lr=5e-1"
        "algo.alpha_edit.hparams.v_weight_decay=1e-3"
        "algo.alpha_edit.hparams.clamp_norm_factor=4"
        "algo.alpha_edit.hparams.v_loss_layer=27"
        "algo.alpha_edit.hparams.L2=1"
      )
      ;;
    qwen2_5_7b)
      MODEL_DESC="Qwen2.5-7B-Instruct"
      OVERRIDES+=(
        "algo.params.models_dict.base.model_id=Qwen/Qwen2.5-7B-Instruct"
        "algo.alpha_edit.hparams.model_name=Qwen2.5-7B-Instruct"
        "algo.alpha_edit.hparams.layers=[4,5,6,7,8]"
      )
      ;;
    *)
      echo "Unknown model label: $label" >&2
      exit 1
      ;;
  esac
}
echo "Running editMF sweep..."
for MODEL in ${MODELS}; do
  set_model_overrides "$MODEL"
for NF in ${NUM_FPS}; do
  for NC in ${NEIGHBOURS}; do
    for NP in ${NUM_PARAS}; do
      for CT in ${USE_CHAT_TEMPLATE}; do    
      echo "=== editMF: model=${MODEL_DESC} | num_fingerprints=${NF}, neighbour_count=${NC}, num_paraphrases_per_fp=${NP} ==="
       CUDA_VISIBLE_DEVICES=0 python -m src.oml.fingerprint.editMF \
        algo.params.num_fingerprints=${NF} \
        algo.params.neighbour_count=${NC} \
        algo.params.num_paraphrases_per_fp=${NP} \
        algo.params.use_chat_template=${CT} \
        "${OVERRIDES[@]}"
    done
  done
done
done
done

echo "editMF sweep completed."


