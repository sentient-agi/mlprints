#!/usr/bin/env bash
set -euo pipefail

# Simple sweep script for FPEdit.
# Usage:
#   bash scripts/sweep_fpedit.sh
# Optional environment overrides:
#   NUM_FPS="10 128"  # space-separated list

NUM_FPS=${NUM_FPS:-"16 128"}


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "${REPO_ROOT}"

echo "Running FPEdit sweep..."

# Models to sweep. Options:
#   default        -> uses config defaults (Llama-3.2-1B-Instruct)
#   llama31_8b     -> Meta-Llama-3.1-8B-Instruct (layers [4,5,6,7,8])
#   qwen2_5_1p5b   -> Qwen2.5-1.5B-Instruct (layers [4,5,6,7,8])
#   qwen2_5_7b     -> Qwen2.5-7B-Instruct (layers [4,5,6,7,8])
MODELS=${MODELS:-"default qwen2_5_1p5b qwen2_5_7b llama31_8b"}

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
        "algo.models_dict.base.model_id=meta-llama/Meta-Llama-3.1-8B-Instruct"
        "algo.alpha_edit.hparams.model_name=Meta-Llama-3.1-8B-Instruct"
        "algo.alpha_edit.hparams.layers=[4,5,6,7,8]"
      )
      ;;
    qwen2_5_1p5b)
      MODEL_DESC="Qwen2.5-1.5B-Instruct"
      OVERRIDES+=(
        "algo.models_dict.base.model_id=Qwen/Qwen2.5-1.5B-Instruct"
        "algo.alpha_edit.hparams.model_name=Qwen2.5-1.5B-Instruct"
        "algo.alpha_edit.hparams.layers=[4,5,6,7,8]"
      )
      ;;
    qwen2_5_7b)
      MODEL_DESC="Qwen2.5-7B-Instruct"
      OVERRIDES+=(
        "algo.models_dict.base.model_id=Qwen/Qwen2.5-7B-Instruct"
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

for MODEL in ${MODELS}; do
  set_model_overrides "$MODEL"
  for NF in ${NUM_FPS}; do
    echo "=== FPEdit: model=${MODEL_DESC} | num_fingerprints=${NF} ==="
    CUDA_VISIBLE_DEVICES=1 python -m src.oml.fingerprint.FPEdit \
      algo.params.num_fingerprints=${NF} \
      "${OVERRIDES[@]}"
  done
done

echo "FPEdit sweep completed."


