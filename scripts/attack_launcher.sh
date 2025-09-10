#!/bin/bash
#SBATCH --job-name=utility_under_attack
#SBATCH --array=1-6
#SBATCH --time=2:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gpus-per-task=1
#SBATCH --chdir=/gscratch/sewoong/anasery/fingerprinting/oml-copy/oml-exploration
#SBATCH --partition=gpu-a40
#SBATCH --account=sewoong
#SBATCH --output=/gscratch/sewoong/anasery/fingerprinting/oml-copy/oml-exploration/slurm_logs/%x_%A_%a.out
#SBATCH --error=/gscratch/sewoong/anasery/fingerprinting/oml-copy/oml-exploration/slurm_logs/%x_%A_%a.err

# set -euo pipefail

REPO_DIR="/gscratch/sewoong/anasery/fingerprinting/oml-copy/oml-exploration"
ENV_ACTIVATE="/gscratch/sewoong/anasery/fingerprinting/fingerprinting_env/bin/activate"
OVERLAY="/gscratch/sewoong/anasery/overlays/instr_diff_overlay.img"
CONTAINER="/gscratch/sewoong/anasery/overlays/cuda121-container.sif"

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

# Run inside container, activate venv, then execute the script
apptainer exec --nv --overlay "${OVERLAY}:ro" "${CONTAINER}" bash -lc "source '${ENV_ACTIVATE}' && cd '${REPO_DIR}' && python -m scripts.measure_utility_under_attack"