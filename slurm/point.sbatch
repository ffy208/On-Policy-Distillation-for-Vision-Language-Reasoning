#!/usr/bin/env bash
# One experiment point on one A100. Submit with environment variables, e.g.
#   sbatch --export=ALL,TASK=chartqa_human,METHOD=opd,BUDGET=300,SEED=1 slurm/point.sbatch
# Idempotent: resubmitting a killed job resumes OPD from its Hub checkpoint and skips finished results.
#SBATCH -J vlmopd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1        # on PACE ICE check `sinfo -o "%P %G"`; H100 nodes also work (80 GB fits the 32B teacher)
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
##SBATCH -A <your-account>      # uncomment and fill if your allocation requires it
##SBATCH -p <gpu-partition>     # e.g. the A100 partition name from `pace-check-queue`
set -euo pipefail

: "${TASK:?set TASK}"; : "${METHOD:?set METHOD}"; : "${BUDGET:?set BUDGET}"; : "${SEED:=42}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repository root
ENV_DIR="${ENV_DIR:-$HOME/vlm_opd_env}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/scratch}/hf_cache}"
export HF_TOKEN="$(cat "$HOME/.hf_token")"
export TOKENIZERS_PARALLELISM=false

module load cuda/12.8 2>/dev/null || module load cuda 2>/dev/null || true
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m vlm_opd.experiment --task "$TASK" --method "$METHOD" --budget "$BUDGET" --seed "$SEED" ${EXTRA_ARGS:-}
