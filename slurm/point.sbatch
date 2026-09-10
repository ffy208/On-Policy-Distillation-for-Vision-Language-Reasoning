#!/usr/bin/env bash
# One experiment point on one GPU. Submit from the repository root with environment variables, e.g.
#   sbatch --export=ALL,TASK=chartqa_human,METHOD=opd,BUDGET=300,SEED=1 slurm/point.sbatch
# Idempotent: resubmitting a killed job resumes OPD from its Hub checkpoint and skips finished results.
#SBATCH -J vlmopd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=48G
# No -p: Slurm uses the account's default partition (coc-gpu here; the ice-gpu H100 nodes are not open to it).
# Override the GPU type on the command line, e.g. `sbatch --gres=gpu:l40s:1 slurm/point.sbatch` (L40S: 48 GB, 32 cards).
#SBATCH --gres=gpu:a100:1
#SBATCH -t 02:30:00              # OPD points take ~90 min; keep walltime short so the job backfills
#SBATCH -o logs/%x-%j.out
set -euo pipefail

: "${TASK:?set TASK}"; : "${METHOD:?set METHOD}"; : "${BUDGET:?set BUDGET}"; : "${SEED:=42}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"   # submit from the repository root
ENV_DIR="${ENV_DIR:-$HOME/vlm_opd_env}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/scratch}/hf_cache}"
export HF_TOKEN="$(cat "$HOME/.hf_token")"
export TOKENIZERS_PARALLELISM=false

module load cuda/12.6.1 2>/dev/null || module load cuda 2>/dev/null || true
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"
git pull -q || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m vlm_opd.experiment --task "$TASK" --method "$METHOD" --budget "$BUDGET" --seed "$SEED" ${EXTRA_ARGS:-}
