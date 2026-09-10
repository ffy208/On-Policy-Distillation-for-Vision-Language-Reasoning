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
# Override the GPU type on the command line, e.g. `sbatch --gres=gpu:a100:1 slurm/point.sbatch` (A100: 40 GB, only 8 cards; L40S: 48 GB, 32 cards).
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 02:30:00              # OPD points take ~90 min; keep walltime short so the job backfills
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frank.yang@gatech.edu
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
git pull --ff-only -q || echo "WARNING: git pull skipped (local branch diverged from origin); running the checked-out code"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# GPU health check: some cards raise uncorrectable ECC errors at first touch. Users cannot edit a job's
# ExcNodeList, so instead of requeueing we record the node in logs/bad_nodes.txt and submit a fresh copy of
# this job with --exclude covering every node recorded so far (the point resumes from its Hub checkpoint).
if ! python -c "import torch; torch.zeros(1).cuda(); torch.cuda.synchronize()" 2>logs/gpu_check-$SLURM_JOB_ID.err; then
  echo "GPU check failed on $SLURMD_NODENAME; recording it and resubmitting elsewhere" >&2
  tail -n 2 logs/gpu_check-$SLURM_JOB_ID.err >&2
  grep -qx "$SLURMD_NODENAME" logs/bad_nodes.txt 2>/dev/null || echo "$SLURMD_NODENAME" >> logs/bad_nodes.txt
  EXC="$(paste -sd, logs/bad_nodes.txt)"
  JOB="$(scontrol show job "$SLURM_JOB_ID")"
  TLIM="$(sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p' <<<"$JOB")"
  GRES="$(sed -n 's/.*TresPerNode=gres\/\([^ ]*\).*/\1/p' <<<"$JOB" | sed 's/=/:/')"
  sbatch --exclude="$EXC" -t "${TLIM:-02:30:00}" --gres="${GRES:-gpu:l40s:1}" -J "$SLURM_JOB_NAME" \
    --export=ALL,TASK="$TASK",METHOD="$METHOD",BUDGET="$BUDGET",SEED="$SEED",EXTRA_ARGS="${EXTRA_ARGS:-}" \
    "$REPO_DIR/slurm/point.sbatch"
  exit 0
fi
rm -f logs/gpu_check-$SLURM_JOB_ID.err
python -m vlm_opd.experiment --task "$TASK" --method "$METHOD" --budget "$BUDGET" --seed "$SEED" ${EXTRA_ARGS:-}
