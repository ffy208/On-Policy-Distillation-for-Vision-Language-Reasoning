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
#SBATCH --requeue
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
# GPU health check: some cards raise uncorrectable ECC errors at first touch. If so, exclude this node
# for this job and go back into the queue instead of failing (the point resumes from its Hub checkpoint).
if ! python -c "import torch; torch.zeros(1).cuda(); torch.cuda.synchronize()" 2>logs/gpu_check-$SLURM_JOB_ID.err; then
  echo "GPU check failed on $SLURMD_NODENAME; excluding it and requeueing" >&2
  cat logs/gpu_check-$SLURM_JOB_ID.err >&2
  PREV="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*ExcNodeList=\([^ ]*\).*/\1/p')"
  EXC="$SLURMD_NODENAME"; [ -n "$PREV" ] && [ "$PREV" != "(null)" ] && EXC="$PREV,$SLURMD_NODENAME"
  scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="$EXC" || true
  scontrol requeue "$SLURM_JOB_ID" || true
  sleep 5; scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="$EXC" || true   # retry once the job is pending
  exit 0
fi
rm -f logs/gpu_check-$SLURM_JOB_ID.err
python -m vlm_opd.experiment --task "$TASK" --method "$METHOD" --budget "$BUDGET" --seed "$SEED" ${EXTRA_ARGS:-}
