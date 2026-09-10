#!/usr/bin/env bash
# Build the Hub datasets for a task or an OOD test set (unified schema), then, for a training task,
# generate and push the verified teacher solutions used by SFT. Submit from the repository root:
#   sbatch --export=ALL,SOURCE=geometry3k slurm/prepare_data.sbatch      # task: train + test + teacher SFT data
#   sbatch --export=ALL,SOURCE=charxiv    slurm/prepare_data.sbatch      # OOD test set only
#   sbatch --export=ALL,SOURCE=chartqapro slurm/prepare_data.sbatch
# Repo names follow configs/tasks.yaml: ffyang/vlm_opd_<source> and ffyang/vlm_opd_sft_<source> for
# tasks, ffyang/vlm_opd_ood_<source> for OOD sets. Override with DATA_REPO / SFT_REPO.
#SBATCH -J vlmopd-data
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frank.yang@gatech.edu
set -euo pipefail

: "${SOURCE:?set SOURCE to geometry3k, charxiv or chartqapro}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_DIR="${ENV_DIR:-$HOME/vlm_opd_env}"
HUB_USER="${HUB_USER:-ffyang}"
TEACHER="${TEACHER:-Qwen/Qwen3-VL-8B-Instruct}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/scratch}/hf_cache}"
export HF_TOKEN="$(cat "$HOME/.hf_token")"
export TOKENIZERS_PARALLELISM=false

module load cuda/12.6.1 2>/dev/null || module load cuda 2>/dev/null || true
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"
git pull --ff-only -q || echo "WARNING: git pull skipped (local branch diverged from origin); running the checked-out code"
mkdir -p outputs logs
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

case "$SOURCE" in
  geometry3k)
    DATA_REPO="${DATA_REPO:-$HUB_USER/vlm_opd_$SOURCE}"
    SFT_REPO="${SFT_REPO:-$HUB_USER/vlm_opd_sft_$SOURCE}"
    python -m vlm_opd.prepare --source "$SOURCE" --n-train "${N_TRAIN:-2101}" --n-test "${N_TEST:-500}" --repo-id "$DATA_REPO"
    python -m vlm_opd.generate_teacher --model "$TEACHER" --prompt-style geometry --data-repo "$DATA_REPO" \
      --out-repo "$SFT_REPO" --stats "outputs/teacher_generation_$SOURCE.json"
    ;;
  charxiv|chartqapro)
    DATA_REPO="${DATA_REPO:-$HUB_USER/vlm_opd_ood_$SOURCE}"
    python -m vlm_opd.prepare --source "$SOURCE" --n-train 0 --n-test "${N_TEST:-500}" --repo-id "$DATA_REPO"
    ;;
  *) echo "unknown SOURCE=$SOURCE" >&2; exit 2 ;;
esac
