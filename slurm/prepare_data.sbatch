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
    --export=ALL,SOURCE="$SOURCE" "$REPO_DIR/slurm/prepare_data.sbatch"
  exit 0
fi
rm -f logs/gpu_check-$SLURM_JOB_ID.err

case "$SOURCE" in
  geometry3k)
    DATA_REPO="${DATA_REPO:-$HUB_USER/vlm_opd_$SOURCE}"
    SFT_REPO="${SFT_REPO:-$HUB_USER/vlm_opd_sft_$SOURCE}"
    python -m vlm_opd.prepare --source "$SOURCE" --n-train "${N_TRAIN:-2101}" --n-test "${N_TEST:-500}" --repo-id "$DATA_REPO"
    # 1536 new tokens: geometry solutions are long, and a truncated solution has no Answer: line and is dropped
    python -m vlm_opd.generate_teacher --model "$TEACHER" --prompt-style geometry --data-repo "$DATA_REPO" \
      --max-new-tokens 1536 --max-model-len 6144 --out-repo "$SFT_REPO" --stats "outputs/teacher_generation_$SOURCE.json"
    ;;
  charxiv|chartqapro)
    DATA_REPO="${DATA_REPO:-$HUB_USER/vlm_opd_ood_$SOURCE}"
    python -m vlm_opd.prepare --source "$SOURCE" --n-train 0 --n-test "${N_TEST:-500}" --repo-id "$DATA_REPO"
    ;;
  *) echo "unknown SOURCE=$SOURCE" >&2; exit 2 ;;
esac
