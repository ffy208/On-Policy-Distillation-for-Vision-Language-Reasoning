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
git pull -q || true
mkdir -p outputs

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
