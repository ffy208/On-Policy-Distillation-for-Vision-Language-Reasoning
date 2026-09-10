#!/usr/bin/env bash
# 15-minute smoke test on one GPU: 5 OPD steps on 100 questions with a throwaway seed, then evaluation.
# Short walltime and small CPU/memory so it backfills quickly. Submit from the repository root:
#   sbatch slurm/smoke.sbatch
#SBATCH -J vlmopd-smoke
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frank.yang@gatech.edu
export TASK="${TASK:-chartqa_human}" METHOD="${METHOD:-opd}" BUDGET="${BUDGET:-100}" SEED="${SEED:-99}"
export EXTRA_ARGS="${EXTRA_ARGS:---opd-steps 5 --sft-steps 5 --skip-ood}"
exec bash "${SLURM_SUBMIT_DIR:-$PWD}/slurm/point.sbatch"
