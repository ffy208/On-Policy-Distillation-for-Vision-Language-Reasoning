#!/usr/bin/env bash
# Submit a grid of points. Every point is its own job; finished points exit immediately.
#   slurm/sweep.sh chartqa_human "300 900" "1 2 3" "sft opd"
set -euo pipefail
TASK="${1:-chartqa_human}"
BUDGETS="${2:-100 300 900 3000}"
SEEDS="${3:-42}"
METHODS="${4:-sft opd}"
GPU="${GPU:-l40s}"     # GPU=a100 slurm/sweep.sh ... to use the (smaller) A100 pool instead
# Nodes whose GPU failed the health check in an earlier job (written by point.sbatch) are excluded up front.
EXCLUDE=""; [ -s logs/bad_nodes.txt ] && EXCLUDE="--exclude=$(paste -sd, logs/bad_nodes.txt)"
# "baseline" (zero-shot student + teacher on the test set and OOD sets) is one job per task, budgets/seeds ignored:
#   slurm/sweep.sh geometry3k "" "" baseline
if [[ " $METHODS " == *" baseline "* ]]; then
  sbatch $EXCLUDE -t "00:45:00" --gres="gpu:${GPU}:1" -J "baseline_${TASK}" --export=ALL,TASK="$TASK",METHOD=baseline,BUDGET=0,SEED=42 slurm/point.sbatch
  METHODS="${METHODS//baseline/}"
fi
for b in $BUDGETS; do for s in $SEEDS; do for m in $METHODS; do
  t=$([ "$m" = "opd" ] && echo "02:30:00" || echo "01:00:00")
  sbatch $EXCLUDE -t "$t" --gres="gpu:${GPU}:1" -J "${m}_${TASK}_q${b}_s${s}" --export=ALL,TASK="$TASK",METHOD="$m",BUDGET="$b",SEED="$s" slurm/point.sbatch
done; done; done
