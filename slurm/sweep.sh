#!/usr/bin/env bash
# Submit a grid of points. Every point is its own job; finished points exit immediately.
#   slurm/sweep.sh chartqa_human "300 900" "1 2 3" "sft opd"
set -euo pipefail
TASK="${1:-chartqa_human}"
BUDGETS="${2:-100 300 900 3000}"
SEEDS="${3:-42}"
METHODS="${4:-sft opd}"
for b in $BUDGETS; do for s in $SEEDS; do for m in $METHODS; do
  t=$([ "$m" = "opd" ] && echo "02:30:00" || echo "01:00:00")
  sbatch -t "$t" -J "${m}_${TASK}_q${b}_s${s}" --export=ALL,TASK="$TASK",METHOD="$m",BUDGET="$b",SEED="$s" slurm/point.sbatch
done; done; done
