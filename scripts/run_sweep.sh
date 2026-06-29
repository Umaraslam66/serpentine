#!/bin/bash
# >=3-SEED SWEEP. Run only after calibration shows from-scratch attention <=2-3%
# single-traj greedy. export SLURM_ACCOUNT=<alloc>.
# Submits {attention, mamba/hilbert, mamba/random, mamba/sort} x seeds, identical 60k-step
# budget, single-trajectory greedy as the judged metric.
#   bash scripts/run_sweep.sh                 # seeds "1 2 3"
#   SEEDS="1 2 3 4 5" bash scripts/run_sweep.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source scripts/_common.sh
ACCT="${SLURM_ACCOUNT:?export SLURM_ACCOUNT=<your Slurm allocation>}"
STEPS=${STEPS:-60000}
SEEDS=${SEEDS:-"1 2 3"}

CFG=(
  "attention hilbert "
  "mamba hilbert --use-kernel"
  "mamba random --use-kernel"
  "mamba sort --use-kernel"
)
n=0
for seed in $SEEDS; do
  for c in "${CFG[@]}"; do
    set -- $c
    enc=$1; order=$2; kflag=${3:-}
    jid=$(STEPS=$STEPS SEED=$seed ENC=$enc ORDER=$order KFLAG="$kflag" \
          sbatch --account="$ACCT" --parsable scripts/slurm/train.sbatch)
    echo "seed=$seed ${enc}/${order} -> job $jid"
    n=$((n + 1))
  done
done
echo "submitted $n jobs ($(echo $SEEDS | wc -w) seeds x ${#CFG[@]} configs), STEPS=$STEPS"
echo "aggregate when done: python scripts/aggregate_sweep.py"
