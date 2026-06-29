#!/bin/bash
# >=3-SEED SWEEP — PREPARED, NOT AUTO-RUN. Launch only after the reviewer approves
# (i.e. only if calibration showed from-scratch attention reaching <=2-3% single-traj greedy).
#
# Submits {attention, mamba/hilbert, mamba/random, mamba/sort} x seeds, identical 60k-step
# budget, single-trajectory greedy as the judged metric. Reuses calibration's seed-1
# attention/hilbert + mamba/hilbert checkpoints (those jobs resume-as-done and just re-eval).
#   bash candidate/run_sweep.sh            # seeds "1 2 3"
#   SEEDS="1 2 3 4 5" bash candidate/run_sweep.sh
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"
STEPS=${STEPS:-60000}
SEEDS=${SEEDS:-"1 2 3"}

# "encoder order kernel-flag"
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
          sbatch --parsable candidate/sbatch_calib.sh)
    echo "seed=$seed ${enc}/${order} -> job $jid"
    n=$((n + 1))
  done
done
echo "submitted $n jobs ($(echo $SEEDS | wc -w) seeds x ${#CFG[@]} configs), STEPS=$STEPS"
echo "aggregate when done: python candidate/aggregate_sweep.py"
