#!/bin/bash
# ONE-COMMAND REPRODUCE (run on a Leonardo login node, after env_setup.sh + build_lkh.sh).
#   bash baseline/run_baseline.sh            # full N=10000
#   N=128 bash baseline/run_baseline.sh      # quick smoke
# Generates the fixed test set, then submits a Slurm DAG: eval (GPU) + lkh (CPU),
# with the gap report depending on both (afterok).
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"
N=${N:-10000}
SEED=${SEED:-1234}
export N SEED

module load python/3.11.7
source .venv/bin/activate
mkdir -p data results logs

echo "== generating fixed test set (N=$N seed=$SEED) =="
python baseline/gen_testset.py --n "$N" --seed "$SEED" --out "data/tsp100_test_seed${SEED}"

echo "== submitting jobs =="
EVAL_ID=$(sbatch --parsable baseline/sbatch_eval.sh)
LKH_ID=$(sbatch --parsable baseline/sbatch_lkh.sh)
GAP_ID=$(sbatch --parsable --dependency=afterok:${EVAL_ID}:${LKH_ID} baseline/sbatch_gap.sh)
echo "eval=$EVAL_ID  lkh=$LKH_ID  gap=$GAP_ID (afterok eval,lkh)"
echo "monitor: squeue -u \$USER ; report -> results/baseline_report_n${N}.json"
