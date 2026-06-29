#!/bin/bash
# ONE-COMMAND REPRODUCE (run on a Leonardo login node, after env_setup.sh + build_lkh.sh).
#   bash baseline/run_baseline.sh            # full N=10000
#   N=128 bash baseline/run_baseline.sh      # quick smoke
# Generates the fixed test set, then submits a Slurm DAG:
#   LKH (CPU, oracle) once  +  eval (GPU) per checkpoint  ->  gap report per checkpoint.
# Both checkpoints are scored against the SAME LKH optimal lengths.
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

CKPT=POMO/NEW_py_ver/TSP/POMO/result
echo "== submitting jobs =="
LKH_ID=$(sbatch --parsable baseline/sbatch_lkh.sh)

# Checkpoint A: longer-trained (repo default, epoch 3100)
E1=$(MODEL_PATH=$CKPT/saved_tsp100_model2_longTrain EPOCH=3100 TAG=longtrain \
      sbatch --parsable baseline/sbatch_eval.sh)
G1=$(TAG=longtrain sbatch --parsable --dependency=afterok:${E1}:${LKH_ID} baseline/sbatch_gap.sh)

# Checkpoint B: standard model (epoch 2000) — paper-reference candidate
E2=$(MODEL_PATH=$CKPT/saved_tsp100_model EPOCH=2000 TAG=standard \
      sbatch --parsable baseline/sbatch_eval.sh)
G2=$(TAG=standard sbatch --parsable --dependency=afterok:${E2}:${LKH_ID} baseline/sbatch_gap.sh)

echo "lkh=$LKH_ID"
echo "longtrain: eval=$E1 gap=$G1"
echo "standard : eval=$E2 gap=$G2"
echo "monitor: squeue -u \$USER"
echo "reports -> results/baseline_report_{longtrain,standard}_n${N}.json"
