#!/bin/bash
#SBATCH --job-name=mamba-base-lkh
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# LKH-3 optimal lengths for the fixed test set (CPU, budget-free serial partition).
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"
mkdir -p logs results
module load python/3.11.7
source .venv/bin/activate
export LKH_BIN="$ROOT/tools/lkh"

N=${N:-10000}
SEED=${SEED:-1234}
echo "[lkh] N=$N seed=$SEED procs=${SLURM_CPUS_PER_TASK:-8} host=$(hostname)"
srun python baseline/lkh_solve.py \
  --instances data/tsp100_test_seed${SEED}.npy \
  --out results/lkh_opt_n${N}.npy \
  --runs 1 --procs ${SLURM_CPUS_PER_TASK:-8}
