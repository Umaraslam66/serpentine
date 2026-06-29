#!/bin/bash
#SBATCH --job-name=serpentine-lkh
#SBATCH --partition=lrd_all_serial
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# LKH-3 optimal lengths for the fixed test set (CPU oracle). Pass --account at submit.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
command -v module >/dev/null 2>&1 && module load python/3.11.7 || true
# shellcheck disable=SC1091
source scripts/_common.sh
export LKH_BIN="$WORKDIR/tools/lkh"

N=${N:-10000}; SEED=${SEED:-1234}
echo "[lkh] N=$N seed=$SEED procs=${SLURM_CPUS_PER_TASK:-8} host=$(hostname)"
srun python -m serpentine.oracle \
  --instances "$WORKDIR/data/tsp100_test_seed${SEED}.npy" \
  --out "$WORKDIR/results/lkh_opt_n${N}.npy" \
  --runs 1 --procs "${SLURM_CPUS_PER_TASK:-8}"
