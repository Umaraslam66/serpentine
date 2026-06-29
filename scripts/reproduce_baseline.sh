#!/bin/bash
# ONE-COMMAND REPRODUCE of the Gate-0 baseline seatbelt (POMO TSP-100 gap).
# Prereqs: scripts/env_setup.sh + scripts/build_lkh.sh ; export SLURM_ACCOUNT=<alloc>.
#   bash scripts/reproduce_baseline.sh          # full N=10000
#   N=128 bash scripts/reproduce_baseline.sh    # quick smoke
# Generates the fixed test set, then submits a Slurm DAG: LKH (CPU oracle) once + eval
# (GPU) per checkpoint -> gap report per checkpoint, all vs the SAME LKH lengths.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source scripts/_common.sh
ACCT="${SLURM_ACCOUNT:?export SLURM_ACCOUNT=<your Slurm allocation>}"
N=${N:-10000}; SEED=${SEED:-1234}; export N SEED
command -v module >/dev/null 2>&1 && module load python/3.11.7 || true

echo "== fixed test set N=$N seed=$SEED =="
python -m serpentine.data --n "$N" --seed "$SEED" --out "$WORKDIR/data/tsp100_test_seed${SEED}"

CKPT="$POMO_ROOT/POMO/result"
SB=(sbatch --account="$ACCT" --parsable)
LKH_ID=$("${SB[@]}" scripts/slurm/lkh.sbatch)
E1=$(MODEL_PATH="$CKPT/saved_tsp100_model2_longTrain" EPOCH=3100 TAG=longtrain "${SB[@]}" scripts/slurm/eval.sbatch)
G1=$(TAG=longtrain "${SB[@]}" --dependency=afterok:${E1}:${LKH_ID} scripts/slurm/gap.sbatch)
E2=$(MODEL_PATH="$CKPT/saved_tsp100_model" EPOCH=2000 TAG=standard "${SB[@]}" scripts/slurm/eval.sbatch)
G2=$(TAG=standard "${SB[@]}" --dependency=afterok:${E2}:${LKH_ID} scripts/slurm/gap.sbatch)
echo "lkh=$LKH_ID  longtrain(eval=$E1 gap=$G1)  standard(eval=$E2 gap=$G2)"
echo "reports -> $WORKDIR/results/baseline_report_{longtrain,standard}_n${N}.json"
