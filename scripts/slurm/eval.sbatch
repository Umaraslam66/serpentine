#!/bin/bash
#SBATCH --job-name=serpentine-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# Pretrained-POMO TSP-100 evaluation on the fixed test set (GPU). Pass --account at submit.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source scripts/_common.sh
command -v module >/dev/null 2>&1 && module load python/3.11.7 || true

N=${N:-10000}; SEED=${SEED:-1234}
MODEL_PATH=${MODEL_PATH:-$POMO_ROOT/POMO/result/saved_tsp100_model2_longTrain}
EPOCH=${EPOCH:-3100}; TAG=${TAG:-longtrain}
echo "[eval] N=$N seed=$SEED tag=$TAG model=$MODEL_PATH epoch=$EPOCH host=$(hostname)"
srun python -m serpentine.eval \
  --instances "$WORKDIR/data/tsp100_test_seed${SEED}.pt" \
  --model-path "$MODEL_PATH" --epoch "$EPOCH" \
  --aug 8 --batch-size 500 \
  --out "$WORKDIR/results/pomo_eval_${TAG}_n${N}.npz"
