#!/bin/bash
#SBATCH --job-name=mamba-base-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# POMO TSP-100 evaluation on the fixed test set (GPU). Single process; debug QOS.
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"
mkdir -p logs results
module load python/3.11.7
source .venv/bin/activate

N=${N:-10000}
SEED=${SEED:-1234}
MODEL_PATH=${MODEL_PATH:-POMO/NEW_py_ver/TSP/POMO/result/saved_tsp100_model2_longTrain}
EPOCH=${EPOCH:-3100}
TAG=${TAG:-longtrain}
echo "[eval] N=$N seed=$SEED tag=$TAG model=$MODEL_PATH epoch=$EPOCH host=$(hostname)"
python - <<'PY'
import torch
print("[eval] torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY

srun python baseline/eval_pomo.py \
  --pomo-root POMO/NEW_py_ver/TSP \
  --instances data/tsp100_test_seed${SEED}.pt \
  --model-path "$MODEL_PATH" \
  --epoch "$EPOCH" \
  --aug 8 --batch-size 500 \
  --out results/pomo_eval_${TAG}_n${N}.npz
