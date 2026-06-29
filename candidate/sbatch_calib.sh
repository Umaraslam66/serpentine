#!/bin/bash
#SBATCH --job-name=mamba-calib
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_548
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# One encoder config per job (so attention + mamba run concurrently on separate nodes).
# Resumable: re-submitting the same job continues from the latest checkpoint.
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"
mkdir -p logs results
module load python/3.11.7 cuda/12.2 gcc/12.2.0
source .venv/bin/activate
# mamba-ssm's compiled .so may need a newer libstdc++ than the compute-node system one.
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)${LD_PRELOAD:+:$LD_PRELOAD}"

STEPS=${STEPS:?set STEPS}
SEED=${SEED:-1}
EVALN=${EVALN:-1000}
ENC=${ENC:-attention}
ORDER=${ORDER:-hilbert}
KFLAG=${KFLAG:-}                 # "--use-kernel" for mamba
INST=data/tsp100_test_seed1234.pt
OPT=results/lkh_opt_n10000.npy
TAG=${ENC}_${ORDER}
echo "[calib] ENC=$ENC ORDER=$ORDER KFLAG='$KFLAG' STEPS=$STEPS SEED=$SEED host=$(hostname)"

srun python candidate/train.py --encoder "$ENC" --order "$ORDER" $KFLAG \
  --steps "$STEPS" --seed "$SEED" \
  --eval-instances "$INST" --eval-opt "$OPT" --eval-n "$EVALN" \
  --ckpt-every 2000 --log-every 200 \
  --out "results/calib_${TAG}_s${SEED}.json"
