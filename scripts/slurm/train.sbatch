#!/bin/bash
#SBATCH --job-name=serpentine-train
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# One encoder config per job (attention + mamba run concurrently on separate nodes).
# Resumable: re-submitting continues from the latest checkpoint. Pass --account at submit.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
command -v module >/dev/null 2>&1 && module load python/3.11.7 cuda/12.2 gcc/12.2.0 || true
# shellcheck disable=SC1091
source scripts/_common.sh
# mamba-ssm's compiled .so may need a newer libstdc++ than the compute-node system one.
command -v gcc >/dev/null 2>&1 && \
  export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)${LD_PRELOAD:+:$LD_PRELOAD}"

STEPS=${STEPS:?set STEPS}; SEED=${SEED:-1}; EVALN=${EVALN:-1000}
ENC=${ENC:-attention}; ORDER=${ORDER:-hilbert}; KFLAG=${KFLAG:-}   # "--use-kernel" for mamba
EVAL_EVERY=${EVAL_EVERY:-0}; MAX_HOURS=${MAX_HOURS:-0}             # periodic eval + wall cap
GLOBAL=${GLOBAL:-none}                                             # none|mean|segment global channel
TAG=${ENC}_${ORDER}; [ "$GLOBAL" != "none" ] && TAG=${TAG}_${GLOBAL}
echo "[train] ENC=$ENC ORDER=$ORDER KFLAG='$KFLAG' GLOBAL=$GLOBAL STEPS=$STEPS SEED=$SEED EVAL_EVERY=$EVAL_EVERY MAX_HOURS=$MAX_HOURS host=$(hostname)"
srun python -m serpentine.rl --encoder "$ENC" --order "$ORDER" $KFLAG \
  --global-channel "$GLOBAL" \
  --steps "$STEPS" --seed "$SEED" \
  --eval-instances "$WORKDIR/data/tsp100_test_seed1234.pt" \
  --eval-opt "$WORKDIR/results/lkh_opt_n10000.npy" --eval-n "$EVALN" \
  --ckpt "$WORKDIR/results/calib_${TAG}_s${SEED}.ckpt.pt" \
  --ckpt-every 2000 --log-every 200 \
  --eval-every "$EVAL_EVERY" --max-hours "$MAX_HOURS" \
  --out "$WORKDIR/results/calib_${TAG}_s${SEED}.json"
