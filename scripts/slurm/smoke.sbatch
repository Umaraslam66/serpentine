#!/bin/bash
#SBATCH --job-name=serpentine-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# Smoke: both encoders, 1 seed, few hundred steps. Throughput + greedy gaps; exercises
# the Hilbert/zorder/random/sort ablation. Pass --account at submit.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
command -v module >/dev/null 2>&1 && module load python/3.11.7 cuda/12.2 gcc/12.2.0 || true
# shellcheck disable=SC1091
source scripts/_common.sh
command -v gcc >/dev/null 2>&1 && \
  export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)${LD_PRELOAD:+:$LD_PRELOAD}"

STEPS=${STEPS:-300}; SEED=${SEED:-1}; EVALN=${EVALN:-1000}; KFLAG=${KFLAG:-}
INST="$WORKDIR/data/tsp100_test_seed1234.pt"
OPT="$WORKDIR/results/lkh_opt_n10000.npy"
echo "[smoke] STEPS=$STEPS SEED=$SEED EVALN=$EVALN host=$(hostname)"

run() {  # encoder order extra-flags
  echo "=== ${1}/${2} ==="
  python -m serpentine.rl --encoder "$1" --order "$2" ${3:-} --steps "$STEPS" --seed "$SEED" \
    --eval-instances "$INST" --eval-opt "$OPT" --eval-n "$EVALN" \
    --ckpt "$WORKDIR/results/smoke_${1}_${2}_s${SEED}.ckpt.pt" \
    --out "$WORKDIR/results/smoke_${1}_${2}_s${SEED}.json"
}
run attention hilbert            # order ignored by attention encoder
run mamba hilbert "$KFLAG"
run mamba random "$KFLAG"
run mamba sort "$KFLAG"
