#!/bin/bash
#SBATCH --job-name=serp-bestofk
#SBATCH --partition=boost_usr_prod
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# Best-of-k serialization-ensemble probe (eval-only). Pass CKPT_TAG (e.g.
# calib_mamba_random_s1), ENC (mamba|bimamba), TRAINED_ORDER, K at submit.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
command -v module >/dev/null 2>&1 && module load python/3.11.7 cuda/12.2 gcc/12.2.0 || true
# shellcheck disable=SC1091
source scripts/_common.sh
command -v gcc >/dev/null 2>&1 && \
  export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)${LD_PRELOAD:+:$LD_PRELOAD}"

CKPT_TAG=${CKPT_TAG:?set CKPT_TAG}; ENC=${ENC:-mamba}
TRAINED_ORDER=${TRAINED_ORDER:-random}; K=${K:-8}; EVALN=${EVALN:-1000}
srun python scripts/bestofk_probe.py --ckpt "$WORKDIR/results/${CKPT_TAG}.ckpt.pt" \
  --encoder "$ENC" --trained-order "$TRAINED_ORDER" --k "$K" --use-kernel \
  --eval-instances "$WORKDIR/data/tsp100_test_seed1234.pt" \
  --eval-opt "$WORKDIR/results/lkh_opt_n10000.npy" --eval-n "$EVALN" \
  --out "$WORKDIR/results/${CKPT_TAG}.bestofk.json"
