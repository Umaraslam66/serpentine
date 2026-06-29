#!/bin/bash
#SBATCH --job-name=serpentine-gate
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# Mechanism gate: mamba-ssm kernel must match the pure-PyTorch reference (<1e-3).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source scripts/_common.sh
command -v module >/dev/null 2>&1 && module load python/3.11.7 cuda/12.2 gcc/12.2.0 || true
command -v gcc >/dev/null 2>&1 && \
  export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)${LD_PRELOAD:+:$LD_PRELOAD}"
srun python tests/test_kernel_parity.py
