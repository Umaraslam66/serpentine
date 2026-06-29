#!/bin/bash
#SBATCH --job-name=mamba-gate
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# Mechanism gate: mamba-ssm kernel must match the pure-PyTorch reference (<1e-3).
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"
mkdir -p logs
module load python/3.11.7 cuda/12.2 gcc/12.2.0
source .venv/bin/activate
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)${LD_PRELOAD:+:$LD_PRELOAD}"
srun python candidate/test_kernel_parity.py
