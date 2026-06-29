#!/bin/bash
#SBATCH --job-name=mamba-base-gap
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out
# Combine POMO lengths + LKH optimal into the final gap report.
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"
module load python/3.11.7
source .venv/bin/activate
N=${N:-10000}
python baseline/compute_gap.py \
  --pomo results/pomo_eval_n${N}.npz \
  --opt results/lkh_opt_n${N}.npy \
  --out results/baseline_report_n${N}.json
