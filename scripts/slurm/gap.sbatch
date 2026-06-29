#!/bin/bash
#SBATCH --job-name=serpentine-gap
#SBATCH --partition=lrd_all_serial
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out
# Combine POMO lengths + LKH optimal into the final gap report. Pass --account at submit.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source scripts/_common.sh
command -v module >/dev/null 2>&1 && module load python/3.11.7 || true
N=${N:-10000}; TAG=${TAG:-longtrain}
python -m serpentine.metrics \
  --pomo "$WORKDIR/results/pomo_eval_${TAG}_n${N}.npz" \
  --opt "$WORKDIR/results/lkh_opt_n${N}.npy" \
  --out "$WORKDIR/results/baseline_report_${TAG}_n${N}.json"
