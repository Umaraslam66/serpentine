#!/bin/bash
#SBATCH --job-name=mamba-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# Gate-0 smoke: both encoders, 1 seed, few hundred steps. Measures throughput
# (to size the shared budget) and greedy gaps; exercises the Hilbert/random/sort ablation.
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"
mkdir -p logs results
module load python/3.11.7
source .venv/bin/activate

STEPS=${STEPS:-300}
SEED=${SEED:-1}
EVALN=${EVALN:-1000}
INST=data/tsp100_test_seed1234.pt
OPT=results/lkh_opt_n10000.npy
echo "[smoke] STEPS=$STEPS SEED=$SEED EVALN=$EVALN host=$(hostname)"

run() {  # encoder order
  echo "=== ${1}/${2} ==="
  python candidate/train.py --encoder "$1" --order "$2" --steps "$STEPS" --seed "$SEED" \
    --eval-instances "$INST" --eval-opt "$OPT" --eval-n "$EVALN" \
    --out "results/smoke_${1}_${2}_s${SEED}.json"
}

run attention hilbert   # order ignored by attention encoder
run mamba hilbert
run mamba random
run mamba sort

echo "=== SUMMARY ==="
python - <<'PY'
import glob, json
rows = [json.load(open(f)) for f in sorted(glob.glob("results/smoke_*_s*.json"))]
for d in rows:
    print(f"{d['encoder']:9s} {d['order']:7s} steps={d['steps']:>4} "
          f"params={d['n_params']:>8} it/s={d['steps_per_sec']:>6.2f} "
          f"loss={d['final_loss']:>7.3f} "
          f"greedy_single={d['greedy_single_traj_gap_pct']:>7.2f}% "
          f"greedy_multi={d['greedy_multistart_gap_pct']:>7.2f}%")
PY
