#!/bin/bash
# One-time: build the pinned Python venv on a Leonardo LOGIN node (has internet).
# Compute nodes have no internet, so all installs happen here.
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"

module load python/3.11.7
python -m venv .venv          # built in $WORK, never $HOME
source .venv/bin/activate
pip install --upgrade pip

# torch 2.2.2 (cu121 wheel, bundles CUDA runtime; needs only the A100 driver).
# 2.2.x still supports the legacy torch.set_default_tensor_type call POMO uses.
pip install "torch==2.2.2" "numpy==1.26.4"

python - <<'PY'
import torch, numpy
print("torch", torch.__version__, "| numpy", numpy.__version__)
print("set_default_tensor_type available:", hasattr(torch, "set_default_tensor_type"))
PY
echo "env_setup done -> $ROOT/.venv"
