#!/bin/bash
# One-time: install the mamba-ssm CUDA kernel + causal-conv1d into the venv.
# Build on a LOGIN node (internet + CUDA toolkit). Compiles from source against the
# pinned torch 2.2.2; MAX_JOBS limited to be a good login-node citizen.
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"
module load python/3.11.7 cuda/12.2 gcc/12.2.0
source .venv/bin/activate

export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export MAX_JOBS="${MAX_JOBS:-4}"
echo "CUDA_HOME=$CUDA_HOME  MAX_JOBS=$MAX_JOBS"
nvcc --version | tail -2

pip install --upgrade ninja packaging setuptools wheel

# Build from source against the installed torch (no isolation -> uses our torch 2.2.2).
pip install -v --no-build-isolation "causal-conv1d==1.4.0"
pip install -v --no-build-isolation "mamba-ssm==2.2.2"
# mamba-ssm pulls a too-new transformers (drops GreedySearchDecoderOnlyOutput, which its
# __init__ chain imports). We only use selective_scan_fn; pin a transformers that imports.
pip install "transformers==4.44.2"

python - <<'PY'
import torch, causal_conv1d, mamba_ssm
print("torch", torch.__version__, "| mamba_ssm", mamba_ssm.__version__)
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print("selective_scan_fn imported OK")
PY
echo "INSTALL DONE"
