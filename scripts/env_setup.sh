#!/bin/bash
# One-time setup (run where there is internet; on an HPC login node). Builds the pinned
# venv, installs the serpentine package, and clones the pinned POMO baseline.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export WORKDIR="${SERPENTINE_WORK:-$REPO_ROOT}"

command -v module >/dev/null 2>&1 && module load python/3.11.7 || true
python -m venv "$WORKDIR/.venv"
# shellcheck disable=SC1091
source "$WORKDIR/.venv/bin/activate"
pip install --upgrade pip

# torch 2.2.2 (cu121 wheel; needs only the GPU driver). numpy pinned for ABI.
pip install "torch==2.2.2" "numpy==1.26.4"
pip install -e "$REPO_ROOT"     # editable serpentine package

# Pinned POMO baseline (external MIT dependency: env, attention encoder, decoder).
POMO_DIR="$REPO_ROOT/POMO"
if [ ! -d "$POMO_DIR" ]; then
  git clone https://github.com/yd-kwon/POMO.git "$POMO_DIR"
  git -C "$POMO_DIR" checkout d7c3d6e
fi

python - <<'PY'
import torch, numpy
print("torch", torch.__version__, "| numpy", numpy.__version__)
PY
echo "env ready -> $WORKDIR/.venv ; POMO -> $POMO_DIR (@ d7c3d6e)"
