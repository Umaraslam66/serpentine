#!/bin/bash
# One-time: build the LKH-3 optimality oracle on a Leonardo LOGIN node.
# Creates a version-independent symlink tools/lkh -> the built binary.
set -euo pipefail
ROOT=/leonardo_work/AIFAC_P02_548/mamba-route
cd "$ROOT"
module load gcc/12.2.0
mkdir -p tools && cd tools

VERS=${LKH_VERSION:-"3.0.13 3.0.12 3.0.11 3.0.9 3.0.6"}
got=""
for v in $VERS; do
  if [ -x "LKH-$v/LKH" ]; then got=$v; break; fi
done
if [ -z "$got" ]; then
  for v in $VERS; do
    url="http://akira.ruc.dk/~keld/research/LKH-3/LKH-$v.tgz"
    echo "trying $url"
    if wget -q "$url"; then
      tar xf "LKH-$v.tgz"
      (cd "LKH-$v" && make -j4)
      got=$v
      break
    fi
  done
fi
[ -n "$got" ] || { echo "ERROR: could not download/build any LKH version"; exit 1; }

ln -sf "$ROOT/tools/LKH-$got/LKH" "$ROOT/tools/lkh"
echo "LKH $got built -> $ROOT/tools/lkh"
"$ROOT/tools/lkh" --version 2>/dev/null || true
ls -la "$ROOT/tools/lkh"
