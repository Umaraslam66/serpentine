#!/bin/bash
# One-time: build the LKH-3 optimality oracle. Creates a version-independent
# symlink <work>/tools/lkh -> the built binary.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WORKDIR="${SERPENTINE_WORK:-$REPO_ROOT}"
command -v module >/dev/null 2>&1 && module load gcc/12.2.0 || true
mkdir -p "$WORKDIR/tools" && cd "$WORKDIR/tools"

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

ln -sf "$WORKDIR/tools/LKH-$got/LKH" "$WORKDIR/tools/lkh"
echo "LKH $got built -> $WORKDIR/tools/lkh"
ls -la "$WORKDIR/tools/lkh"
