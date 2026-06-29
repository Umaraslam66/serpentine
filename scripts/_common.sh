# Shared environment for serpentine scripts. Caller sets REPO_ROOT, then `source`s this.
# Portable: no hardcoded cluster paths. Override data/results location with SERPENTINE_WORK.
: "${REPO_ROOT:?set REPO_ROOT before sourcing scripts/_common.sh}"
export WORKDIR="${SERPENTINE_WORK:-$REPO_ROOT}"
export POMO_ROOT="${POMO_ROOT:-$REPO_ROOT/POMO/NEW_py_ver/TSP}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
mkdir -p "$WORKDIR/data" "$WORKDIR/results" "$WORKDIR/logs"
if [ -f "$WORKDIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$WORKDIR/.venv/bin/activate"
fi
