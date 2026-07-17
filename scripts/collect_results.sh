#!/bin/bash
# Idempotent result collector for the 2026-07-17 launch wave (see results/gate0/ablation/
# ABLATION.md §4 for why this exists: completed cluster results must never again sit
# unfetched). Safe to run any time; fetches only files that exist remotely.
#
#   bash scripts/collect_results.sh        # status + fetch into results/gate0/incoming/
#
# Jobs in this wave (submitted 2026-07-17, account AIFAC_P02_548):
#   49636355 serp-bimamba   -> calib_bimamba_hilbert_s1
#   49636356 serp-abl-h2    -> calib_mamba_hilbert_s2
#   49636357 serp-abl-h3    -> calib_mamba_hilbert_s3
#   49636358 serp-abl-r2    -> calib_mamba_random_s2
#   49636359 serp-abl-r3    -> calib_mamba_random_s3
#   (global-channel pair submitted separately -> calib_mamba_hilbert_{mean,segment}_s1)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REMOTE=leonardo
RDIR=/leonardo_work/AIFAC_P02_548/mamba-route
DEST=results/gate0/incoming
mkdir -p "$DEST"

echo "== queue =="
ssh -o BatchMode=yes "$REMOTE" 'squeue -u $USER'
echo "== this wave (sacct) =="
ssh -o BatchMode=yes "$REMOTE" 'sacct -u $USER -S 2026-07-17 -X --format=JobID,JobName%20,State,Elapsed,ExitCode' | grep -E 'serp|JobID|----' || true

echo "== fetching finished artifacts =="
TAGS="bimamba_hilbert_s1 mamba_hilbert_s2 mamba_hilbert_s3 mamba_random_s2 mamba_random_s3 mamba_hilbert_mean_s1 mamba_hilbert_segment_s1"
for t in $TAGS; do
  for ext in json curve.jsonl; do
    f="calib_${t}.${ext}"
    if ssh -o BatchMode=yes "$REMOTE" "test -f $RDIR/results/$f"; then
      scp -q -o BatchMode=yes "$REMOTE:$RDIR/results/$f" "$DEST/" && echo "fetched $f"
    fi
  done
done
echo "== done; artifacts in $DEST — analyze + commit in a session, do not auto-score =="
