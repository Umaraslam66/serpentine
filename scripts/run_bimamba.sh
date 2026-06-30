#!/bin/bash
# PREPARED, NOT LAUNCHED — bidirectional-Mamba discriminator probe (reviewer-staged).
#
# Trains a bidirectional (forward+backward scan, summed) Mamba encoder FROM SCRATCH,
# everything else identical to the provisionally-KILL'd unidirectional run:
#   250k steps, seed 1, Hilbert order, same POMO decoder / RL / budget,
#   single-traj + multistart eval every 20k. Param-matched (5 bidir layers = 1.248M total
#   vs the 10-layer unidirectional mamba's 1.250M), so this isolates DIRECTIONALITY, not capacity.
#
# Reads the verdict against the unidirectional 8.34% single-traj baseline at 250k:
#   * closes most of the ~5.4% gap to attention -> failure was CAUSALITY (shallow, fixable, O(N) kept)
#   * barely moves                              -> FIXED-STATE CAPACITY (deep)
#                                                  -> next: Mamba + a THIN global channel (cheap
#                                                     attention over Hilbert-segment summaries)
#
# Launch ONLY on reviewer ruling. Rides the next GPU window. Set SLURM_ACCOUNT before running.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT before launching}"

sbatch --account="$SLURM_ACCOUNT" --time=20:00:00 --job-name=serp-bimamba \
  --export=ALL,STEPS=250000,SEED=1,EVALN=1000,ENC=bimamba,ORDER=hilbert,KFLAG=--use-kernel,EVAL_EVERY=20000,MAX_HOURS=0 \
  scripts/slurm/train.sbatch
