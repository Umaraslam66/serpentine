#!/bin/bash
# PREPARED, NOT LAUNCHED — global-channel controlled experiment (reviewer-staged, behind
# tonight's confirmatory runs). Changes ONLY the encoder global channel; everything else is
# identical to the mamba/hilbert baseline (POMO-RL, 250k, seed 1, Hilbert, same decoder/budget,
# mamba-ssm kernel, single-traj+multistart eval every 20k).
#
#   Variant 0  GLOBAL=none     -> the existing KILL baseline (8.34%), NOT re-run here.
#   Variant A  GLOBAL=mean     -> ECO-style mean-pool global g (cheapest channel).        +1.32% params
#   Variant B  GLOBAL=segment  -> structured: pool S=10 Hilbert segments, O(S^2) attention  +2.62% params
#                                 over the summaries, scatter back to nodes (repairs the ~14%
#                                 scattered neighbours from Task-1 geometry that mean-pool blurs).
#
# Hypothesis (vs attention 2.95%, baseline 8.34%): if B closes most of the gap and A only
# partially, the STRUCTURED channel is the contribution — and it works under RL, which ECO
# (SFT+DPO) never tested. Launch ONLY on reviewer ruling. Set SLURM_ACCOUNT before running.
# Do NOT build the SFT+DPO pipeline (Phase 2 only if global-channel-under-RL still loses).
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT before launching}"

COMMON=(--account="$SLURM_ACCOUNT" --time=20:00:00)
EXPORT_BASE="ALL,STEPS=250000,SEED=1,EVALN=1000,ENC=mamba,ORDER=hilbert,KFLAG=--use-kernel,EVAL_EVERY=20000,MAX_HOURS=0"

sbatch "${COMMON[@]}" --job-name=serp-gc-mean \
  --export="${EXPORT_BASE},GLOBAL=mean"    scripts/slurm/train.sbatch
sbatch "${COMMON[@]}" --job-name=serp-gc-seg \
  --export="${EXPORT_BASE},GLOBAL=segment" scripts/slurm/train.sbatch
