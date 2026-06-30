# Staged follow-up: bidirectionally-trained Mamba encoder (discriminator probe)

**Status: PREPARED, NOT LAUNCHED.** Awaiting reviewer ruling; staged to ride the next GPU window.

## Why this is the highest-information next test

The CPU diagnostic battery isolated the candidate's failure to a **receptive-field**
limitation, not representation collapse (B: mamba eff-rank 31.7 ≥ attention 26.0, 0 dead
dims). Specifically the encoder is **strictly causal** — diagnostic C measured *exactly*
zero upstream sensitivity — and short (≈99% decay by 20 sequence positions), while the
Hilbert curve scatters ~14% of spatial neighbours >10 positions apart (D). Directionality
probe A could only *suggest* this with untrained reverse weights (reverse +6.65, avg +1.48,
no drop). The clean test is to **train** a bidirectional scan from scratch.

## What it changes (and what it does not)

- **Encoder:** `BiMambaEncoder` — each layer runs a forward MambaBlock and a backward
  MambaBlock (on the flipped Hilbert sequence, re-flipped), summed. 5 layers (two mixers
  each) ⇒ **1,248,512 total params vs the unidirectional 1,249,792 (0.10% diff)**. Test
  `test_bimamba.py` pins the bidirectional upstream-sensitivity property (causal = exact 0,
  bidir = 1.4e-2 one step up) and the param match.
- **Unchanged:** Hilbert order, POMO decoder, RL/optimizer (Adam 1e-4, no warmup), batch
  64×100, 250k steps, seed 1, single-traj+multistart eval every 20k, mamba-ssm kernel
  (`--use-kernel`; each sub-mixer is a standard MambaBlock covered by the parity gate).

## Decision rule (vs the unidirectional 8.34% single-traj @ 250k; attention 2.95%)

| outcome at 250k | reading | next direction |
|---|---|---|
| closes most of the ~5.4% gap | failure was **CAUSALITY** — shallow, fixable, O(N) preserved | bidirectional Mamba is the fix |
| barely moves | failure is **FIXED-STATE CAPACITY** — deep | **hybrid**: Mamba locally + a thin global channel (cheap attention over Hilbert-segment cluster summaries) — the genuinely novel, scale-relevant question |

## Launch (on ruling only)

```
SLURM_ACCOUNT=<account> bash scripts/run_bimamba.sh
```

Writes `results/calib_bimamba_hilbert_s1.{ckpt.pt,curve.jsonl,json}`. ~14–16h on one A100
(20h wall, no early cap), same shape as the unidirectional 250k run.
