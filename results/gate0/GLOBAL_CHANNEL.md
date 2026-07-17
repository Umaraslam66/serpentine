# Staged experiment: structured vs mean-pool global channel under RL

**Status: LAUNCHED 2026-07-17** — jobs 49636372 (A/mean) + 49636373 (B/segment), after
GPU smokes (49636341/49636354, param counts match the table below exactly). NOTE the
intervening ablation result (`ablation/ABLATION.md`): trained-from-scratch random order
(6.40%) beats hilbert (8.34%), so the "repairs the scattered Hilbert neighbours" story in
this doc is under pressure — read any Variant-B win against the 6.40% random arm too, not
only the 8.34% hilbert baseline. One controlled experiment; **only the encoder global
channel changes**,
everything else identical to the mamba/hilbert baseline (POMO-RL, 250k, seed 1, Hilbert,
same POMO decoder/budget, mamba-ssm kernel, single-traj+multistart eval every 20k).

## Why this is our novel increment (given ECO)

ECO (arXiv 2602.20730) already runs a Mamba backbone on routing, but: (i) its encoder is plain
stacked Mamba over **raw** coordinates with a **mean-pool** as its only global mechanism; (ii) it
trains by **SFT + iterative DPO** on LKH/local-search tours, **not RL**; (iii) its wins are in
large-N generalization, not N=100, and its ablation credits the SFT/LS bootstrapping heavily.
So vanilla Mamba+Hilbert+POMO-RL losing at N=100 (our KILL) is consistent with ECO. Our open
increment is a **structured** global channel (vs ECO's mean-pool), grounded in our
receptive-field diagnostics, and tested **under RL** — which ECO never did.

## The three variants (decoder byte-identical; only the channel differs)

Both channels add a per-node contribution to the un-serialized encoder output, `H → H + G(H)`;
the POMO decoder is untouched.

| variant | global channel | total params | Δ vs baseline | channel params |
|---|---|---|---|---|
| **0** | none (existing KILL baseline, 8.34%) | 1,249,792 | — | 0 |
| **A** | mean-pool `g = proj(mean H)`, broadcast (ECO-style) | 1,266,304 | **+1.32%** | 16,512 |
| **B** | segment: pool S=10 Hilbert segments → O(S²) attention → scatter to nodes | 1,282,560 | **+2.62%** | 32,768 |

Identical 10-layer Mamba backbone across all three. A is uniform across nodes, so it can only
reach the routing decision through the decoder's query path (a single graph vector cannot
differentiate nodes via the selection softmax). B is per-node: it lets two physically-adjacent
nodes that the Hilbert curve scattered into different segments exchange information in one
segment-attention hop — directly repairing the ~14% scattered neighbours from Task-1 geometry,
which a single mean blurs together.

## Correctness (TDD, `tests/test_global_channel.py`, 8/8)

Node identity is the risk (a wrong scatter silently swaps nodes). Pinned three ways:
`_scatter` gives node n exactly `seg_ctx[node_segment[n]]`; `node_segment[n] == hilbert_pos[n] // (P/S)`;
and the whole encoder with the segment channel is **permutation-equivariant** to 1e-5 (permuting
input nodes permutes the output identically — the end-to-end analogue of the Hilbert un-serialize
test). `global_mode="none"` leaves the encoder byte-identical to the KILL baseline.

## Decision rule (vs attention 2.95%, baseline 8.34%)

| outcome | reading |
|---|---|
| **B closes most of the gap, A only partial** | the **structured** global channel is the contribution — and it works under RL (ECO untested) |
| A ≈ B (both close it) | a cheap mean-pool suffices; structure is not the lever (still a positive RL result) |
| neither closes it | the global channel is not the fix under RL → Phase 2 (SFT+DPO warm-up, ECO-style) — **not built yet** |

## Launch (on ruling only)

```
SLURM_ACCOUNT=<account> bash scripts/run_global_channel.sh   # submits A (mean) + B (segment)
```

Writes `results/calib_mamba_hilbert_{mean,segment}_s1.{ckpt.pt,curve.jsonl,json}`. Variant 0 is
the existing baseline (not re-run). ~14–16h/job on one A100. **Phase 2 (SFT+DPO) is NOT built** —
only to be considered if global-channel-under-RL still loses.
