# Model-free Hilbert-geometry studies (no model, CPU)

Both motivate/refute the hybrid direction *before* building it. Script:
`scripts/geometry_scaling.py`; raw numbers: `geometry_scaling.json`. k=5 nearest 2D
neighbours, "scattered" = >10 apart on the Hilbert line (probe D's metric). Sanity: the
N=100 real-set scattered fraction reproduces probe D exactly (14.426%).

## Task 1 — Segment-reconnect (would a thin global channel reconnect the scattered 14%?)

Chunk the 100-node Hilbert line into S contiguous segments; for the 7,213 scattered k-NN
pairs (14.4% of 50,000), measure segment-index separation. All scattered pairs are in
*different* segments by construction (a segment spans ≤ its size in line positions < 10).

| S | segment size | segdiff median | segdiff mean | p90 | within 1 seg | within 2 segs | within 3 segs |
|---|---|---|---|---|---|---|---|
| 10 | 10 | 2 | 3.10 | 8 | 26.9% | **61.2%** | 74.7% |
| 20 | 5 | 4 | 6.27 | 15 | 0%* | 11.4% | 34.9% |

\*with size-5 segments a >10 line gap is always ≥2 segments, so segdiff=1 is impossible.

**Answer: YES — a thin global channel reconnects the scattered neighbours, cheaply.**
At S=10, ~61% of scattered spatial neighbours sit within 2 segment hops, so even a *local*
coarse channel (a short scan over segment summaries) recovers most. And a **full all-to-all
attention over the S segment summaries is O(S²) = 100 (S=10) / 400 (S=20)** — trivially
cheap — so it reconnects **100% of scattered pairs in a single segment hop** regardless of
their line separation. The geometric precondition for the hybrid holds.

*Caveat:* segment summaries are pooled, so this proves a node can *reach* its scattered
neighbour's segment, not that the pooled summary preserves that specific neighbour's signal —
that sufficiency is the empirical question the hybrid itself would answer.

## Task 2 — Scaling wall (does serialization-locality loss worsen with N?)

Fraction of k=5 spatial neighbours >10 apart on the line, and their absolute line separation,
vs N (bits=10 grid so quantization is negligible; real-set bits=7 row anchors to probe D).

| N | instances | frac >10 apart | frac > N/10 apart | median sep | mean sep | p90 sep |
|---|---|---|---|---|---|---|
| 100 (real, bits 7) | 100 | 14.43% | 14.43% | 2 | 6.87 | 17 |
| 100 (uniform) | 100 | 14.57% | 14.57% | 2 | 7.04 | 17 |
| 500 | 50 | 18.95% | 6.02% | 3 | 15.29 | 28 |
| 1000 | 20 | 20.70% | 4.22% | 3 | 21.22 | 37 |
| 5000 | 5 | **22.22%** | 1.97% | 3 | **48.46** | 51 |

**There is a scaling wall — measured against a fixed receptive field.**
- The **absolute** scatter grows: fraction >10-apart rises 14.6% → 22.2%, and the **mean line
  separation of spatial neighbours grows 7 → 48** (p90 17 → 51) from N=100 to N=5000.
- The **relative** scatter (> N/10) *shrinks* 14.6% → 2.0% — Hilbert's asymptotic locality is
  fine in relative terms.
- The catch: diagnostic C showed the causal Mamba's effective receptive field is **absolute and
  fixed** (~99% decayed by ~20 sequence positions). At N=100 the mean neighbour separation (7)
  sits inside that window; by N=5000 it (48) is **~2.5× beyond it**. So a fixed-window scanner
  falls progressively further behind as N grows — exactly where the large-N thesis lives, and
  exactly what a scale-independent global channel is needed to fix.
