# CPU diagnostic battery — mamba/hilbert s1 @ 250k (forward-only, no retraining)

All probes pure-PyTorch on CPU (use_kernel=False; kernel↔pure parity <1e-3), fixed
**100-instance** held-out subset, weights frozen. Script: `scripts/diagnostics.py`.
Raw numbers: `calib_mamba_hilbert_s1.diagnostics.json`.

Harness check: the directionality "forward" re-implementation reproduces the real
`encoder.forward` single-traj gap to 1e-12 (8.31566% == 8.31566%), and the forward
baseline matches the full-set 250k value (8.34%) — the override path is faithful.

## A. DIRECTIONALITY (suggestive only — reverse weights are untrained)

| pass | single-traj gap % | Δ vs forward |
|---|---|---|
| forward (baseline) | 8.316 | — |
| Hilbert reversed (R→L) | 14.962 | **+6.65** |
| avg(forward, reverse) embeddings | 9.794 | +1.48 |

**One-line:** No gap *drop* — reversing or averaging only hurts, so there is no free
global signal a naive bidirectional pass recovers here; reverse being far worse just
re-confirms the model is strongly tuned to the forward scan direction.

## B. REPRESENTATION HEALTH (mamba vs attention encoder embeddings, N=10,000 node vecs)

| metric | mamba | attention |
|---|---|---|
| participation ratio (eff. rank, /128) | **31.7** (24.7%) | 26.0 (20.3%) |
| mean pairwise cosine | 0.116 | 0.102 |
| per-dim variance (mean / min / max) | 1.74 / 0.80 / 3.17 | 1.12 / 0.81 / 1.45 |
| dead dims (<1% of max var) | 0 | 0 |

**One-line:** **No representation bottleneck** — mamba's embeddings are *higher*-rank than
attention's, similarly de-correlated, with zero dead dimensions; the failure is not a
collapsed/low-rank encoder, so the problem lies in *what* the embeddings encode, not their health.

## C. STATE DECAY / RECEPTIVE FIELD (finite-difference, ordering held fixed)

Sensitivity = L2 change in other nodes' embeddings when one node's coord is nudged (ε=0.02).

| |sequence separation| | mean Δ‖emb‖ |
|---|---|
| 1–2 | 0.362 |
| 3–5 | 0.128 |
| 6–10 | 0.048 |
| 11–20 | 0.019 |
| 21–50 | 0.0036 |
| 51–99 | 0.00019 |

Directional: **downstream mean Δ = 0.0574, upstream mean Δ = 0.0000 (exactly)**.

**One-line:** The encoder is **strictly causal and short-sighted** — a node influences *only*
nodes after it in the Hilbert sequence (zero upstream), and even downstream the effect decays
~99% by 20 positions, so each node integrates only a short one-directional window and never
sees global structure (unlike attention's all-pairs, both-direction mixing). This is the
strongest mechanistic candidate for the capacity gap.

## D. CURVE GEOMETRY (model-free; k=5 nearest 2D neighbours, 100 instances, 50k pairs)

| Hilbert-line separation of spatial neighbours | value |
|---|---|
| mean / median / p90 / max | 6.87 / 2.0 / 17.0 / 91 |
| fraction of spatial neighbours **>10 apart on the line** | **14.4%** |

**One-line:** Hilbert preserves locality for most neighbours (median separation 2), but its
space-filling boundary discontinuities scatter **~1 in 7** physically-adjacent nodes >10
positions apart on the line — and (per C) the scan's short causal window cannot reconnect them,
so ~14% of true spatial adjacencies are effectively invisible to the Mamba encoder.

## Synthesis (mechanism, not a new ruling)

The candidate's embeddings are *healthy* (B), so the gap is not encoder collapse. It is a
**receptive-field** limitation: the scan is one-directional and short (C), while the Hilbert
serialization — though locally good — throws ~14% of spatial neighbours far apart on the line
(D). Those scattered, causally-masked, distance-decayed relationships are exactly what
all-pairs attention captures for free. Directionality (A) shows a naive bidirectional patch
does not recover them with the current weights. Net: a structural encoder limitation, not a
training/representation artifact — consistent with the provisional KILL. The from-scratch
random/sort ablation (queued behind GPU maintenance) remains the formal capacity control.
