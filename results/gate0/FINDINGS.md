# Gate 0 — Findings (mechanistic story so far)

**Question:** can a Hilbert-ordered Mamba encoder drive routing decisions as well as a
standard attention encoder on Euclidean TSP-100? **Answer so far: NO for the
encoder-swap as specified** (vanilla causal Mamba, single Hilbert curve) — and the formal
ablation (§4b) shows the Hilbert serialization itself is counterproductive: random
ordering trains better. This document is the running synthesis; the BiMamba and
global-channel results slot into the clearly-marked TODOs below.

## 1. Provisional KILL (seed-1, 250k steps, single-traj greedy vs LKH)

| encoder | single-traj gap | multi-start gap | params |
|---|---|---|---|
| attention (baseline) | **2.95%** | 1.91% | 1.27M |
| mamba/hilbert (candidate) | **8.34%** | 4.46% | 1.25M |
| Δ (candidate − baseline) | **+5.38% abs** | +2.55% abs | — |

Attention clears the ~3% seatbelt at 240k steps (15.36M examples); mamba/hilbert plateaus
at 8.3% and the curves *diverge* over the extension, not converge. Single seed — the formal
KILL needs the ablation (TODO C). Provisional KILL stands for the encoder-swap-as-specified.

## 2. It is not a wiring bug — order reaches the decoder

Inference-time order swap on the trained mamba/hilbert checkpoint (no retrain, n=200):
hilbert **8.38%** → sort **33.49%** → random **56.38%** single-traj. Feeding the wrong order
collapses the model (+48 abs), so the serialize→scan→un-serialize→decode path is intact and
strongly order-dependent. **Bug hypothesis refuted.**

## 3. It is not representation collapse (diagnostic B)

Encoder embeddings (10k node vectors): mamba effective rank **31.7** ≥ attention **26.0**,
mean pairwise cosine 0.116 vs 0.102, **0 dead dims** in either. The candidate's embeddings
are at least as healthy/high-rank as the baseline's — the failure is in *what* they encode,
not encoder health.

## 4. It is a receptive-field limitation (diagnostics A, C + geometry D, 1, 2)

- **C — causal & short.** Finite-difference: upstream sensitivity is **exactly 0** (a node
  never sees sequence-successors), and even downstream the effect **decays ~99% by ~20
  positions**. Each node integrates only a short, one-directional window.
- **A — no free bidirectional fix with current weights.** Reversing the Hilbert scan costs
  +6.65; averaging forward+reverse +1.48 (no drop) — suggestive only (reverse weights are
  untrained); the clean test is to *train* bidirectionally (TODO B).
- **D — Hilbert scatters ~14% of spatial neighbours.** 14.4% of k=5 nearest 2D neighbours land
  >10 positions apart on the line (curve boundary discontinuities); the short causal window
  cannot reconnect them, whereas all-pairs attention sees them for free.
- **Task 1 — a thin global channel would reconnect them.** Over S=10 segment summaries, 61% of
  scattered neighbours sit within 2 segment hops; full O(S²) attention over the segment
  summaries reconnects 100% in one hop, cheaply. Geometric precondition for the hybrid holds.
- **Task 2 — the gap is a scaling wall.** Mean line-separation of spatial neighbours grows
  7 → 48 as N grows 100 → 5000 (fraction >10-apart 14.6% → 22.2%), while the encoder's
  receptive field is fixed (~20). Relative locality is fine; **absolute** locality outruns a
  fixed-window scanner — so the limitation *worsens* with scale. (The scaling wall itself is
  geometry, not our contribution; ECO already runs a Mamba backbone on large-N TSP — see §6.)

## 4b. Formal ablation (ran 2026-07-01, retrieved 2026-07-17): the ordering hierarchy INVERTS

From-scratch mamba/random and mamba/sort (seed 1, 250k, identical everything else):
**random 6.40% < sort 7.12% < hilbert 8.34%** single-traj — and random beats hilbert at
*every* 20k eval, whole-curve (see `ablation/ABLATION.md`). The pre-registered KILL rule
("worse than attention by >1.0% AND hilbert clearly beats random") resolves in the
unanticipated direction: the second clause fails **inverted**. Consequences:

- The KILL broadens: every vanilla-Mamba ordering loses to attention (2.95%) under RL;
  the best is now *random* at 6.40% (gap 3.5% abs, not 5.4%).
- **The Hilbert-locality premise is refuted, not merely unproven** — spatial-locality
  serialization *hurts* training. Random ordering (fresh permutation per instance) acts as
  an invariance regularizer, forcing order-robust global features; locality is a crutch
  the short causal window overfits.
- No contradiction with §2's order probe: that shows inference-time dependence of a
  hilbert-*trained* model on its training order, not which order trains better.
- Single seed; hilbert/random seeds 2–3 queued 2026-07-17 to bound variance. §5's fork
  and the global-channel design read against the 8.34% hilbert baseline; interpretation of
  any global-channel win must now also be checked against the 6.40% random arm.

## 5. The fork this sets up (BiMamba discriminator — TODO B)

Param-matched bidirectional Mamba (forward+backward scan, 5 layers = 1.2485M ≈ uni 1.2498M),
everything else identical, trained from scratch 250k/seed-1. The point-cloud literature (§6)
expects bidirectionality to help **partially** — a backward scan still compresses history into a
finite hidden state (HydraMamba) — so read the result in **three** buckets, not binary:

| BiMamba @ 250k vs uni 8.34% / attn 2.95% | reading | next direction |
|---|---|---|
| **(a)** closes most of the ~5.4% gap | **causality-dominated** — directionality is the fix | bidirectional scan suffices (Vim/Hydra-style) |
| **(b)** **partial** close *(literature-predicted)* | directionality helps **AND** the fixed-state wall is real | **hybrid**: bi-Mamba + thin global channel over Hilbert-segment summaries (Task 1 shows it's viable & cheap; Task 2 shows it's increasingly needed at scale) |
| **(c)** ~no move | deep capacity limit / implementation issue | re-check wiring; reconsider SSM decision-transfer feasibility at this budget |

Holding: the hybrid (and any multi-curve arm) is **not** built yet — it is the bucket-(b)/(c)
contingency only.

## 6. Related work (reviewer-supplied; we cannot reach arXiv — taken as given, NOT verified by us)

**Honesty note: the diagnosis and the candidate fixes below are TABLE STAKES in the point-cloud
Mamba literature, not contributions of this study. FINDINGS must not claim them as novel.**

- The causal/short-window limitation of unidirectional Mamba vs attention's all-pairs is the
  canonical **PointMamba** observation.
- Known mitigations, all prior art: **bidirectional scan** (Vision Mamba, HydraMamba, Pamba);
  **multi-curve / shuffle serialization** (Trans-Hilbert); **hybrid local-conv + global
  aggregation** (PillarMamba). HydraMamba further notes a bidirectional scan *still* compresses
  history into the finite hidden state and therefore adds convolution — which is why we expect
  BiMamba to only **partially** close the gap (bucket (b) in §5).
- **ECO (2026)** already applies a Mamba backbone to TSP/CVRP at large N. The open question this
  study owns is **decision-transfer under RL** (POMO REINFORCE) — whether a Hilbert+Mamba encoder
  yields routing *decisions* as good as attention, which the point-cloud papers do not test. Any
  claim we make is about RL routing-decision quality, **not** about Mamba-for-points, the scaling
  wall, or these fixes being new.

---

## TODO — pending numbers (do not treat conclusions as final until filled)

- [ ] **(B) BiMamba seed-1 @ 250k** — RUNNING since 2026-07-17 21:27 (job 49636355,
      24h wall, GPU-smoked first). Decides the §5 fork. Collect via
      `scripts/collect_results.sh`.
- [x] **(C) Formal ablation** — mamba/random + mamba/sort seed-1 @250k COMPLETE
      (2026-07-01): **random 6.40% < sort 7.12% < hilbert 8.34%** — rule resolves
      inverted; Hilbert-locality premise refuted (§4b, `ablation/ABLATION.md`).
      Remaining: hilbert + random seeds 2–3 (queued 2026-07-17) to bound seed variance.
- [ ] Only if BiMamba lands in bucket (b)/(c): prototype + measure the hybrid thin-global-channel.
      NOT started — no hybrid or multi-curve arm exists yet (held by reviewer).

*Sources: `calibration/extension/{EXTENSION,ORDER_PROBE,DIAGNOSTICS,BIMAMBA}.md`, `GEOMETRY.md`.*
