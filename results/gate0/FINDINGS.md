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
- **SEED UPDATE 2026-07-18:** seeds 2–3 do NOT confirm the strict inversion — hilbert
  spans 6.36–8.34 (mean 7.50 ± 1.03), random 6.40–7.45 (mean 6.90 ± 0.53), ranges overlap.
  Supported claim downgraded to: **Hilbert confers no benefit over random and doubles seed
  variance.** The §5 fork resolved to bucket (b) with BiMamba at 5.55/1.89 (multistart
  parity with attention, seed-1); both global-channel variants refuted. Full scoring:
  `incoming/WAVE_ANALYSIS.md`.

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

## 6. Related work (web-verified 2026-07-17 with arXiv IDs; originally reviewer-supplied)

**Honesty note: the diagnosis and the candidate fixes below are TABLE STAKES in the point-cloud
Mamba literature, not contributions of this study. FINDINGS must not claim them as novel.**

- The causal/short-window limitation of unidirectional Mamba vs attention's all-pairs is the
  canonical **PointMamba** (2402.10739, NeurIPS 2024) observation.
- Known mitigations, all prior art: **bidirectional scan** (Vision Mamba 2401.09417, HydraMamba
  2507.19778, Pamba 2406.17442); **multi-curve / shuffle serialization** (e.g. PTv3's
  Hilbert+Trans-Hilbert shuffle, 2312.10035); **hybrid local + global aggregation**
  (PillarMamba 2505.05397). HydraMamba further notes a bidirectional scan *still* compresses
  history into the finite hidden state and therefore adds convolution — which is why we expect
  BiMamba to only **partially** close the gap (bucket (b) in §5).
- **ECO (2602.20730, 2026)** applies a Mamba encoder-decoder to TSP/CVRP, trained via SFT +
  iterative DPO (not policy-gradient RL); it targets large instances (smallest TSP tested is
  N=200, where it beats POMO at every tested size, with margin growing in N) and never tests
  N=100. The open question this study owns is **decision-transfer under RL** (POMO REINFORCE).
- **Positioning of the §4b inversion (fact-checked):** the published serialization ablations
  point the OTHER way in supervised point clouds — PointMamba reports Hilbert/Trans-Hilbert
  *beating* random by +1.20/+1.73% on ScanObjectNN; Point Cloud Mamba (2403.00762, AAAI 2025)
  reports simple ≈ curve orders (neutral). No paper found reporting random > space-filling for
  Mamba/SSM in any domain, and no Mamba-NCO work occupies the matched-budget POMO-RL N=100
  cell. So the *strong* claim ("locality prior actively hurts under RL") appears novel; the
  *weak* claim ("locality not essential") is partially anticipated by PCM — cite it to preempt.
  Any claim we make is about RL routing-decision quality, **not** about Mamba-for-points, the
  scaling wall, or these fixes being new.

---

## TODO — pending numbers (do not treat conclusions as final until filled)

- [x] **(B) BiMamba seed-1 @ 250k** — DONE 2026-07-18 (job 49636355): single-traj
      **5.55%** (52% of the gap closed → **bucket (b)**), multistart **1.89% ≈ attention
      1.91%** (parity, unregistered outcome). See `incoming/WAVE_ANALYSIS.md` §2.
- [x] **Global-channel A/B @ 250k** — DONE 2026-07-18: mean 7.86 (within hilbert seed
      noise), segment 8.38 (= baseline). **Both refuted under RL**; line closed, Phase 2
      moot. See `incoming/WAVE_ANALYSIS.md` §3.
- [x] **BiMamba seeds 2–3 + compose** — DONE 2026-07-19: parity SEED-ROBUST (multistart
      1.94 ± 0.09 vs attn 1.91; single 5.57 ± 0.11, 10x tighter than uni-hilbert's ±1.03).
      bimamba+random does NOT compose (7.61/3.30): random helps only causal scans —
      bidirectionality makes Hilbert locality exploitable again.
- [x] **500k extensions** — attention had NOT plateaued (2.45 single @500k); bimamba
      4.89/1.544; multistart parity holds at 500k (±0.05 noise band).
- [x] **Hybrid (4 BiMamba + 1 attention layer, −2.9% params)** — 250k: **4.01 single /
      1.542 multistart** — beats attention's multistart at HALF the budget; curve still
      falling. Sparse exact attention succeeds where pooled channels failed. Seed-1 only.
- [x] **Hybrid seeds 2–3 + 500k extension** — DONE 2026-07-20: 3-seed @250k
      **4.37 ± 0.33 / 1.63 ± 0.10** (every seed beats attention's matched-budget 1.91);
      @500k **3.01 / 1.016 vs attention 2.45 / 1.589** — hybrid BEATS attention multistart
      at equal budget (seed-1 vs seed-1, 11x noise band). WAVE_ANALYSIS.md Wave 3.
- [ ] **Fairness confirmations** — hybrid s2/s3 → 500k, attention s2/s3 @500k (launched
      2026-07-20). Then: N≥1000 scale probe (where O(N) actually pays) is the next gate.
- [x] **(C) Formal ablation** — mamba/random + mamba/sort seed-1 @250k COMPLETE
      (2026-07-01): **random 6.40% < sort 7.12% < hilbert 8.34%** — rule resolves
      inverted; Hilbert-locality premise refuted (§4b, `ablation/ABLATION.md`).
      Remaining: hilbert + random seeds 2–3 (queued 2026-07-17) to bound seed variance.
- [ ] Only if BiMamba lands in bucket (b)/(c): prototype + measure the hybrid thin-global-channel.
      NOT started — no hybrid or multi-curve arm exists yet (held by reviewer).

*Sources: `calibration/extension/{EXTENSION,ORDER_PROBE,DIAGNOSTICS,BIMAMBA}.md`, `GEOMETRY.md`.*
