# Gate 0 — Findings (mechanistic story so far)

**Question:** can a Hilbert-ordered Mamba encoder drive routing decisions as well as a
standard attention encoder on Euclidean TSP-100? **Provisional answer: NO for the
encoder-swap as specified** (vanilla causal Mamba, single Hilbert curve). This document is
the running synthesis; the BiMamba result (tonight) and the formal ablation slot into the
clearly-marked TODOs below.

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
  fixed-window scanner — so the limitation *worsens* with scale, which is where the thesis lives.

## 5. The fork this sets up (BiMamba discriminator — TODO B)

Param-matched bidirectional Mamba (forward+backward scan, 5 layers = 1.2485M ≈ uni 1.2498M),
everything else identical, trained from scratch 250k/seed-1. Reading vs the uni 8.34% / attn 2.95%:

| BiMamba @ 250k | reading | next direction |
|---|---|---|
| closes most of the ~5.4% gap | failure was **CAUSALITY** — shallow, fixable, O(N) preserved | bidirectionality is the fix |
| barely moves | **FIXED-STATE CAPACITY** — deep | **hybrid**: Mamba locally + thin global channel over Hilbert-segment summaries (motivated by Task 1; needed more at scale by Task 2) |

---

## TODO — pending numbers (do not treat conclusions as final until filled)

- [ ] **(B) BiMamba seed-1 @ 250k** single-traj/multi-start curve — staged
      (`scripts/run_bimamba.sh`), runs on next GPU window. Decides the §5 fork.
- [ ] **(C) Formal ablation** — from-scratch mamba/random + mamba/sort (seed 1) and
      mamba/hilbert seeds 2–3 @ 250k (jobs queued behind GPU maintenance). Formalizes the
      KILL: PASS = hilbert ≤ attn+1.0% across ≥3 seeds; KILL = worse by >1.0% AND hilbert
      clearly beats random. (Inference proxy already shows hilbert ≪ sort ≪ random.)
- [ ] If BiMamba lands in "capacity" branch: prototype + measure the hybrid thin-global-channel.

*Sources: `calibration/extension/{EXTENSION,ORDER_PROBE,DIAGNOSTICS,BIMAMBA}.md`, `GEOMETRY.md`.*
