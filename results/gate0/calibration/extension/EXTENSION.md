# Gate 0 — Bounded calibration extension (seed 1)

Reviewer-authorized bounded extension of the failed 60k seatbelt. **No sweep, no
goalpost move.** Both seed-1 runs were *resumed* from their existing 60k checkpoints
(not restarted/deleted) and extended to a hard cap of **250k steps OR 12h wall/config,
whichever first**, identical for both. Single-traj greedy (+ multi-start) evaluated on
the fixed 1000 held-out set every 20k steps.

## Run outcome

| config | end reason | final step | capped | wall elapsed | it/s | GPU-h | GPU peak | params |
|---|---|---|---|---|---|---|---|---|
| attention/hilbert (48097768) | 250k reached | 250000 | no | 10:07:45 | 5.219 | 10.112 | 5.10 GB | 1,269,760 |
| mamba/hilbert  (48097771) | 250k reached | 250000 | no | 10:56:26 | 4.832 | 10.923 | 5.48 GB | 1,249,792 |

Both hit the **250k step cap before the 12h wall** (`capped: false`). They therefore
share the full 20k grid — **largest common step = 250,000**, no asymmetry to reconcile.
Topology unchanged from the 60k phase: 1 GPU/job, single-process (not DDP), batch 64×100,
Adam lr 1e-4, no warmup/scheduler. Co-scheduled on one node (1 A100-64GB each).
Total compute: **21.04 GPU-hours** (2 GPUs of a 4-GPU node, ~10.5h wall).

## Curves — single-trajectory greedy gap % (the judged metric)

| step | examples seen | attention | mamba/hilbert |
|---|---|---|---|
| 60000  |  3.84M | 6.169 | 10.116 |
| 80000  |  5.12M | 5.499 |  9.724 |
| 100000 |  6.40M | 4.510 |  9.391 |
| 120000 |  7.68M | 3.896 |  9.170 |
| 140000 |  8.96M | 3.659 |  8.976 |
| 160000 | 10.24M | 3.438 |  8.878 |
| 180000 | 11.52M | 3.377 |  8.737 |
| 200000 | 12.80M | 3.283 |  8.554 |
| 220000 | 14.08M | 3.134 |  8.518 |
| 240000 | 15.36M | **2.970** |  8.425 |
| 250000 | 16.00M | **2.953** |  8.336 |

## Curves — multi-start greedy gap % (reference only, NOT the judged metric)

| step | attention | mamba/hilbert |
|---|---|---|
| 60000  | 2.850 | 5.542 |
| 100000 | 2.530 | 5.026 |
| 150000 (interp) | — | — |
| 200000 | 2.120 | 4.554 |
| 250000 | **1.906** | 4.457 |

## Findings

1. **Attention crosses ~3% single-traj between 220k and 240k.** First sub-3% reading
   is **step 240,000 = 15.36M examples seen** (2.970%); it settles at 2.953% by 250k.
   That is **4× the original 60k budget** to clear the seatbelt — and only barely:
   the single-traj curve is flattening at ~2.95%, not still dropping fast.

2. **At the largest common step (250k):**
   - attention/hilbert single-traj **2.953%**, multi-start 1.906%
   - mamba/hilbert  single-traj **8.336%**, multi-start 4.457%
   - **Δ single-traj = +5.38% abs worse for mamba** — far outside the +1.0% band, at every step.

3. **Mamba is not closing the gap.** Over 190k extra steps it moved 10.12% → 8.34%
   single-traj (−1.78 abs across the whole extension) while attention moved
   6.17% → 2.95% (−3.22 abs). The two curves diverge, not converge.

4. **Harness/metric note (carried forward):** single-traj greedy is a deliberately harsh
   readout for POMO-trained models (training optimizes a 100-start shared-baseline
   objective). Multi-start tells the same *ordering* (attention 1.91% < mamba 4.46%) but at
   roughly half the absolute gap. The seatbelt is now met by attention, so the comparison
   is valid — but note it took 240k steps to get there.

5. **Baseline integrity (carried forward):** attention is fed raw unordered nodes — the
   "hilbert" tag is cosmetic (verified earlier: 0 vs 2 `serialize_order` calls). Identical
   seed/optimizer/decoder/budget between the two configs.

**Single-seed signal: mamba/hilbert is decisively worse (≈5.4% abs single-traj, ≈2.5% abs
multi-start) and not catching up.** Formal KILL still requires the ≥3-seed sweep with the
random-order control — held pending reviewer ruling.
