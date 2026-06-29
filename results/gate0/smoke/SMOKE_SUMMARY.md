# Gate 0 — Candidate smoke (1 seed, 300 steps)

Both encoders trained from scratch, identical RL (POMO REINFORCE, Adam 1e-4/wd1e-6,
batch 64 × pomo 100), seed 1, matched params. Greedy gaps vs LKH on 1000 fixed instances.

## Unit tests (BEFORE training) — required gate
- Hilbert serialization: **7/7 PASS** (adjacency property on 8×8..32×32, bijection, locality, all orderings permutations).
- Encoder/torch: **6/6 PASS** incl. serialize→Mamba→un-serialize preserves node identity; param match.

## #params (matched)
- attention encoder = 1,269,760 total (encoder 1,187,712)
- mamba encoder    = 1,249,792 total (encoder 1,167,744)  → **1.57% diff**

## Smoke results @ 300 steps (FAR from converged — plumbing/throughput only)
| config | it/s | final loss | greedy single-traj | greedy multi-start |
|---|---|---|---|---|
| attention (—)      | ~5.3 (rising) | -3.25 | 30.8 % | 21.0 % |
| mamba / hilbert    | 0.70 | -1.67 | 38.7 % | 32.2 % |
| mamba / random     | 0.70 | -2.10 | 36.2 % | 22.1 % |
| mamba / sort       | 0.70 | -1.76 | 36.2 % | 29.5 % |

Ordering ablation NOT yet discriminative at 300 steps (expected); plumbing for all three orders works.

## Throughput / budget (the headline finding)
- attention ~5.3 it/s; **pure-PyTorch Mamba ~0.70 it/s → 7.6× slower** (sequential SSM scan: 10 blocks × 100-step python loop/forward).
- 3.5 h/seed ⇒ attention ~66,800 steps but mamba only ~8,800 steps.
- A shared step budget is bottlenecked to ~8,800 steps (mamba's 3.5 h) — likely too few for from-scratch attention to reach ≤2–3%.
- Pure-PyTorch mamba at a sane budget (~50k steps) ≈ **20 h/seed** → infeasible for ≥3 seeds × 4 configs.

## Recommendation
Install the official **mamba-ssm** (fast CUDA selective scan; this is what PointMamba/PTv3 use)
so mamba throughput ≈ attention, making a single ~50k-step / 3–4 h shared budget viable where
attention reaches a sane gap. Alternative: raise wall-clock budget (slow) or use a smaller-N proxy.
Awaiting reviewer ruling before full runs.
