# Gate 0 — Calibration (1 seed, 60k steps from scratch)

Both encoders trained from scratch, identical RL (POMO REINFORCE, Adam 1e-4/wd 1e-6,
batch 64 × pomo 100), seed 1, identical 60,000-step budget, same decoder. Greedy gaps vs
LKH on 1000 fixed held-out instances (opt mean 7.7494). Judged metric = single-traj greedy.

| config | single-traj greedy | multi-start greedy | it/s | wall | GPU peak | #params |
|---|---|---|---|---|---|---|
| attention (unserialized) | **6.17 %** | 2.85 % | 5.19 | 3.21 h | 5.10 GB | 1,269,760 |
| mamba / hilbert | 10.12 % | 5.54 % | 4.78 | 3.49 h | 5.49 GB | 1,249,792 |

Train tour-length trajectory (best-of-pomo, sampled): attention 8.20→7.97 (plateau),
mamba 8.51→8.33→8.23 (still descending) — Hilbert+Mamba **is** learning.

**SEATBELT: FAIL.** From-scratch attention single-traj greedy = 6.17 % > ~2–3 % at 60k.
Per the locked rule, do NOT run the >=3-seed sweep against an under-trained baseline.
Single-trajectory greedy converges far slower than multi-start (best-of-100); attention's
multi-start gap (2.85 %) is near-sane, but the deciding single-traj metric is not.

Recommendation: raise the shared budget and re-run the 1-seed calibration to find where
attention single-traj crosses ~3 %, before any sweep. Reviewer to rule.
