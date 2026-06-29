# Gate 0 — Baseline reproduction result (POMO TSP-100)

**Seatbelt: PASS.** Reproduced POMO's published TSP-100 gaps within tolerance.

- Repo: `yd-kwon/POMO` @ `d7c3d6e`, pretrained checkpoints, upstream untouched.
- Test set: 10,000 instances, Uniform[0,1]², seed 1234 (POMO's exact generator).
- Oracle: LKH-3.0.13, RUNS=1, coords ×1e6 EUC_2D. **mean optimal = 7.76456**
  (matches the canonical TSP-100 LKH value ~7.765 → oracle validated).
- Gap = mean(model)/mean(opt) − 1 (POMO/Kool convention). Env: torch 2.2.2+cu121, A100-64GB.
- #params (encoder+decoder) = **1,269,760**.

| checkpoint | greedy single-traj | ×8-aug | multi-start no-aug | eval wall | GPU mem |
|---|---|---|---|---|---|
| **standard** (ep 2000) | **0.97 %** | **0.149 %** | 0.41 % | ~50 s | 6.45 GB |
| longtrain (ep 3100) | 0.87 % | 0.125 % | 0.36 % | ~45 s | 6.45 GB |

Targets: greedy **1.07 %**, ×8-aug **0.14 %**. Pass = greedy ≤ 1.07 % + 0.30 % abs (≤ 1.37 %).
Standard checkpoint = closest match (greedy −0.10 % of target; aug essentially exact).

Reproduce: `bash baseline/env_setup.sh && bash baseline/build_lkh.sh && bash baseline/run_baseline.sh`
