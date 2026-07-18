# Gate 0 — Formal ordering ablation (seed 1, 250k steps)

> **SEED UPDATE 2026-07-18 — the §1 inversion is NOT seed-robust.** Seeds 2–3
> (`../incoming/WAVE_ANALYSIS.md`): hilbert 6.36/7.81, random 6.84/7.45 — hilbert s2 beats
> every random seed. 3-seed means: random 6.90 ± 0.53, hilbert 7.50 ± 1.03. The supported
> claim is the weak form — **Hilbert confers no benefit and doubles seed variance** — not
> "random beats hilbert whole-curve", which was a seed-1 artifact. §2's invariance-
> regularizer reading survives only as an interpretation of the variance gap. Read §§1–3
> below with this correction in mind.

**Jobs:** `serp-abl-rand` 48132839, `serp-abl-sort` 48132840 (Leonardo, A100). Submitted
2026-06-30, held by the `maint_3006_boost` reservation, auto-started 18:56, completed
2026-07-01 08:44/08:45 (~13.8 h each, 5.03–5.04 it/s). Results retrieved 2026-07-17
(collection lapsed for 16 days — see §4).

Identical to the mamba/hilbert calibration run in every respect except `--order`:
same encoder (1,249,792 params), POMO decoder, REINFORCE budget (250k × bs 64), seed 1,
eval set (n=1000, LKH opt_mean 7.749371521), mamba-ssm kernel.

## 1. Result: the ordering hierarchy INVERTS when trained from scratch

| order (trained) | single-traj gap @250k | multi-start gap @250k |
|---|---|---|
| **random** | **6.40%** | **3.32%** |
| sort (lexicographic) | 7.12% | 3.64% |
| hilbert | 8.34% | 4.46% |
| *(attention/hilbert baseline)* | *2.95%* | *1.91%* |

Not an endpoint artifact — **random < sort < hilbert (lower = better) at every 20k-step
eval from the first common checkpoint**:

| step | random | sort | hilbert |
|---|---|---|---|
| 60k | 8.48 | 9.79 | 10.12 |
| 100k | 7.64 | 8.85 | 9.39 |
| 140k | 7.06 | 8.27 | 8.98 |
| 180k | 6.74 | 7.89 | 8.74 |
| 220k | 6.59 | 7.63 | 8.52 |
| 250k | 6.40 | 7.12 | 8.34 |

## 2. Reading against the pre-registered decision rule

TODO C's rule was: *KILL = hilbert worse than attention by >1.0% AND hilbert clearly beats
random*. The first clause holds (+5.4% abs); the second **fails in the inverted
direction** — hilbert is 1.9% abs *worse* than random. So the outcome is not the
anticipated "capacity KILL with ordering vindicated"; it is a **refutation of the
Hilbert-locality premise itself**: spatial-locality serialization does not help this
encoder route — it hurts.

Reconciliation with the inference-time order probe (hilbert-trained ckpt: hilbert 8.38 ≪
sort 33.49 ≪ random 56.38): no contradiction. The probe shows a model *trained on* one
order depends on that order at inference; it says nothing about which order is the better
*training* signal. Trained from scratch, random ordering (fresh permutation per instance)
acts as an invariance regularizer — the encoder cannot exploit sequence-local structure,
so it is forced toward order-robust, more global per-node features. Hilbert's locality is
a crutch: it lets the short (~20-position) causal window fit *local* structure well while
leaving the scattered ~14% of spatial neighbours (GEOMETRY.md) permanently out of reach.

## 3. Status of the claims this changes

- **Encoder-swap KILL** stands and broadens: *every* vanilla-Mamba ordering (6.40–8.34%)
  loses to attention (2.95%) at matched params/budget under RL. Best mamba variant is now
  random-order at 6.40%, cutting the headline gap from 5.4% to 3.5% abs.
- **"Hilbert + Mamba" as the candidate architecture is dead** as specified: the
  serialization that names the project is the worst of the three tested.
- **The receptive-field diagnosis (FINDINGS §4) survives** — attention still wins by a
  wide margin, consistent with a global-context deficit — but the *mechanistic story must
  shift* from "curve boundaries scatter neighbours" to "any fixed-window causal scan
  under-integrates global context; locality priors make it worse, invariance pressure
  makes it better".
- **Caveat:** single seed per arm. The whole-curve uniform separation (§1) makes a
  seed-noise inversion unlikely but not impossible; seeds 2–3 for hilbert and random are
  queued (2026-07-17) to bound variance before any external claim.

## 4. Process post-mortem (why this sat 16 days)

The Jun-30 session ended holding for a reviewer ruling "tomorrow ~07:00"; the queued
ablation auto-ran when the maintenance reservation cleared, but no follow-up session
occurred: results were never fetched and the two manually-gated launchers
(`run_bimamba.sh`, `run_global_channel.sh`) were never invoked. Fix applied 2026-07-17:
results retrieved and committed, staged experiments smoke-tested and launched, and result
collection scheduled rather than left to a manual return.

*Sources: `calib_mamba_{random,sort}_s1.{json,curve.jsonl}`, `abl_ids.txt`, job logs
`logs/serp-abl-{rand,sort}-481328{39,40}.out` on Leonardo `$WORK/mamba-route`.*
