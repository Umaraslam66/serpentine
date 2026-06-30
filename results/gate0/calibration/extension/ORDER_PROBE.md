# Inference-time order-sensitivity probe (CPU, no retraining)

Cheap proxy run while the from-scratch random/sort ablation is queued behind GPU
maintenance. Loads the trained **mamba/hilbert seed-1** checkpoint (step 250000) and
evaluates single-traj greedy gap on a fixed **200**-instance held-out subset under three
INPUT orderings, swapping only `encoder.order_mode` — **weights untouched, no retraining**.
Pure-PyTorch Mamba path on CPU login node (kernel↔pure parity gate <1e-3 makes this
faithful to the kernel-trained weights). Runtime: 31s. Script: `scripts/order_probe.py`.

## Result

| input order | single-traj gap % | multi-start gap % |
|---|---|---|
| **hilbert** (matches training) | **8.376** | 4.555 |
| **sort** | 33.487 | 22.063 |
| **random** | 56.377 | 39.739 |

**Harness sanity check:** hilbert on this 200-subset = 8.376% vs the full 1000-set 250k
curve value 8.336% → Δ 0.04%. The probe reproduces the trained metric exactly, confirming
the CPU pure-PyTorch path and the checkpoint load are correct.

## Interpretation

**Order propagates strongly to the decision — this is a GENUINE-CAPACITY signal, NOT a bug.**
Feeding the hilbert-trained model a random ordering at inference collapses it from 8.4% to
**56.4%** single-traj (random) and 33.5% (sort). If the serialization were *not* reaching the
decoder, swapping the order would leave the gap unchanged (~8.4% for all three). It does not:
the gap moves by **+48 abs** under random. The serialize → Mamba scan → un-serialize → decode
path is intact and the model is highly order-dependent. Ordering monotonicity matches
expectation: **hilbert ≪ sort ≪ random** (in badness).

## Caveats (what this does and does NOT show)

- This is a **proxy**, not the formal control. It tests a *train/test order mismatch* on one
  fixed set of weights — it proves order **reaches** the decoder (rules out the bug hypothesis).
- It does **not** by itself prove the from-scratch claim "a model TRAINED on hilbert beats one
  TRAINED on random." A network trained from scratch on random ordering could in principle
  learn to cope. That capacity question is exactly what the queued from-scratch
  mamba/random + mamba/sort ablation (250k, seed 1) will answer.

**Bottom line for the KILL decision:** the bug hypothesis (ordering not propagating) is
**refuted** by this probe. The genuine-capacity reading is supported. Formal confirmation
still pending the from-scratch ablation.
