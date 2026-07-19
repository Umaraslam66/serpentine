# Launch-wave results (submitted 2026-07-17, completed 2026-07-18)

All 7 jobs COMPLETED 0:0 (~14h each, A100). Artifacts in this directory; scored in-session
2026-07-18 against attention 2.95/1.91 and the seed-1 arms. Eval n=1000, opt_mean 7.749371521.

## Full table — single-traj gap % @250k (multistart in parens)

| arm | seed 1 | seed 2 | seed 3 | mean ± std |
|---|---|---|---|---|
| attention (ref) | 2.95 (1.91) | — | — | — |
| **bimamba/hilbert** | **5.55 (1.89)** | — | — | — |
| mamba/hilbert | 8.34 (4.46) | 6.36 (2.40) | 7.81 (4.06) | 7.50 ± 1.03 (3.64) |
| mamba/random | 6.40 (3.32) | 6.84 (3.13) | 7.45 (3.40) | 6.90 ± 0.53 (3.28) |
| mamba/sort | 7.12 (3.64) | — | — | — |
| gc-mean (Variant A) | 7.86 (4.13) | — | — | — |
| gc-segment (Variant B) | 8.38 (4.51) | — | — | — |

## 1. Ordering: the seed-1 inversion is NOT seed-robust — downgrade to the weak claim

Hilbert seeds span **6.36–8.34** (2.0pp); seed 2 (6.36/2.40) beats every random seed at
every common checkpoint. Random spans 6.40–7.45 (0.9pp). Means favor random
(6.90 vs 7.50) and random's variance is half of hilbert's — consistent with the
invariance-regularizer reading — but with n=3 and overlapping ranges the ABLATION.md §1
claim "random < hilbert at every eval, whole-curve" is a **seed-1 artifact** and must not
be stated unqualified. Supported claim: **the Hilbert locality prior confers no measurable
benefit over random ordering and roughly doubles run-to-run variance.** (This is closer to
Point Cloud Mamba's "simple ≈ curve orders" than to a strict inversion; the pre-registered
expectation Hilbert ≥ sort > random still finds no support.)

## 2. BiMamba: bucket (b), with an unregistered twist — multistart PARITY with attention

5 bidirectional layers, 1,248,512 params (−0.10% vs uni), 4.78 it/s (−5%).
Single-traj 5.55 closes **52%** of the seed-1 gap (8.34→2.95), 43% against the hilbert
seed-mean (7.50) — a partial close, i.e. **bucket (b)** of the FINDINGS §5 rule:
directionality helps substantially AND a wall remains. The twist: **multistart 1.89 ≈
attention 1.91** — under POMO multi-start decoding the bidirectional encoder supports
tours as good as attention's, at O(N). The residual deficit is concentrated in
single-trajectory greedy decoding, not in what the representation can support.
(Caveat: seed 1 only; given §1's variance lesson, parity needs seeds 2–3 before being
stated as a finding.)

## 3. Global channels: both variants FAIL under RL

Variant A (mean-pool, +1.32% params): 7.86 — a −0.48pp move that is *within hilbert's
seed noise* (σ≈1.0). Variant B (structured segment, +2.62% params): 8.38 — indistinguishable
from baseline 8.34, and above it for most of the curve. The GLOBAL_CHANNEL.md hypothesis
("B closes most, A partial → structured channel is the contribution") is **refuted**: the
receptive-field deficit is not repaired by cheap global summaries under RL, while
bidirectionality (§2) attacks the same diagnostic and works. Per the pre-registered
decision rule, the SFT+DPO Phase 2 is moot; the global-channel line is closed.

## 4. Revised mechanistic story (one paragraph)

The vanilla causal scan fails for two separable reasons. The dominant, fixable one is
**directionality** — a node must see both curve directions; fixing it recovers half the
greedy gap and (seed-1) all of the multistart gap at matched params. The rest is not
reachable by bolting global context onto a causal scan (both channel variants ≈ 0), and
serialization choice is second-order noise (hilbert ≈ random within seed variance, hilbert
noisier). Gate-0 verdict unchanged — attention still wins single-traj at N=100 — but the
surviving research object is the **bidirectional-Mamba multistart-parity result**, pending
seed confirmation.

## Next (cheap, decisive)

1. BiMamba seeds 2–3 (~29h GPU): is multistart parity seed-robust?
2. BiMamba + random ordering (1 run): do the two surviving positive effects compose?
3. Only if 1–2 hold: scale probe at N≥1000 where O(N) vs O(N²) actually matters.

---

# Wave 2 results (scored 2026-07-19): parity is seed-robust; the hybrid is the new headline

All jobs COMPLETED 0:0. Single-traj (multistart) @ final step:

| arm | steps | params | single | multi |
|---|---|---|---|---|
| attention s1 (extended) | 500k | 1,269,760 | **2.454** | 1.589 |
| hybrid (4 bi + 1 attn) s1 | 250k | 1,213,184 | **4.014** | **1.542** |
| bimamba/hilbert s1 (extended) | 500k | 1,248,512 | 4.893 | 1.544 |
| bimamba/hilbert s1/s2/s3 | 250k | 1,248,512 | 5.55/5.69/5.47 → **5.57 ± 0.11** | 1.89/2.04/1.89 → **1.94 ± 0.09** |
| bimamba/random s1 | 250k | 1,248,512 | 7.612 | 3.298 |

**1. Multistart parity is SEED-ROBUST.** BiMamba 3-seed multistart 1.94 ± 0.09 vs attention
2.95-budget-matched 1.91 — and BiMamba's seed variance (±0.11 single-traj) is an order of
magnitude tighter than uni-mamba/hilbert's (±1.03). Bidirectionality stabilizes training.

**2. BiMamba + random does NOT compose — the ordering story flips with direction.** 7.61/3.30,
far worse than bimamba/hilbert 5.57. Random ordering (weakly) helped the *unidirectional*
scan; it clearly hurts the *bidirectional* one. Mechanistically consistent: with both scan
directions available, Hilbert locality becomes exploitable structure; random destroys it.
The uni-scan "locality prior buys nothing" claim (§1 above) is now scoped: **it holds only
for causal scans.** Consequence: the order-ensembling trick (bestofk, uni: 3.33→2.72@k=8)
loses its path to the headline — its natural host (a strong random-trained model) does not
exist in the bidirectional regime.

**3. 500k extensions: budget was PART of the story, for both sides.** Attention had not
plateaued (2.95→2.45 single); bimamba 5.55→4.89 with a flattening tail. At 500k the
multistart figures (bim 1.544 / attn 1.589) sit inside a ±0.05 eval-noise band — parity,
not a win. Single-traj gap at 500k: 2.44pp, essentially the 250k gap shifted down.

**4. The HYBRID (one POMO attention layer inside a 4-layer BiMamba stack, −2.9% params) is
the result.** At 250k it posts 4.01/1.542: single-traj 1.5pp better than bimamba@250k
(59% of the bimamba→attention deficit closed), and multistart 1.542 — better than
attention at DOUBLE the budget (1.589@500k, within noise → parity) and clearly better than
attention's budget-matched 1.91. Its curve is still falling steeply at 250k (−0.13pp/10k).
Where ten pooled-summary channels failed, ONE exact-attention layer succeeds: routing
under RL needs sparse exact all-pairs context, not global averages, over a linear backbone.

**Caveats:** hybrid is seed-1 only (seeds 2–3 + 500k extension launched 2026-07-19);
multistart differences below ~0.05 are eval noise; all N=100.
