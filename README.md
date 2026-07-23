# serpentine — When Mamba Needs Attention

**Can a Hilbert-ordered Mamba (state-space) encoder drive routing decisions as well as a
standard attention encoder on Euclidean TSP-100?**

📄 **Paper:** *When Mamba Needs Attention* (Umar Aslam, 2026) — LaTeX source and PDF in
[`paper/`](paper/). **Answer: not alone — but a hybrid with ONE attention layer beats the
attention baseline on multistart tour quality at 4.5 % fewer parameters.**

A space-filling-curve serialization turns an unordered set of cities into a 1-D sequence; a
Mamba encoder then reads that sequence. The representation trick is already validated in 3-D
point-cloud learning (PointMamba, Point Transformer V3). The open question — **Gate 0** — is
whether that encoder transfers to routing *decisions*: sequencing nodes into a tour under a
construction + RL setup, matching attention at N=100. A clean negative is a useful result.

## Approach

- **Baseline:** POMO (Kwon et al., NeurIPS 2020) — 6-layer attention encoder, pointer decoder,
  100-start REINFORCE. Kept pristine (pinned `yd-kwon/POMO @ d7c3d6e`).
- **Candidate:** swap **only the encoder** for a Hilbert-serialized Mamba stack; identical
  decoder, RL, seeds, and matched parameters. Order ablation: hilbert / zorder / sort / random.
- **Oracle:** LKH-3 for optimality gaps on a fixed, seeded TSP-100 test set.
- **Discipline:** Hilbert ordering is unit-tested; the mamba-ssm CUDA kernel is numerically
  gated against a pure-PyTorch reference (`< 1e-3`) before any training.

## Results (Gate 0 — CLOSED 2026-07-21)

Optimality gap vs LKH-3 on 1000 held-out TSP-100 instances, single-trajectory (multistart),
identical POMO REINFORCE + decoder, matched params. Full scoring:
[`results/gate0/incoming/WAVE_ANALYSIS.md`](results/gate0/incoming/WAVE_ANALYSIS.md).

| encoder | @250k, 3 seeds | @500k converged, 3 seeds |
|---|---|---|
| Attention (baseline) | 2.95 (1.91) — s1 | 2.52 ± 0.07 (1.61 ± 0.02) |
| Uni-Mamba / Hilbert (original candidate) | 7.50 ± 1.03 (3.64) — **killed** | — |
| BiMamba / Hilbert | 5.57 ± 0.11 (1.94 ± 0.09 — parity) | 4.89 (1.54) — s1 |
| **Hybrid: 4 BiMamba + 1 attention layer** | 4.37 ± 0.33 (**1.63 ± 0.10**) | 3.08 ± 0.06 (**1.14 ± 0.19**) |

Key findings: the vanilla Hilbert+Mamba recipe fails for a diagnosable reason (short causal
receptive field); pooled global channels don't fix it; bidirectionality reaches multistart
parity and cuts seed variance 10×; **one exact attention layer beats the full attention
encoder on multistart** (worst hybrid seed 1.365 < best attention seed 1.589). A locality
prior helps only non-causal scans. All at N=100; the O(N) payoff at N≥1000 is future work.
Pre-registered kill/decision rules: `docs/gate0_agent_spec.md` + `results/gate0/`. The
earlier seatbelt still holds: official POMO checkpoint reproduced at 0.97 % greedy /
0.149 % ×8-aug on our oracle before any experiment.

## Layout

```
src/serpentine/
  serialization/   hilbert.py, zorder.py   (+ random/sort)  — unit-tested numpy core
  encoders/        attention.py (POMO), mamba.py (Mamba-1 block + Hilbert encoder)
  decoder.py       POMO pointer decoder (shared, unchanged)
  model.py         build_model — swaps ONLY the encoder
  rl.py            from-scratch POMO REINFORCE (encoder-pluggable, checkpoint/resume)
  oracle.py        LKH-3 driver
  eval.py / metrics.py / data.py
scripts/           env_setup.sh, build_lkh.sh, install_mamba.sh, reproduce_baseline.sh,
                   run_sweep.sh, slurm/*.sbatch
tests/             test_hilbert.py, test_zorder.py, test_encoders.py, test_kernel_parity.py
docs/  configs/  results/gate0/
```

## Reproduce

POMO is an external dependency; LKH-3 and the Mamba kernel compile from source.

```bash
# one-time (needs internet + CUDA toolkit; on an HPC login node)
bash scripts/env_setup.sh        # venv (torch 2.2.2+cu121, numpy 1.26.4), package, clone POMO@d7c3d6e
bash scripts/build_lkh.sh        # LKH-3 oracle
bash scripts/install_mamba.sh    # mamba-ssm + causal-conv1d (CUDA kernel)

export SLURM_ACCOUNT=<your-allocation>

# Gate-0 baseline seatbelt (Slurm DAG: LKH oracle + eval -> gap report)
bash scripts/reproduce_baseline.sh

# unit tests (CPU)
PYTHONPATH=src python tests/test_hilbert.py
PYTHONPATH=src python tests/test_zorder.py
# kernel mechanism gate (GPU): sbatch --account=$SLURM_ACCOUNT scripts/slurm/kernel_gate.sbatch
```

Paths are deployment-agnostic: set `POMO_ROOT` and `SERPENTINE_WORK` to override the POMO
clone and the data/results location (both default to the repo root).

## License & citation

MIT (`LICENSE`). If you use this, please cite via `CITATION.cff`. POMO and LKH-3 retain their
own licenses.
