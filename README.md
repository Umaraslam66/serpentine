# serpentine

**Can a Hilbert-ordered Mamba (state-space) encoder drive routing decisions as well as a
standard attention encoder on Euclidean TSP-100?**

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

## Status (Gate 0)

| stage | result |
|---|---|
| Baseline reproduction (seatbelt) | ✅ POMO TSP-100 reproduced: greedy 0.97 %, ×8-aug 0.149 % (vs published 1.07 % / 0.14 %) |
| Hilbert serialization unit tests | ✅ 7/7 (adjacency on 8×8–32×32, bijection, locality) |
| Mamba kernel mechanism gate | ✅ kernel vs pure-PyTorch max\|Δ\| 1.1e-5 < 1e-3 |
| Param match (attention vs mamba) | ✅ 1,269,760 vs 1,249,792 (1.57 %) |
| Calibration (60k steps, 1 seed) | 🟢 running — must show attention ≤ 2–3 % single-traj greedy before the ≥3-seed sweep |

Judged metric: **single-trajectory greedy gap**; PASS if mamba/hilbert ≤ attention + 1.0 %.
See `results/gate0/` for committed reports and `docs/` for the agent spec + HPC notes.

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
