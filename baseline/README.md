# Gate 0 — POMO TSP-100 baseline reproduction

Reproduce the published POMO Euclidean TSP-100 optimality gap **before** building the
Mamba candidate. This is the seatbelt: if the baseline does not reproduce, the
Hilbert+Mamba comparison is meaningless.

## Target (reviewer-set)
| metric | published | pass condition |
|---|---|---|
| greedy (POMO multi-start, no aug) | **1.07 %** | achieved greedy gap ≤ 1.07 % + **0.30 %** abs |
| ×8 augmentation | **0.14 %** | (reported; primary gate is greedy) |

## Method (no edits to upstream POMO)
- **Baseline repo:** `yd-kwon/POMO` (NeurIPS 2020), pinned at commit `d7c3d6e`.
  Pretrained TSP-100 weights: `result/saved_tsp100_model2_longTrain/checkpoint-3100.pt`.
- **Model = decoder + encoder we will later swap:** 6-layer attention encoder,
  embedding 128, 8 heads, pomo size 100, greedy (`argmax`) decode. Params reported by the run.
- **Fixed test set:** `gen_testset.py` reproduces POMO's exact generator
  (`torch.rand((N,100,2))`, Uniform[0,1]²) under `seed=1234`, N=10000 — saved once and
  fed to *both* the model and the oracle (same instances → no sampling noise in the gap).
- **Oracle:** LKH-3 (`lkh_solve.py`), coords scaled ×1e6 → integer EUC_2D, RUNS=1.
- **Gap:** `compute_gap.py` → `100·(mean(model)/mean(opt) − 1)` (POMO/Kool convention),
  plus per-instance mean gap. A single aug=8 pass yields both no-aug and ×8-aug lengths.
- **Injection:** `eval_pomo.py` monkeypatches `TSPEnv.get_random_problems` to serve the
  fixed instances sequentially; POMO source files are untouched.

## Reproduce (Leonardo)
```bash
# one-time setup on a LOGIN node (internet)
bash baseline/env_setup.sh      # pinned venv: torch==2.2.2, numpy==1.26.4
bash baseline/build_lkh.sh      # builds tools/lkh

# full run (login node submits the Slurm DAG)
bash baseline/run_baseline.sh           # N=10000
# quick smoke:
N=128 bash baseline/run_baseline.sh
```
Outputs: `results/baseline_report_n<N>.json` (gaps, wall-clock, GPU mem, #params).

## Slurm
- eval: `boost_usr_prod` / `boost_qos_dbg`, 1 GPU, ~minutes.
- lkh: `lrd_all_serial` (budget-free), 8 cores.
- gap: `lrd_all_serial`, depends `afterok` on the two above.
- account `AIFAC_P02_548`, all under `/leonardo_work/AIFAC_P02_548/mamba-route`.

## Seatbelt
If achieved greedy gap > 1.07 % + 0.30 % abs → **STOP and report** (do not build candidate).
