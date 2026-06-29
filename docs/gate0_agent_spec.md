# GATE 0 — AGENT SPEC (Claude Code)
### Project: State-Space (Mamba) Solver for Large-Scale Routing — Gate 0

## Your role
You are the **implementing agent**. You build, test, run, and report. You do **not** decide go/no-go and you do **not** proceed past the gate on your own. A **senior reviewer** (separate chat, briefed by `GATE0_REVIEWER_BRIEF.md`) rules via the human. Your reports go to that reviewer; you then **WAIT** for the reviewer's ruling, which the human pastes back to you.

**The loop:** you act → report → reviewer decides → you follow the next instruction.

## Prime directive: FAIL FAST
The goal is **not** to build the solver. It is to answer one question as cheaply as possible:

> *Can a Hilbert-ordered Mamba encoder drive routing decisions as well as a standard attention encoder on Euclidean TSP at N=100?*

Build the minimum to get that signal. **No** foundation-model machinery, **no** large runs, **no** gold-plating. Reuse existing code. Cap wall-clock.

## You are NOT starting cold — build on these
The core trick (space-filling-curve serialization + Mamba over unordered spatial points) is **already proven in 3D point-cloud learning**. Reuse it; do not reinvent.
- **Encoder + serialization:** **PointMamba** (NeurIPS 2024) and **Point Transformer V3** (CVPR 2024). Lift their **Hilbert / Z-order space-filling-curve serialization** of an unordered point set into a 1-D sequence. This is the validated recipe; port it, don't redesign it.
- **Attention baseline (Gate-0 reference):** an existing **POMO** or **Kool Attention Model** implementation.
- **Oracle (for optimality gap):** **LKH-3** (or Concorde).
- **Later stages ONLY (NOT Gate 0):** GLOP / H-TSP (decomposition for large-N); the Amazon Last-Mile dataset (real-world eval). Ignore these for now.

## What Gate 0 *actually* tests — read carefully
Representing scattered spatial points with Hilbert+Mamba is **already validated** in point clouds, so that is **not** the open question. Gate 0's real bar:

> **Does that encoder drive competitive routing _decisions_** — sequencing nodes into a tour under the construction + RL setting — matching attention at N=100?

If it fails, the break is in the **decision transfer**, not the representation. **Log that explicitly** — a clean, well-isolated negative is itself a useful finding, not a wasted run.

## The experiment
- **Problem:** Euclidean TSP, N=100, nodes ~ Uniform[0,1]^2. Fixed seed.
- **Reference (baseline):** known-good attention model (POMO / Kool AM) — reuse a repo. **FIRST reproduce its published TSP-100 optimality gap.** Seatbelt: if you cannot reproduce it, **STOP and report** — the comparison is meaningless otherwise.
- **Candidate:** replace **only the encoder** with a **Mamba** stack over **Hilbert-serialized** nodes (lift PointMamba/PTv3 serialization). Keep the **same decoder** and **same RL training** as the baseline. Match parameter/compute budget.
- **Ordering ablation (required, cheap):** Mamba encoder with (a) Hilbert order, (b) random order, (c) coordinate-sort. Proves the result is about locality, not luck. Expect Hilbert >= sort > random; if random ~ Hilbert, the order isn't propagating — **flag it**.
- **Training:** same RL for both (POMO or REINFORCE + greedy-rollout baseline), same gradient steps / same wall-clock cap (target a few hours on **1x A100**). Fixed seed, reproducible, checkpointed.
- **Eval:** optimality gap vs LKH-3 (or Concorde) on a fixed held-out set of TSP-100 instances. Report greedy gap, augmented/sampling gap, wall-clock, GPU memory.

## SFC serialization — known gotchas (borrowed, so you skip the pain)
- **Grid size matters.** Curve serialization is sensitive to the grid/quantization size — choose sensibly and record it.
- **Single curve first.** Start with ONE Hilbert curve for the minimal test. Multi-curve (Hilbert + Trans-Hilbert) can capture different spatial views but **lengthens the sequence and adds redundancy** — only add if Gate 0 is borderline.
- **Order indicator.** If you use >1 curve, tag each token with which curve produced it.
- **Direction.** A single SSM scan is causal/one-directional; a cheap bidirectional scan is a fallback if one-directional underperforms.

## Proposed PASS/KILL — the reviewer LOCKS the threshold (do not assume it)
- Baseline reproduced within tolerance → else **HALT**.
- **PASS:** candidate gap <= baseline gap + delta (delta ~1–2% abs, **reviewer-set**), across >=3 seeds.
- **KILL:** candidate clearly worse by > delta **AND** Hilbert clearly beats random (genuine gap, not a bug).

## Engineering rules (non-negotiable)
- **Verify the mechanism, not the metric.** A loss going down is not success. Assert: both models share decoder + budget; Hilbert ordering is correct (**unit-test it**); the baseline reproduces known numbers **before** you trust any comparison.
- **Reproducibility:** fixed seeds, pinned deps, one command to reproduce, commit configs + results. **Commit method/scripts BEFORE** producing the numbers they judge.
- **Seatbelts that abort early:** baseline-not-reproduced → halt; ordering unit-test fails → halt; obvious degeneracy → halt.
- **Report at the gate. Do NOT start Gate 1.**

## Leonardo HPC — READ THIS FIRST
In your working directory there is a doc explaining how to use the **EuroHPC Leonardo** cluster (GPU + socket/login) — find it and follow it. **Hard rules:**
- The **human** performs login and opens the socket. **You do not attempt the login yourself.**
- You work **only** inside the dedicated project directory.
- You **prepare, submit, and monitor** GPU jobs via `sbatch` yourself, **safely**: `sbatch`/`squeue`/`sacct`/`scancel` on **your own jobs only**. **No interactive GPU runs.** **Never delete or modify other users' work, or anything outside the project directory.**
- Delete stale checkpoints before a fresh run; a resubmit should resume cleanly.

## What to report back to the reviewer (structured)
1. **Baseline reproduction:** target gap vs achieved gap (seatbelt PASS/FAIL).
2. **Candidate vs baseline:** optimality gaps (greedy + augmented), >=3 seeds.
3. **Ordering ablation:** Hilbert vs random vs sort.
4. **Cost:** wall-clock, GPU memory, # params each.
5. **Localization:** if it failed, does the break look like representation or decision-transfer?
6. **Anything else:** surprises, seatbelt trips, honest caveats.

Then **STOP and wait** for the reviewer's ruling.
