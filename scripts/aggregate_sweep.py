#!/usr/bin/env python3
"""Aggregate the >=3-seed sweep and apply the reviewer's PASS/KILL logic.

Judged metric: SINGLE-TRAJECTORY greedy gap (locked 1.07% baseline target). Multi-start
reported for context only. Delta = 1.0% abs, head-to-head vs the from-scratch attention
baseline:
  PASS  if mamba/hilbert mean gap <= attention mean gap + 1.0%
  KILL  if mamba/hilbert clearly worse by > 1.0% AND Hilbert clearly beats random
        (i.e. ordering genuinely propagates -> a real gap, not a bug)
The reviewer locks the final ruling; this only scaffolds it.
"""
import collections
import glob
import json
import os

import numpy as np

RESULTS = os.path.join(os.environ.get("SERPENTINE_WORK", "."), "results")


def load():
    by = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RESULTS, "calib_*_s*.json"))):
        d = json.load(open(f))
        order = d["order"] if d["encoder"] == "mamba" else "-"
        by[(d["encoder"], order)].append(d)
    return by


def arr(ds, key):
    return np.array([d[key] for d in ds], dtype=float)


def main():
    by = load()
    print(f"{'config':22} {'n':>2}  {'single-traj greedy %':>22}  {'multi-start %':>16}  seeds")
    order_keys = [("attention", "-"), ("mamba", "hilbert"), ("mamba", "random"), ("mamba", "sort")]
    summary = {}
    for k in order_keys:
        ds = by.get(k, [])
        if not ds:
            print(f"{k[0]+'/'+k[1]:22} {'0':>2}  (no runs yet)")
            continue
        s = arr(ds, "greedy_single_traj_gap_pct")
        m = arr(ds, "greedy_multistart_gap_pct")
        seeds = sorted(d["seed"] for d in ds)
        summary[k] = (s.mean(), s.std(), m.mean())
        print(f"{k[0]+'/'+k[1]:22} {len(ds):>2}  {s.mean():7.2f} +/- {s.std():5.2f}        "
              f"{m.mean():7.2f}          {seeds}")

    base = summary.get(("attention", "-"))
    hil = summary.get(("mamba", "hilbert"))
    rnd = summary.get(("mamba", "random"))
    if not (base and hil):
        print("\n[verdict] incomplete — need attention + mamba/hilbert runs.")
        return

    delta = 1.0
    base_g, hil_g = base[0], hil[0]
    print(f"\nbaseline (attention) single-traj greedy = {base_g:.2f}%")
    print(f"candidate (mamba/hilbert)               = {hil_g:.2f}%  (delta budget = {delta:.1f}%)")
    if hil_g <= base_g + delta:
        print(f"[verdict] PASS  (candidate {hil_g:.2f}% <= baseline {base_g:.2f}% + {delta:.1f}%)")
    else:
        hilbert_beats_random = rnd is not None and (rnd[0] - hil_g) > 0.5
        worse = hil_g - base_g
        if hilbert_beats_random:
            print(f"[verdict] KILL  (candidate worse by {worse:.2f}% > {delta:.1f}% AND "
                  f"Hilbert beats random by {rnd[0]-hil_g:.2f}% -> genuine gap)")
        else:
            print(f"[verdict] INCONCLUSIVE (worse by {worse:.2f}% but Hilbert ~ random -> "
                  f"order may not be propagating; investigate before KILL)")
    print("Reviewer locks the final ruling.")


if __name__ == "__main__":
    main()
