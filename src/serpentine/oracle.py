#!/usr/bin/env python3
"""Solve a fixed TSP instance set with LKH-3 (the optimality oracle).

Coordinates in [0,1] are scaled by 1e6 and rounded to integers (TSPLIB EUC_2D uses
integer rounded distances); the returned tour length is divided back by 1e6. With a
1e6 scale the rounding error is ~1e-5 relative — negligible for an optimality gap.
LKH on uniform TSP-100 finds the optimum essentially always with RUNS=1.
"""
import argparse
import os
import subprocess
import tempfile
import time
from multiprocessing import Pool

import numpy as np

SCALE = 10 ** 6
LKH_BIN = os.environ.get("LKH_BIN", "LKH")
RUNS = 1


def solve_one(task):
    idx, coords = task
    n = coords.shape[0]
    with tempfile.TemporaryDirectory() as d:
        tsp = os.path.join(d, "p.tsp")
        par = os.path.join(d, "p.par")
        tour = os.path.join(d, "p.tour")
        with open(tsp, "w") as f:
            f.write(f"NAME : p\nTYPE : TSP\nDIMENSION : {n}\n")
            f.write("EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
            for i in range(n):
                x = int(round(float(coords[i, 0]) * SCALE))
                y = int(round(float(coords[i, 1]) * SCALE))
                f.write(f"{i + 1} {x} {y}\n")
            f.write("EOF\n")
        with open(par, "w") as f:
            f.write(f"PROBLEM_FILE = {tsp}\nTOUR_FILE = {tour}\n")
            f.write(f"RUNS = {RUNS}\nSEED = 1234\nTRACE_LEVEL = 0\n")
        subprocess.run([LKH_BIN, par], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        length = None
        with open(tour) as f:
            for line in f:
                if "Length" in line:
                    length = int(line.strip().split("=")[-1])
                    break
        if length is None:
            raise RuntimeError(f"LKH produced no length for instance {idx}")
    return idx, length / SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, help=".npy (N, n, 2)")
    ap.add_argument("--out", required=True, help="output .npy of optimal lengths")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="0 = all instances")
    a = ap.parse_args()

    global RUNS
    RUNS = a.runs

    data = np.load(a.instances)
    if a.limit:
        data = data[:a.limit]
    N = data.shape[0]
    tasks = [(i, data[i]) for i in range(N)]
    lengths = np.zeros(N, dtype=np.float64)

    t0 = time.time()
    with Pool(a.procs) as pool:
        for idx, length in pool.imap_unordered(solve_one, tasks, chunksize=8):
            lengths[idx] = length
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, lengths)
    print(f"solved N={N} runs={RUNS} procs={a.procs} elapsed={elapsed:.1f}s "
          f"mean_opt={lengths.mean():.6f} bin={LKH_BIN}")


if __name__ == "__main__":
    main()
