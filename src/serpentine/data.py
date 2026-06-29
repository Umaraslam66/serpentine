#!/usr/bin/env python3
"""Generate a fixed, seeded Euclidean TSP test set matching POMO's distribution.

POMO's get_random_problems is exactly `torch.rand(size=(batch, n, 2))` (Uniform[0,1]^2).
We reproduce it under a fixed torch seed so the SAME instances can be fed to both the
POMO model (for tour lengths) and LKH-3 (for the optimal reference). No GPU needed.
"""
import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="number of instances")
    ap.add_argument("--problem-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True, help="output path prefix (writes .pt and .npy)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    # Identical call to POMO's get_random_problems (CPU, default generator).
    problems = torch.rand(size=(a.n, a.problem_size, 2))

    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(problems, a.out + ".pt")
    np.save(a.out + ".npy", problems.numpy().astype(np.float64))

    print(
        f"wrote n={a.n} problem_size={a.problem_size} seed={a.seed} "
        f"-> {a.out}.pt / {a.out}.npy"
    )
    # Checksum so the eval and LKH sides can assert they got the same instances.
    print(f"checksum_sum={float(problems.sum()):.6f}")
    print(f"checksum_first_xy={problems[0,0,0].item():.6f},{problems[0,0,1].item():.6f}")


if __name__ == "__main__":
    main()
