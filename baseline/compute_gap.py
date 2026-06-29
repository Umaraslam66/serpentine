#!/usr/bin/env python3
"""Combine POMO tour lengths + LKH optimal lengths into the optimality gap report."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pomo", required=True, help=".npz from eval_pomo.py")
    ap.add_argument("--opt", required=True, help=".npy from lkh_solve.py")
    ap.add_argument("--out", required=True, help="output .json report")
    a = ap.parse_args()

    d = np.load(a.pomo)
    no_aug = np.asarray(d["no_aug_len"], dtype=np.float64)
    aug = np.asarray(d["aug_len"], dtype=np.float64)
    opt = np.load(a.opt).astype(np.float64)

    n = min(len(no_aug), len(aug), len(opt))
    no_aug, aug, opt = no_aug[:n], aug[:n], opt[:n]

    def block(model_len):
        return {
            "mean_len": float(model_len.mean()),
            # mean-ratio gap (POMO/Kool convention): mean(model)/mean(opt) - 1
            "gap_meanratio_pct": float(100 * (model_len.mean() / opt.mean() - 1)),
            # per-instance mean gap
            "gap_perinst_pct": float(100 * np.mean(model_len / opt - 1)),
        }

    report = {
        "N": int(n),
        "opt_mean_len": float(opt.mean()),
        "no_aug": block(no_aug),
        "aug": block(aug),
        "n_params": int(d["n_params"]),
        "eval_elapsed_sec": float(d["elapsed_sec"]),
        "gpu_mem_gb": float(d["gpu_mem_gb"]),
        "pomo_size": int(d["pomo_size"]),
        "aug_factor": int(d["aug"]),
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
