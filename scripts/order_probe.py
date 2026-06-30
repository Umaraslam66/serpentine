#!/usr/bin/env python3
"""Inference-time order-sensitivity probe (CPU, no retraining).

Loads a trained mamba/hilbert checkpoint and evaluates single-trajectory greedy gap on a
fixed held-out subset under three INPUT orderings (hilbert / random / coordinate-sort)
WITHOUT retraining. Only the encoder's serialization order is swapped between passes; the
weights are untouched.

Interpretation:
  * random ~= hilbert  -> the ordering is not reaching the decoder (BUG signal)
  * random clearly worse -> order propagates to the decision (genuine-capacity signal)

This is a PROXY, not the formal control: the model was *trained* on hilbert, so a
train/test order mismatch is expected to hurt if (and only if) order actually matters.
The from-scratch random/sort ablation remains the formal confirmation.

Runs the pure-PyTorch Mamba path (use_kernel=False) so it needs no GPU; the kernel<->pure
parity gate (<1e-3) makes this faithful to the kernel-trained weights.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from serpentine.model import build_model
from serpentine.rl import evaluate, model_params
from serpentine.pomo import ensure_pomo_on_path

ensure_pomo_on_path()
from TSPEnv import TSPEnv as Env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained mamba/hilbert checkpoint (.ckpt.pt)")
    ap.add_argument("--eval-instances", required=True, help="fixed held-out .pt")
    ap.add_argument("--eval-opt", required=True, help="LKH optimal .npy")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--order-seed", type=int, default=1234,
                    help="fixed rng seed for the random ordering (reproducibility)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cpu")  # probe is CPU-only by design

    env = Env(problem_size=100, pomo_size=100)
    # Build with the trained architecture (mamba => encoder_layer_num=10), pure-PyTorch scan.
    model = build_model("mamba", **model_params("mamba", "hilbert", a.seed, use_kernel=False))
    ck = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    trained_step = int(ck.get("step", -1))

    single, multi = {}, {}
    for mode in ["hilbert", "random", "sort"]:
        model.encoder.order_mode = mode
        model.encoder._rng = np.random.default_rng(a.order_seed)  # deterministic per mode
        ev = evaluate(model, env, a.eval_instances, a.eval_opt, a.eval_n)
        single[mode] = ev["greedy_single_traj_gap_pct"]
        multi[mode] = ev["greedy_multistart_gap_pct"]
        print(f"[probe] order={mode:7s} single={single[mode]:.3f}%  "
              f"multi={multi[mode]:.3f}%  (n={ev['eval_n']})", flush=True)

    out = {
        "probe": "inference_order_sensitivity",
        "checkpoint": os.path.basename(a.ckpt),
        "trained_step": trained_step,
        "trained_order": "hilbert",
        "eval_n": int(a.eval_n),
        "use_kernel": False,
        "device": "cpu",
        "single_traj_gap_pct": single,
        "multistart_gap_pct": multi,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)
    sys.stdout.flush()
    os._exit(0)  # dodge GPU-less teardown hang noted in the Leonardo how-to


if __name__ == "__main__":
    main()
