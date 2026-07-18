#!/usr/bin/env python3
"""Best-of-k serialization-ensemble probe (eval-only, no retraining).

A model TRAINED with order_mode=random sees a fresh permutation per batch, so at
inference we can run the SAME weights under k different serialization orders and keep,
per instance, the best tour. This is a Mamba-native diversity source — attention has no
ordering knob, its only decode diversity is the start node. The probe asks whether order
diversity (a) substitutes for start diversity and (b) stacks on top of it.

Metrics per k (cumulative over passes, all greedy/argmax, no augmentation):
  avg_start_best_of_k : per instance & start, best tour over k orders, then mean over
                        starts -> the single-traj analogue with order ensembling.
  full_best_of_k      : per instance, best over (k orders x 100 starts) -> the
                        multistart analogue with order ensembling.
k=1 reproduces the standard single-traj / multistart numbers for the checkpoint
(pass 0 uses order seeds disjoint from training; any k=1 mismatch vs the training-time
eval is the order-resampling noise floor, itself informative).

Only meaningful for random-trained checkpoints (hilbert-trained models collapse off
their training order — see ORDER_PROBE.md). Mirrors rl.py's evaluate() batching exactly.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from serpentine.model import build_model
from serpentine.rl import model_params
from serpentine.pomo import ensure_pomo_on_path

ensure_pomo_on_path()
import TSPEnv as tspenv_module  # noqa: E402
from TSPEnv import TSPEnv as Env  # noqa: E402


@torch.no_grad()
def greedy_lengths(model, env, data, batch=500):
    """All-start greedy rollout -> per-instance per-start tour lengths (N, pomo)."""
    device = next(model.parameters()).device
    N = data.size(0)
    cur = {"i": 0}
    saved = tspenv_module.get_random_problems

    def loader(bs, ps):
        i = cur["i"]
        b = data[i:i + bs].to(device)
        cur["i"] = i + bs
        return b

    tspenv_module.get_random_problems = loader
    model.eval()
    lens = []
    done = 0
    while done < N:
        bs = min(batch, N - done)
        env.load_problems(bs, 1)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, reward, d = env.pre_step()
        while not d:
            selected, _ = model(state)
            state, reward, d = env.step(selected)
        lens.append((-reward).cpu().numpy())   # (bs, pomo)
        done += bs
    tspenv_module.get_random_problems = saved
    return np.concatenate(lens, axis=0)        # (N, pomo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", choices=["mamba", "bimamba"], default="mamba")
    ap.add_argument("--trained-order", choices=["random", "hilbert", "sort"], default="random",
                    help="order the checkpoint was trained with (recorded; probe runs random orders)")
    ap.add_argument("--k", type=int, default=8, help="number of serialization orders")
    ap.add_argument("--eval-instances", required=True)
    ap.add_argument("--eval-opt", required=True)
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--order-seed", type=int, default=990000,
                    help="base rng seed for probe orders (disjoint from training seeds)")
    ap.add_argument("--use-kernel", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        try:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        except Exception:
            torch.set_default_device("cuda")
            torch.set_default_dtype(torch.float32)
    else:
        device = torch.device("cpu")

    torch.manual_seed(1)
    np.random.seed(1)
    env = Env(problem_size=100, pomo_size=100)
    # order_mode=random so each pass draws serializations from the pass-specific rng
    model = build_model(a.encoder, **model_params(a.encoder, "random", 1, a.use_kernel))
    ck = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.to(device)

    data = torch.load(a.eval_instances, map_location="cpu").float()[:a.eval_n]
    opt = np.load(a.eval_opt).astype(np.float64)[:a.eval_n]
    opt_mean = float(opt.mean())

    all_lens = []                              # k x (N, pomo)
    for j in range(a.k):
        model.encoder._rng = np.random.default_rng(a.order_seed + j)
        lens = greedy_lengths(model, env, data)
        all_lens.append(lens)
        print(f"[pass {j + 1}/{a.k}] single={100 * (lens.mean() / opt_mean - 1):.3f}% "
              f"multi={100 * (lens.min(axis=1).mean() / opt_mean - 1):.3f}%", flush=True)

    stack = np.stack(all_lens)                 # (k, N, pomo)
    curve = []
    for k in range(1, a.k + 1):
        best_per_start = stack[:k].min(axis=0)          # (N, pomo): best order per start
        avg_start = float(100 * (best_per_start.mean(axis=1).mean() / opt_mean - 1))
        full = float(100 * (best_per_start.min(axis=1).mean() / opt_mean - 1))
        curve.append({"k": k, "avg_start_best_of_k_gap_pct": avg_start,
                      "full_best_of_k_gap_pct": full})
        print(f"[bestof] k={k} avg-start={avg_start:.3f}% full={full:.3f}%", flush=True)

    out = {
        "probe": "bestofk_serialization_ensemble",
        "checkpoint": os.path.basename(a.ckpt),
        "trained_step": int(ck.get("step", -1)),
        "encoder": a.encoder, "trained_order": a.trained_order,
        "k": int(a.k), "eval_n": int(data.size(0)), "opt_mean": opt_mean,
        "order_seed": int(a.order_seed), "use_kernel": bool(a.use_kernel),
        "device": "cuda" if use_cuda else "cpu",
        "curve": curve,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
