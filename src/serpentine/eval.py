#!/usr/bin/env python3
"""Evaluate a pretrained POMO TSP model on a FIXED instance set.

We reuse POMO's own TSPEnv / TSPModel unchanged (repo pinned). The only injection
is monkeypatching TSProblemDef.get_random_problems (as imported into the TSPEnv
module namespace) so load_problems serves our fixed instances sequentially instead
of random ones. A single aug_factor=8 pass yields BOTH metrics:
  - NO-AUG length  = best-over-pomo on augmentation slice 0 (identity orientation)
  - x8-AUG length  = best-over-pomo-and-8-orientations
matching POMO's reported "no augmentation" and "x8 augmentation" numbers.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pomo-root", default=None,
                    help="POMO .../NEW_py_ver/TSP (else POMO_ROOT env or ./POMO)")
    ap.add_argument("--instances", required=True, help=".pt file (N, problem, 2)")
    ap.add_argument("--model-path", required=True, help="dir holding checkpoint-<epoch>.pt")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--problem-size", type=int, default=100)
    ap.add_argument("--pomo-size", type=int, default=100)
    ap.add_argument("--aug", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=500, help="raw instances per batch (pre-aug)")
    ap.add_argument("--out", required=True, help="output .npz")
    a = ap.parse_args()

    # --- make POMO importable (repo stays pristine) ---
    if a.pomo_root:
        os.environ["POMO_ROOT"] = a.pomo_root
    from serpentine.pomo import ensure_pomo_on_path
    ensure_pomo_on_path()
    import TSPEnv as tspenv_module
    from TSPEnv import TSPEnv as Env
    from TSPModel import TSPModel as Model

    # --- device (mirror POMO's tester) ---
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

    # --- fixed instances ---
    problems_all = torch.load(a.instances, map_location="cpu")
    if not torch.is_tensor(problems_all):
        problems_all = torch.as_tensor(problems_all)
    problems_all = problems_all.float()
    N = problems_all.size(0)
    assert problems_all.size(1) == a.problem_size, "problem size mismatch"

    cursor = {"i": 0}

    def fixed_loader(batch_size, problem_size):
        i = cursor["i"]
        batch = problems_all[i:i + batch_size].to(device)
        cursor["i"] = i + batch_size
        return batch

    tspenv_module.get_random_problems = fixed_loader  # the injection

    # --- build env + model, load checkpoint ---
    env = Env(problem_size=a.problem_size, pomo_size=a.pomo_size)
    model_params = {
        "embedding_dim": 128, "sqrt_embedding_dim": 128 ** 0.5, "encoder_layer_num": 6,
        "qkv_dim": 16, "head_num": 8, "logit_clipping": 10, "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }
    model = Model(**model_params)
    ckpt = os.path.join(a.model_path, f"checkpoint-{a.epoch}.pt")
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    n_params = int(sum(p.numel() for p in model.parameters()))

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # --- run ---
    # Four per-instance metrics from a single aug=8 pass over the identity orientation
    # (aug slice 0) and the 100 pomo multi-starts:
    #   st_mean  : mean over the 100 single-start greedy tours  -> "single-trajectory" (POMO 1.07%)
    #   st0      : the single-start greedy tour from city 0      -> single-trajectory (alt estimate)
    #   no_aug   : best over the 100 starts                      -> multi-start, no augmentation
    #   aug      : best over 100 starts x 8 orientations         -> x8 augmentation (POMO 0.14%)
    st_mean_lens, st0_lens, no_aug_lens, aug_lens = [], [], [], []
    cursor["i"] = 0
    done = 0
    t0 = time.time()
    with torch.no_grad():
        while done < N:
            bs = min(a.batch_size, N - done)
            env.load_problems(bs, a.aug)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, step_done = env.pre_step()
            while not step_done:
                selected, _ = model(state)
                state, reward, step_done = env.step(selected)
            # reward: (aug*bs, pomo) of NEGATIVE tour lengths
            aug_reward = reward.reshape(a.aug, bs, env.pomo_size)
            ident_len = -aug_reward[0]                  # (bs, pomo) identity-orientation lengths
            st_mean = ident_len.mean(dim=1)            # (bs,)
            st0 = ident_len[:, 0]                      # (bs,)
            max_pomo, _ = aug_reward.max(dim=2)        # (aug, bs)
            no_aug = -max_pomo[0, :]                   # (bs,)
            best_aug, _ = max_pomo.max(dim=0)          # (bs,)
            aug = -best_aug                            # (bs,)
            st_mean_lens.append(st_mean.detach().cpu().numpy())
            st0_lens.append(st0.detach().cpu().numpy())
            no_aug_lens.append(no_aug.detach().cpu().numpy())
            aug_lens.append(aug.detach().cpu().numpy())
            done += bs
    elapsed = time.time() - t0

    st_mean_lens = np.concatenate(st_mean_lens)
    st0_lens = np.concatenate(st0_lens)
    no_aug_lens = np.concatenate(no_aug_lens)
    aug_lens = np.concatenate(aug_lens)
    gpu_mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if use_cuda else 0.0

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, st_mean_len=st_mean_lens, st0_len=st0_lens,
             no_aug_len=no_aug_lens, aug_len=aug_lens, n_params=n_params,
             elapsed_sec=elapsed, gpu_mem_gb=gpu_mem_gb, aug=a.aug, pomo_size=a.pomo_size)

    print(json.dumps({
        "N": int(N),
        "single_traj_mean_len": float(st_mean_lens.mean()),
        "no_aug_mean_len": float(no_aug_lens.mean()),
        "aug_mean_len": float(aug_lens.mean()),
        "n_params": n_params,
        "elapsed_sec": round(elapsed, 2),
        "gpu_mem_gb": round(gpu_mem_gb, 3),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu",
    }))
    sys.stdout.flush()
    os._exit(0)  # dodge GPU-less teardown hang noted in the Leonardo how-to


if __name__ == "__main__":
    main()
