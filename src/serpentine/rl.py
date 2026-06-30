#!/usr/bin/env python3
"""From-scratch POMO REINFORCE training, encoder pluggable (attention | mamba).

The RL step is copied verbatim from POMO's TSPTrainer._train_one_batch (advantage =
reward - pomo-mean; loss = -(advantage * sum log p)). ONLY the encoder differs, so
baseline and candidate share identical RL / optimizer / seeds / budget. Greedy gap is
evaluated on the fixed held-out test set against the LKH oracle.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from serpentine.model import build_model
from serpentine.pomo import ensure_pomo_on_path

ensure_pomo_on_path()
import TSPEnv as tspenv_module  # noqa: E402
from TSPEnv import TSPEnv as Env  # noqa: E402


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def model_params(encoder, order, seed, use_kernel=False):
    # bimamba holds two mixers per layer, so it uses ~half the layers to keep total
    # parameters matched to the 10-layer unidirectional mamba (a clean discriminator).
    layers = {"mamba": 10, "bimamba": 5}.get(encoder, 6)
    return dict(
        embedding_dim=128, sqrt_embedding_dim=128 ** 0.5,
        encoder_layer_num=layers,
        qkv_dim=16, head_num=8, logit_clipping=10, ff_hidden_dim=512,
        eval_type="argmax", order_mode=order, order_seed=seed, use_kernel=use_kernel,
    )


@torch.no_grad()
def evaluate(model, env, fixed_pt, opt_npy, eval_n, batch=500):
    """Greedy (argmax) eval on fixed instances -> single-traj & multi-start no-aug gaps."""
    device = next(model.parameters()).device
    data = torch.load(fixed_pt, map_location="cpu").float()[:eval_n]
    opt = np.load(opt_npy).astype(np.float64)[:eval_n]
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
    st_mean, no_aug = [], []
    done = 0
    while done < N:
        bs = min(batch, N - done)
        env.load_problems(bs, 1)               # no augmentation
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        state, reward, d = env.pre_step()
        while not d:
            selected, _ = model(state)
            state, reward, d = env.step(selected)
        # reward: (bs, pomo) negative lengths
        lens = -reward
        st_mean.append(lens.mean(dim=1).cpu().numpy())          # single-traj (mean start)
        maxr, _ = reward.max(dim=1)
        no_aug.append((-maxr).cpu().numpy())                    # best of 100 starts
        done += bs
    tspenv_module.get_random_problems = saved

    st_mean = np.concatenate(st_mean)
    no_aug = np.concatenate(no_aug)
    return {
        "eval_n": int(N),
        "opt_mean": float(opt.mean()),
        "greedy_single_traj_gap_pct": float(100 * (st_mean.mean() / opt.mean() - 1)),
        "greedy_multistart_gap_pct": float(100 * (no_aug.mean() / opt.mean() - 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["attention", "mamba", "bimamba"], required=True)
    ap.add_argument("--order", choices=["hilbert", "random", "sort"], default="hilbert")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-instances", required=True)   # fixed .pt
    ap.add_argument("--eval-opt", required=True)          # LKH .npy
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--use-kernel", action="store_true", help="use mamba-ssm CUDA scan")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: <out>.ckpt.pt)")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="periodic eval interval in steps (0 = end only)")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="wall-clock hard cap for this job in hours (0 = none)")
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

    set_all_seeds(a.seed)
    env = Env(problem_size=100, pomo_size=100)
    model = build_model(a.encoder, **model_params(a.encoder, a.order, a.seed, a.use_kernel))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    n_params = int(sum(p.numel() for p in model.parameters()))

    # Resume from checkpoint if present (functional resume; survives walltime cuts).
    ckpt_path = a.ckpt or (os.path.splitext(a.out)[0] + ".ckpt.pt")
    start_step, losses = 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_step = int(ck["step"])
        losses = list(ck.get("losses", []))
        print(f"[resume] {ckpt_path} at step {start_step}/{a.steps}", flush=True)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    model.train()
    curve_path = os.path.splitext(a.out)[0] + ".curve.jsonl"

    def run_eval():
        ev_ = evaluate(model, env, a.eval_instances, a.eval_opt, a.eval_n)
        model.train()  # evaluate() switched to eval(); restore for training
        return ev_

    def log_curve(step, ev_, elapsed):
        row = {"step": int(step),
               "single_traj_gap_pct": ev_["greedy_single_traj_gap_pct"],
               "multistart_gap_pct": ev_["greedy_multistart_gap_pct"],
               "elapsed_sec": round(elapsed, 1)}
        with open(curve_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[curve] step {row['step']} single={row['single_traj_gap_pct']:.2f}% "
              f"multi={row['multistart_gap_pct']:.2f}% elapsed={elapsed / 3600:.2f}h", flush=True)

    t0 = time.time()
    # anchor the curve at the resume point (e.g. step 60000)
    if a.eval_every > 0 and start_step < a.steps:
        log_curve(start_step, run_eval(), 0.0)

    capped = False
    last_step = start_step
    for step in range(start_step + 1, a.steps + 1):
        env.load_problems(a.batch_size)        # fresh random instances (aug=1)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        prob_list = torch.zeros(size=(a.batch_size, env.pomo_size, 0))
        state, reward, done = env.pre_step()
        while not done:
            selected, prob = model(state)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        # --- POMO REINFORCE (verbatim) ---
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = (-advantage * log_prob).mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        last_step = step
        if step % a.ckpt_every == 0:
            torch.save({"step": step, "model": model.state_dict(),
                        "optim": optimizer.state_dict(), "losses": losses}, ckpt_path)
        if step % a.log_every == 0:
            recent = float(np.mean(losses[-a.log_every:]))
            max_r, _ = reward.max(dim=1)
            ips = (step - start_step) / (time.time() - t0)
            print(f"[{a.encoder}/{a.order}] step {step}/{a.steps} "
                  f"loss {recent:.4f} train_score {(-max_r.float().mean()).item():.4f} "
                  f"({ips:.2f} it/s)", flush=True)
        if a.eval_every > 0 and step % a.eval_every == 0:
            log_curve(step, run_eval(), time.time() - t0)
        if a.max_hours > 0 and (time.time() - t0) >= a.max_hours * 3600:
            print(f"[cap] wall-clock {a.max_hours}h reached at step {step}", flush=True)
            capped = True
            break

    train_sec = time.time() - t0
    steps_done = max(1, last_step - start_step)
    torch.save({"step": last_step, "model": model.state_dict(),
                "optim": optimizer.state_dict(), "losses": losses}, ckpt_path)

    ev = run_eval()
    if a.eval_every > 0:
        log_curve(last_step, ev, train_sec)
    gpu_mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if use_cuda else 0.0
    result = {
        "encoder": a.encoder, "order": a.order, "seed": a.seed,
        "steps": last_step, "target_steps": a.steps, "capped": capped,
        "batch_size": a.batch_size, "n_params": n_params,
        "train_sec": round(train_sec, 1), "steps_per_sec": round(steps_done / train_sec, 3),
        "gpu_hours": round(train_sec / 3600, 3),
        "final_loss": float(np.mean(losses[-min(50, len(losses)):])) if losses else 0.0,
        "gpu_mem_gb": round(gpu_mem_gb, 3),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        **ev,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
