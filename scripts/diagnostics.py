#!/usr/bin/env python3
"""CPU diagnostic battery on the trained mamba/hilbert s1 checkpoint (forward-only, no GPU).

All four probes run pure-PyTorch (use_kernel=False) on a fixed held-out subset, with NO
retraining. The kernel<->pure parity gate (<1e-3) makes the CPU path faithful to the
kernel-trained weights.

  A DIRECTIONALITY     re-encode with the Hilbert sequence reversed / forward+reverse averaged;
                       single-traj greedy gap vs the forward baseline.
  B REPRESENTATION     effective rank (participation ratio), mean pairwise cosine, per-dim
                       variance of encoder embeddings — mamba vs attention.
  C STATE DECAY        finite-difference receptive field: L2 change in other nodes' embeddings
                       vs their separation in the Hilbert sequence (ordering held fixed).
  D CURVE GEOMETRY     model-free: separation in Hilbert-line index of each node's k nearest
                       2D neighbours; fraction of spatial neighbours >10 apart on the line.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from serpentine.model import build_model
from serpentine.rl import model_params
from serpentine.serialization import serialize_order
from serpentine.pomo import ensure_pomo_on_path

ensure_pomo_on_path()
import TSPEnv as tspenv_module  # noqa: E402
from TSPEnv import TSPEnv as Env  # noqa: E402


def load_model(encoder, ckpt, seed=1):
    m = build_model(encoder, **model_params(encoder, "hilbert", seed, use_kernel=False))
    ck = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(ck["model"])
    m.eval()
    return m, int(ck.get("step", -1))


@torch.no_grad()
def run_with_order(enc, data, order):
    """MambaEncoder forward using a GIVEN serialization order (seq->node). Output in node order."""
    h = enc.embedding(data)
    idx = order.unsqueeze(-1).expand(-1, -1, h.size(-1))
    hs = torch.gather(h, 1, idx)
    for blk, nrm in zip(enc.blocks, enc.norms):
        hs = hs + blk(nrm(hs))
    inv = torch.argsort(order, dim=1)
    return torch.gather(hs, 1, inv.unsqueeze(-1).expand(-1, -1, hs.size(-1)))


# ----------------------------------------------------------------------------- A
@torch.no_grad()
def eval_single_traj(model, env, data, opt, embed_fn=None, batch=100):
    """single-traj greedy gap %; embed_fn(problems)->(B,P,d) overrides the encoder if given."""
    N = data.size(0)
    cur = {"i": 0}
    saved = tspenv_module.get_random_problems

    def loader(bs, ps):
        i = cur["i"]; b = data[i:i + bs]; cur["i"] = i + bs; return b

    tspenv_module.get_random_problems = loader
    model.eval()
    st_mean, done = [], 0
    while done < N:
        bs = min(batch, N - done)
        env.load_problems(bs, 1)
        reset_state, _, _ = env.reset()
        if embed_fn is None:
            model.pre_forward(reset_state)
        else:
            emb = embed_fn(reset_state.problems)
            model.encoded_nodes = emb
            model.decoder.set_kv(emb)
        state, reward, d = env.pre_step()
        while not d:
            selected, _ = model(state)
            state, reward, d = env.step(selected)
        st_mean.append((-reward).mean(dim=1).cpu().numpy())
        done += bs
    tspenv_module.get_random_problems = saved
    st_mean = np.concatenate(st_mean)
    return float(100 * (st_mean.mean() / opt.mean() - 1))


def diag_A(model, env, data, opt):
    enc = model.encoder
    order_fwd, _ = enc._orders(data[:1])  # warmup (unused); orders are per-batch below

    def fwd(p):
        o, _ = enc._orders(p); return run_with_order(enc, p, o)

    def rev(p):
        o, _ = enc._orders(p); return run_with_order(enc, p, torch.flip(o, dims=[1]))

    def avg(p):
        o, _ = enc._orders(p)
        return 0.5 * (run_with_order(enc, p, o) + run_with_order(enc, p, torch.flip(o, dims=[1])))

    base = eval_single_traj(model, env, data, opt, embed_fn=None)   # the real encoder.forward
    g_fwd = eval_single_traj(model, env, data, opt, embed_fn=fwd)   # must equal base (parity check)
    g_rev = eval_single_traj(model, env, data, opt, embed_fn=rev)
    g_avg = eval_single_traj(model, env, data, opt, embed_fn=avg)
    return {
        "forward_baseline_pct": base,
        "forward_reimpl_pct": g_fwd,
        "reimpl_matches_baseline": abs(g_fwd - base) < 1e-6,
        "reverse_pct": g_rev,
        "avg_fwd_rev_pct": g_avg,
        "delta_reverse_vs_forward": g_rev - base,
        "delta_avg_vs_forward": g_avg - base,
    }


# ----------------------------------------------------------------------------- B
@torch.no_grad()
def embed_stats(model, data, batch=100):
    embs, N, done = [], data.size(0), 0
    while done < N:
        bs = min(batch, N - done)
        e = model.encoder(data[done:done + bs])           # (bs,P,d)
        embs.append(e.reshape(-1, e.size(-1)))
        done += bs
    X = torch.cat(embs, 0)                                 # (N*P, d)
    n, d = X.shape
    Xc = X - X.mean(0, keepdim=True)
    lam = torch.linalg.svdvals(Xc) ** 2                    # covariance eigenvalues (up to scale)
    pr = float((lam.sum() ** 2 / (lam ** 2).sum()).item())  # participation ratio (1..d)
    Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
    mean_vec = Xn.mean(0)
    sum_all = (n ** 2) * float(mean_vec @ mean_vec)        # sum_{i,j} cos(x_i,x_j)
    mean_cos = (sum_all - n) / (n * (n - 1))               # exclude diagonal (self-cos=1)
    var = X.var(0, unbiased=False)
    return {
        "n_vectors": int(n), "d": int(d),
        "participation_ratio": pr,
        "effective_rank_frac_of_d": pr / d,
        "mean_pairwise_cosine": float(mean_cos),
        "perdim_var_mean": float(var.mean()), "perdim_var_min": float(var.min()),
        "perdim_var_max": float(var.max()),
        "dead_dims_lt_1pct_of_max": int((var < 0.01 * var.max()).sum().item()),
    }


# ----------------------------------------------------------------------------- C
@torch.no_grad()
def diag_C(model, data, n_inst=10, source_nodes=(0, 25, 50, 75), eps=0.02):
    enc = model.encoder
    sub = data[:n_inst]
    order, _ = enc._orders(sub)                # FIXED order from unperturbed coords
    inv = torch.argsort(order, dim=1)          # node -> sequence position
    E0 = run_with_order(enc, sub, order)
    P = sub.size(1)
    by_abs = {}                                 # |sep| -> deltas
    down, up = [], []                           # downstream (sep>0) / upstream (sep<0)
    for s in source_nodes:
        d2 = sub.clone(); d2[:, s, 0] += eps    # perturb x of node s
        E1 = run_with_order(enc, d2, order)
        delta = (E1 - E0).norm(dim=2)           # (n_inst, P) L2 change per node
        for b in range(n_inst):
            sp = int(inv[b, s].item())
            for j in range(P):
                if j == s:
                    continue
                sep = int(inv[b, j].item()) - sp
                by_abs.setdefault(abs(sep), []).append(float(delta[b, j].item()))
                (down if sep > 0 else up).append(float(delta[b, j].item()))
    bins = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 99)]
    curve = []
    for lo, hi in bins:
        vals = [v for k, lst in by_abs.items() if lo <= k <= hi for v in lst]
        curve.append({"sep_bin": f"{lo}-{hi}", "n": len(vals),
                      "mean_delta": float(np.mean(vals)) if vals else None})
    src_delta = float(np.mean([by_abs.get(0, [0])[0]])) if 0 in by_abs else None
    return {
        "eps": eps, "n_inst": n_inst, "source_nodes": list(source_nodes),
        "mean_delta_downstream": float(np.mean(down)),
        "mean_delta_upstream": float(np.mean(up)),
        "upstream_over_downstream_ratio": float(np.mean(up) / np.mean(down)) if np.mean(down) else None,
        "decay_curve": curve,
    }


# ----------------------------------------------------------------------------- D
def diag_D(data, k=5, bits=7, n_inst=100):
    sub = data[:n_inst].numpy()
    seps, total, far = [], 0, 0
    for b in range(n_inst):
        pts = sub[b]
        order = serialize_order(pts, bits=bits, mode="hilbert")
        inv = np.argsort(order)                          # node -> line index
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        P = pts.shape[0]
        for i in range(P):
            for j in np.argpartition(D[i], k)[:k]:
                sep = abs(int(inv[i]) - int(inv[j]))
                seps.append(sep); total += 1
                if sep > 10:
                    far += 1
    seps = np.array(seps)
    return {
        "k": k, "bits": bits, "n_inst": n_inst, "pairs": int(total),
        "sep_mean": float(seps.mean()), "sep_median": float(np.median(seps)),
        "sep_p90": float(np.percentile(seps, 90)), "sep_max": int(seps.max()),
        "frac_neighbors_gt10_apart": float(far / total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mamba-ckpt", required=True)
    ap.add_argument("--attn-ckpt", required=True)
    ap.add_argument("--eval-instances", required=True)
    ap.add_argument("--eval-opt", required=True)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    data = torch.load(a.eval_instances, map_location="cpu").float()[:a.eval_n]
    opt = np.load(a.eval_opt).astype(np.float64)[:a.eval_n]
    env = Env(problem_size=100, pomo_size=100)

    mamba, mstep = load_model("mamba", a.mamba_ckpt, a.seed)
    attn, astep = load_model("attention", a.attn_ckpt, a.seed)

    print("[A] directionality ...", flush=True)
    A = diag_A(mamba, env, data, opt)
    print("[B] representation health ...", flush=True)
    B = {"mamba": embed_stats(mamba, data), "attention": embed_stats(attn, data)}
    print("[C] state decay ...", flush=True)
    C = diag_C(mamba, data)
    print("[D] curve geometry ...", flush=True)
    D = diag_D(data)

    out = {
        "mamba_step": mstep, "attn_step": astep, "eval_n": a.eval_n,
        "use_kernel": False, "device": "cpu",
        "A_directionality": A, "B_representation": B,
        "C_state_decay": C, "D_curve_geometry": D,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
