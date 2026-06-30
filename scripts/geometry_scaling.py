#!/usr/bin/env python3
"""Model-free Hilbert-geometry studies (no model, CPU-only).

TASK 1  SEGMENT-RECONNECT: chunk the Hilbert line into S contiguous segments; for the
        spatial k-NN pairs that land >T apart on the line (probe D's "scattered" pairs),
        measure their segment-index separation. Answers whether a thin global channel over
        per-segment summaries could reconnect scattered spatial neighbours, and at what hop range.

TASK 2  SCALING-WALL: repeat probe D's "fraction of k=5 spatial neighbours >10 apart on the
        line" for N in {100,500,1000,5000}. Quantifies whether serialization-locality loss
        worsens with N (the core large-N thesis question).
"""
import argparse
import json
import os
import sys

import numpy as np

from serpentine.serialization import serialize_order


def knn_indices(pts, k):
    """(N,k) nearest-neighbour node indices (self excluded), memory-safe via row chunks."""
    N = pts.shape[0]
    idx = np.empty((N, k), dtype=np.int64)
    chunk = 1024
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d = np.linalg.norm(pts[s:e, None, :] - pts[None, :, :], axis=2)  # (rows, N)
        rows = np.arange(e - s)
        d[rows, s + rows] = np.inf                                       # exclude self
        idx[s:e] = np.argpartition(d, k, axis=1)[:, :k]
    return idx


def line_index(pts, bits):
    """node -> position on the Hilbert line (0..N-1)."""
    order = serialize_order(pts, bits=bits, mode="hilbert")
    return np.argsort(order)


def seps_for_instance(pts, bits, k):
    inv = line_index(pts, bits)
    nn = knn_indices(pts, k)
    return np.abs(inv[:, None] - inv[nn]).ravel(), inv, nn


# --------------------------------------------------------------------------- task 1
def segment_reconnect(instances, S_list, bits=7, k=5, thresh=10):
    out = []
    for S in S_list:
        all_pairs = scattered = diff_seg = 0
        segdiffs = []
        seg_size = None
        for pts in instances:
            N = pts.shape[0]
            seg_size = N // S
            inv = line_index(pts, bits)
            nn = knn_indices(pts, k)
            seps = np.abs(inv[:, None] - inv[nn])               # (N,k)
            seg = (inv * S) // N                                 # node -> segment 0..S-1
            sd = np.abs(seg[:, None] - seg[nn])                 # (N,k)
            mask = seps > thresh
            all_pairs += seps.size
            scattered += int(mask.sum())
            diff_seg += int((sd[mask] > 0).sum())
            segdiffs.append(sd[mask])
        sd = np.concatenate(segdiffs) if segdiffs else np.array([])
        out.append({
            "S": S, "segment_size": seg_size, "thresh": thresh,
            "all_knn_pairs": all_pairs, "scattered_pairs": scattered,
            "frac_scattered": scattered / all_pairs,
            "scattered_in_different_segment_frac": (diff_seg / scattered) if scattered else None,
            "segdiff_mean": float(sd.mean()), "segdiff_median": float(np.median(sd)),
            "segdiff_p90": float(np.percentile(sd, 90)), "segdiff_max": int(sd.max()),
            "frac_segdiff_eq1": float((sd == 1).mean()),
            "frac_segdiff_le2": float((sd <= 2).mean()),
            "frac_segdiff_le3": float((sd <= 3).mean()),
        })
    return out


# --------------------------------------------------------------------------- task 2
def scaling_wall(specs, k=5, thresh=10):
    rows = []
    for label, gen, bits, n_inst in specs:
        seps_all = []
        for _ in range(n_inst):
            pts = gen()
            s, _, _ = seps_for_instance(pts, bits, k)
            seps_all.append(s)
        seps = np.concatenate(seps_all)
        N = gen.N
        rows.append({
            "label": label, "N": N, "bits": bits, "instances": n_inst, "pairs": int(seps.size),
            "frac_gt10": float((seps > thresh).mean()),
            "frac_gt_Nover10": float((seps > N / 10).mean()),
            "sep_median": float(np.median(seps)), "sep_mean": float(seps.mean()),
            "sep_p90": float(np.percentile(seps, 90)),
        })
    return rows


class UniformGen:
    def __init__(self, N, rng):
        self.N = N; self.rng = rng
    def __call__(self):
        return self.rng.random((self.N, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, help="real test .pt for the N=100 anchor")
    ap.add_argument("--n-anchor", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    data = torch.load(a.instances, map_location="cpu").float()[:a.n_anchor].numpy()
    real_100 = [data[i] for i in range(data.shape[0])]   # 100 real instances of 100 nodes

    # TASK 1 on the real held-out instances (bits=7, matches probe D's 14.4% scattered)
    t1 = segment_reconnect(real_100, S_list=[10, 20], bits=7, k=5, thresh=10)

    # TASK 2 scaling: consistent bits=10 (grid >> N, quantization negligible) across N,
    # plus the real-set bits=7 anchor to tie back to probe D.
    rng = np.random.default_rng(20260630)
    specs = [
        ("N=100 real bits=7 (probe-D anchor)", _ConstGen(real_100), 7, len(real_100)),
        ("N=100 uniform bits=10", UniformGen(100, rng), 10, 100),
        ("N=500 uniform bits=10", UniformGen(500, rng), 10, 50),
        ("N=1000 uniform bits=10", UniformGen(1000, rng), 10, 20),
        ("N=5000 uniform bits=10", UniformGen(5000, rng), 10, 5),
    ]
    t2 = scaling_wall(specs, k=5, thresh=10)

    out = {"task1_segment_reconnect": t1, "task2_scaling_wall": t2,
           "k": 5, "thresh": 10}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    sys.stdout.flush()
    os._exit(0)


class _ConstGen:
    """Replays a fixed list of instances (for the real-data anchor row)."""
    def __init__(self, insts):
        self.insts = insts; self.N = insts[0].shape[0]; self._i = 0
    def __call__(self):
        p = self.insts[self._i % len(self.insts)]; self._i += 1; return p


if __name__ == "__main__":
    main()
