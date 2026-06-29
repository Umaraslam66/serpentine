#!/usr/bin/env python3
"""Pluggable TSP encoders: POMO attention (baseline) vs Hilbert-serialized Mamba.

build_model() reuses POMO's TSPModel wholesale and swaps ONLY the encoder attribute,
so the decoder, the POMO multi-start forward, and the RL interface are byte-identical
between baseline and candidate. The Mamba block is a faithful pure-PyTorch Mamba-1
(in_proj -> causal conv -> selective SSM -> gated out_proj); for Gate-0 smoke this
avoids a CUDA-kernel compile. Ordering uses the unit-tested numpy core in hilbert.py.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hilbert import serialize_order

# Make POMO importable relative to this file (repo pinned, untouched).
_HERE = os.path.dirname(os.path.abspath(__file__))
_POMO_TSP = os.path.normpath(os.path.join(_HERE, "..", "POMO", "NEW_py_ver", "TSP"))
for _p in (os.path.join(_POMO_TSP, "POMO"), _POMO_TSP):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class MambaBlock(nn.Module):
    """Mamba-1 mixer block (pure PyTorch, sequential selective scan)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None,
                 use_kernel=False):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or max(1, d_model // 16)
        self.use_kernel = use_kernel

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))      # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, _ = x.shape
        x_and_z = self.in_proj(x)
        xi, z = x_and_z.chunk(2, dim=-1)             # (B, L, d_inner) each
        xi = self.conv1d(xi.transpose(1, 2))[..., :L].transpose(1, 2)  # causal
        xi = F.silu(xi)
        x_dbl = self.x_proj(xi)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))            # (B, L, d_inner)
        A = -torch.exp(self.A_log)                   # (d_inner, d_state)
        y = self._selective_scan(xi, dt, A, Bm, Cm)
        y = y + xi * self.D                          # skip (D) term
        y = y * F.silu(z)                            # gating
        return self.out_proj(y)

    def _selective_scan(self, u, dt, A, B, C):
        # u, dt: (b, l, d_inner); A: (d_inner, n); B, C: (b, l, n). Returns scan (no D).
        if self.use_kernel:
            return self._selective_scan_kernel(u, dt, A, B, C)
        b, l, d = u.shape
        dA = torch.exp(dt.unsqueeze(-1) * A)         # (b, l, d, n)
        dBu = (dt.unsqueeze(-1) * B.unsqueeze(2)) * u.unsqueeze(-1)  # (b, l, d, n)
        h = torch.zeros(b, d, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(l):
            h = dA[:, t] * h + dBu[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
        return torch.stack(ys, dim=1)                # (b, l, d)

    def _selective_scan_kernel(self, u, dt, A, B, C):
        # mamba-ssm fast CUDA scan. Kernel layout is channels-first (b, d, l);
        # B, C are (b, n, l) (ngroups=1, shared across channels) matching the pure path.
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        y = selective_scan_fn(
            u.transpose(1, 2).contiguous(),
            dt.transpose(1, 2).contiguous(),
            A,
            B.transpose(1, 2).contiguous(),
            C.transpose(1, 2).contiguous(),
            D=None, z=None, delta_bias=None, delta_softplus=False,
        )
        return y.transpose(1, 2)                      # (b, l, d)


class MambaEncoder(nn.Module):
    """Embed -> serialize (Hilbert/sort/random) -> Mamba stack -> un-serialize.

    Output is in ORIGINAL node order so the POMO decoder indexes nodes correctly.
    """

    def __init__(self, **mp):
        super().__init__()
        d = mp["embedding_dim"]
        n_layers = mp["encoder_layer_num"]
        self.order_mode = mp.get("order_mode", "hilbert")
        self.bits = mp.get("hilbert_bits", 7)
        self.embedding = nn.Linear(2, d)
        self.blocks = nn.ModuleList([
            MambaBlock(d, d_state=mp.get("d_state", 16), expand=mp.get("expand", 2),
                       use_kernel=mp.get("use_kernel", False))
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self._rng = np.random.default_rng(mp.get("order_seed", 1234))

    def _orders(self, data):
        pts = data.detach().cpu().numpy()
        B, P, _ = pts.shape
        orders = np.empty((B, P), dtype=np.int64)
        for b in range(B):
            orders[b] = serialize_order(pts[b], bits=self.bits,
                                        mode=self.order_mode, rng=self._rng)
        order = torch.from_numpy(orders).to(data.device)
        inv = torch.argsort(order, dim=1)
        return order, inv

    def forward(self, data):
        # data: (B, P, 2)  ->  (B, P, embedding)
        order, inv = self._orders(data)
        h = self.embedding(data)
        idx = order.unsqueeze(-1).expand(-1, -1, h.size(-1))
        h_ser = torch.gather(h, 1, idx)
        for blk, nrm in zip(self.blocks, self.norms):
            h_ser = h_ser + blk(nrm(h_ser))          # pre-norm residual
        inv_idx = inv.unsqueeze(-1).expand(-1, -1, h_ser.size(-1))
        return torch.gather(h_ser, 1, inv_idx)       # back to original node order


def build_model(encoder="attention", **mp):
    """POMO TSPModel with the encoder optionally swapped for Mamba. Decoder untouched."""
    from TSPModel import TSPModel
    model = TSPModel(**mp)
    if encoder == "mamba":
        model.encoder = MambaEncoder(**mp)
    elif encoder != "attention":
        raise ValueError(f"unknown encoder: {encoder}")
    return model
