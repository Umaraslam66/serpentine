#!/usr/bin/env python3
"""Hilbert-serialized Mamba encoder (the Gate-0 candidate).

A faithful pure-PyTorch Mamba-1 block (in_proj -> causal conv -> selective SSM -> gated
out_proj), with an optional mamba-ssm CUDA kernel for the scan (mechanism-gated against
the pure path). The encoder embeds, serializes via the unit-tested curve ordering, runs
the Mamba stack, then un-serializes back to original node order so the shared POMO
decoder indexes nodes correctly. Model assembly lives in serpentine.model.build_model.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from serpentine.serialization import serialize_order


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


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba mixer: a forward scan + a backward scan over the SAME sequence.

    Vanilla MambaBlock is strictly causal (diagnostic C measured exactly zero upstream
    sensitivity), so each node only sees its sequence-predecessors. This block adds a
    second mixer that runs on the reversed sequence (then re-reverses), so a node also
    integrates its successors. The two are summed (default, parameter-clean: output stays
    d_model) or concatenated then projected back to d_model.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_kernel=False,
                 combine="sum"):
        super().__init__()
        self.combine = combine
        self.fwd = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                              use_kernel=use_kernel)
        self.bwd = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                              use_kernel=use_kernel)
        if combine == "concat":
            self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        elif combine != "sum":
            raise ValueError(f"unknown combine: {combine}")

    def forward(self, x):
        # x: (B, L, d_model), already in serialization order
        yf = self.fwd(x)
        yb = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        if self.combine == "sum":
            return yf + yb
        return self.proj(torch.cat([yf, yb], dim=-1))


class MeanPoolGlobal(nn.Module):
    """ECO-style global channel: a single graph-mean vector, projected, broadcast to all nodes.

    Cheapest global mechanism (one pooled vector for the whole graph). A uniform per-node add
    only reaches the routing decision through the decoder's query path (it cancels in the
    node-selection softmax otherwise) — i.e. it can modulate the shared context but not
    differentiate nodes. signature matches SegmentGlobal so the encoder calls them uniformly.
    """

    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, H, order, inv):           # order/inv unused (global mean)
        g = self.proj(H.mean(dim=1, keepdim=True))      # (B,1,d)
        return g.expand(-1, H.size(1), -1)              # (B,P,d)


class SegmentGlobal(nn.Module):
    """Structured global channel (the contribution): chunk the Hilbert line into S contiguous
    segments, mean-pool each, run cheap full attention (O(S^2)) over the S summaries, then
    scatter each summary back to the nodes of its segment.

    Unlike mean-pool this is PER-NODE, so it repairs the ~14% of spatial neighbours the Hilbert
    curve scatters >10 positions apart (Task-1 geometry): two physically-adjacent nodes in
    different segments can now exchange information in one segment-attention hop. The pooled
    summary is coarse (necessary, not proven sufficient — that is what the RL run tests).
    """

    def __init__(self, d, n_segments=10, attn_dim=64):
        super().__init__()
        self.S = n_segments
        self.scale = attn_dim ** -0.5
        self.q = nn.Linear(d, attn_dim, bias=False)
        self.k = nn.Linear(d, attn_dim, bias=False)
        self.v = nn.Linear(d, attn_dim, bias=False)
        self.o = nn.Linear(attn_dim, d, bias=False)

    def _node_segment(self, order, inv):
        P = inv.size(1)
        return (inv * self.S) // P                       # (B,P) node -> segment 0..S-1

    def _pool(self, H, order):
        B, P, d = H.shape
        H_seq = torch.gather(H, 1, order.unsqueeze(-1).expand(-1, -1, d))  # Hilbert order
        return H_seq.view(B, self.S, P // self.S, d).mean(dim=2)           # (B,S,d)

    def _attn(self, seg):
        scores = (self.q(seg) @ self.k(seg).transpose(1, 2)) * self.scale  # (B,S,S)
        return self.o(torch.softmax(scores, dim=-1) @ self.v(seg))         # (B,S,d)

    def _scatter(self, seg_ctx, node_seg):
        idx = node_seg.unsqueeze(-1).expand(-1, -1, seg_ctx.size(-1))
        return torch.gather(seg_ctx, 1, idx)             # (B,P,d) each node <- its segment ctx

    def forward(self, H, order, inv):
        seg_ctx = self._attn(self._pool(H, order))
        return self._scatter(seg_ctx, self._node_segment(order, inv))


class MambaEncoder(nn.Module):
    """Embed -> serialize (Hilbert/sort/random) -> Mamba stack -> un-serialize -> (+global channel).

    Output is in ORIGINAL node order so the POMO decoder indexes nodes correctly. An optional
    global channel (global_mode = none|mean|segment) adds a per-node global contribution to the
    un-serialized embeddings; the decoder is untouched, so 0/A/B differ ONLY in this channel.
    """

    block_cls = MambaBlock

    def __init__(self, **mp):
        super().__init__()
        d = mp["embedding_dim"]
        n_layers = mp["encoder_layer_num"]
        self.order_mode = mp.get("order_mode", "hilbert")
        self.bits = mp.get("hilbert_bits", 7)
        self.embedding = nn.Linear(2, d)
        self.blocks = nn.ModuleList([self._make_block(d, mp) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.global_channel = self._make_global(d, mp)
        self._rng = np.random.default_rng(mp.get("order_seed", 1234))

    def _make_block(self, d, mp):
        return MambaBlock(d, d_state=mp.get("d_state", 16), expand=mp.get("expand", 2),
                          use_kernel=mp.get("use_kernel", False))

    def _make_global(self, d, mp):
        mode = mp.get("global_mode", "none")
        if mode == "none":
            return None
        if mode == "mean":
            return MeanPoolGlobal(d)
        if mode == "segment":
            return SegmentGlobal(d, n_segments=mp.get("n_segments", 10),
                                 attn_dim=mp.get("global_attn_dim", 64))
        raise ValueError(f"unknown global_mode: {mode}")

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
        H = torch.gather(h_ser, 1, inv_idx)          # back to original node order
        if self.global_channel is not None:
            H = H + self.global_channel(H, order, inv)
        return H


class BiMambaEncoder(MambaEncoder):
    """MambaEncoder with bidirectional blocks (forward+backward scan per layer).

    Identical serialize / un-serialize / pre-norm-residual structure as MambaEncoder, so
    everything downstream (decoder, RL, budget) is unchanged. Use ~half the layers of the
    unidirectional encoder to keep total parameters matched (each layer holds two mixers).
    """

    def _make_block(self, d, mp):
        return BiMambaBlock(d, d_state=mp.get("d_state", 16), expand=mp.get("expand", 2),
                            use_kernel=mp.get("use_kernel", False),
                            combine=mp.get("bi_combine", "sum"))
