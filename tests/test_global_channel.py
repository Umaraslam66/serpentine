#!/usr/bin/env python3
"""Tests for the encoder global-channel variants (ECO-style mean-pool vs structured segment).

The reviewer-required check: the segment pool + scatter must PRESERVE NODE IDENTITY — every
node receives the summary of the Hilbert segment it actually sits in, never a neighbour's
(the analogue of the Hilbert un-serialize round-trip test). We pin that three ways: the
scatter index mapping, the segment assignment vs Hilbert position, and end-to-end
permutation-equivariance of the whole encoder with the structured channel.

Self-contained runner (needs torch + POMO): `python3 tests/test_global_channel.py`.
"""
import torch

from serpentine.encoders.mamba import MambaEncoder, MeanPoolGlobal, SegmentGlobal
from serpentine.model import build_model

MP = dict(embedding_dim=128, sqrt_embedding_dim=128 ** 0.5, encoder_layer_num=10,
          qkv_dim=16, head_num=8, logit_clipping=10, ff_hidden_dim=512,
          eval_type="argmax", order_mode="hilbert")


def test_meanpool_shape_and_uniform():
    ch = MeanPoolGlobal(16)
    H = torch.randn(2, 50, 16)
    g = ch(H, None, None)
    assert g.shape == (2, 50, 16)
    assert torch.allclose(g[:, 0], g[:, 7])          # same global vector for every node


def test_segment_channel_shape():
    ch = SegmentGlobal(128, n_segments=10)
    enc = MambaEncoder(**{**MP, "encoder_layer_num": 1})
    data = torch.rand(3, 100, 2)
    order, inv = enc._orders(data)
    g = ch(torch.randn(3, 100, 128), order, inv)
    assert g.shape == (3, 100, 128)


def test_segment_scatter_picks_correct_segment():
    # _scatter(seg_ctx, node_seg)[b,n] must equal seg_ctx[b, node_seg[b,n]] (no scrambling).
    ch = SegmentGlobal(8, n_segments=10)
    B, P, S, d = 2, 100, 10, 8
    seg_ctx = torch.arange(S, dtype=torch.float32).view(1, S, 1).expand(B, S, d).contiguous()
    node_seg = torch.randint(0, S, (B, P))
    out = ch._scatter(seg_ctx, node_seg)
    assert torch.equal(out[..., 0].long(), node_seg)            # row carries its segment id
    assert torch.allclose(out, out[..., :1].expand(-1, -1, d))  # constant across the d dims


def test_node_segment_matches_hilbert_position():
    ch = SegmentGlobal(128, n_segments=10)
    enc = MambaEncoder(**{**MP, "encoder_layer_num": 1})
    data = torch.rand(3, 100, 2)
    order, inv = enc._orders(data)
    P, S = 100, 10
    assert torch.equal(ch._node_segment(order, inv), (inv * S) // P)


def test_segment_pool_is_contiguous_hilbert_mean():
    ch = SegmentGlobal(4, n_segments=10)
    enc = MambaEncoder(**{**MP, "encoder_layer_num": 1})
    data = torch.rand(2, 100, 2)
    order, inv = enc._orders(data)
    H = torch.randn(2, 100, 4)
    seg = ch._pool(H, order)
    assert seg.shape == (2, 10, 4)
    H_seq = torch.gather(H, 1, order.unsqueeze(-1).expand(-1, -1, 4))
    assert torch.allclose(seg[:, 0], H_seq[:, :10].mean(1), atol=1e-6)   # seg 0 = first 10 on line


def test_global_encoder_permutation_equivariant():
    # End-to-end node identity: permuting input nodes permutes the per-node output identically.
    torch.manual_seed(0)
    enc = MambaEncoder(**{**MP, "encoder_layer_num": 2, "global_mode": "segment",
                          "hilbert_bits": 16}).eval()           # high bits => no quantization ties
    data = torch.rand(1, 100, 2)
    perm = torch.randperm(100)
    with torch.no_grad():
        out = enc(data)
        out_perm = enc(data[:, perm, :])
    assert torch.allclose(out[:, perm, :], out_perm, atol=1e-5)


def test_baseline_unchanged_when_global_none():
    enc = MambaEncoder(**{**MP, "encoder_layer_num": 2})
    assert enc.global_channel is None                            # default = exact KILL baseline


def test_build_global_variants_param_counts_within_5pct():
    base = build_model("mamba", **MP)                            # variant 0
    a = build_model("mamba", **{**MP, "global_mode": "mean"})    # variant A
    b = build_model("mamba", **{**MP, "global_mode": "segment"}) # variant B
    p0 = sum(p.numel() for p in base.parameters())
    pa = sum(p.numel() for p in a.parameters())
    pb = sum(p.numel() for p in b.parameters())
    assert pa > p0 and pb > p0
    assert abs(pa - p0) / p0 < 0.05, (pa, p0)
    assert abs(pb - p0) / p0 < 0.05, (pb, p0)


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
