#!/usr/bin/env python3
"""Tests for the Jamba-style hybrid encoder (BiMamba stack + ONE interleaved attention layer).

Diagnosis motivation: BiMamba closes most of the gap to full attention but leaves a residual
deficit that cheap pooled global channels do not fix — the missing ingredient is exact
all-pairs context. HybridEncoder supplies it with a SINGLE POMO-native attention EncoderLayer
inserted mid-stack (after ceil(N_BI/2) bimamba layers), reusing the baseline's own attention
block (not a reimplementation). The attention layer is permutation-equivariant and carries no
positional encoding, so it runs INSIDE the serialized stream — no extra un/re-serialization.

We pin: forward shape, total params matched to the 10-layer uni-mamba anchor (within 5%), the
same end-to-end permutation-equivariance contract as the other encoders, and that gradients
reach BOTH the attention layer and every mamba layer.

Self-contained runner (needs torch + POMO): `python3 tests/test_hybrid.py`.
"""
import torch

from serpentine.encoders.mamba import BiMambaBlock, HybridEncoder
from serpentine.model import build_model

MP = dict(embedding_dim=128, sqrt_embedding_dim=128 ** 0.5, encoder_layer_num=4,
          qkv_dim=16, head_num=8, logit_clipping=10, ff_hidden_dim=512,
          eval_type="argmax")

ANCHOR = 1_249_792   # 10-layer unidirectional mamba total (the param budget)


def test_hybrid_encoder_shape():
    enc = HybridEncoder(order_mode="hilbert", **MP)
    assert enc(torch.rand(4, 100, 2)).shape == (4, 100, 128)


def test_hybrid_has_exactly_one_attention_layer_mid_stack():
    # Exactly one attention layer, inserted after ceil(N_BI/2) bimamba layers (here 4 -> pos 2).
    enc = HybridEncoder(order_mode="hilbert", **MP)
    assert len(enc.blocks) == 4
    assert all(isinstance(b, BiMambaBlock) for b in enc.blocks)
    assert enc.attn_pos == 2


def test_hybrid_total_params_within_5pct_of_anchor():
    hy = build_model("hybrid", **{**MP, "order_mode": "hilbert"})
    ph = sum(p.numel() for p in hy.parameters())
    assert abs(ph - ANCHOR) / ANCHOR < 0.05, (ph, ANCHOR)


def test_build_model_hybrid_keeps_decoder():
    att = build_model("attention", **MP)
    hy = build_model("hybrid", **{**MP, "order_mode": "hilbert"})
    from serpentine.decoder import TSP_Decoder
    assert isinstance(hy.decoder, TSP_Decoder)
    dp_att = sum(p.numel() for p in att.decoder.parameters())
    dp_hy = sum(p.numel() for p in hy.decoder.parameters())
    assert dp_att == dp_hy, (dp_att, dp_hy)


def test_hybrid_encoder_permutation_equivariant():
    # End-to-end node identity: permuting input nodes permutes the per-node output identically.
    torch.manual_seed(0)
    enc = HybridEncoder(**{**MP, "encoder_layer_num": 4, "order_mode": "hilbert",
                           "order_seed": 7, "hilbert_bits": 16}).eval()  # high bits => no ties
    data = torch.rand(1, 100, 2)
    perm = torch.randperm(100)
    with torch.no_grad():
        out = enc(data)
        out_perm = enc(data[:, perm, :])
    assert torch.allclose(out[:, perm, :], out_perm, atol=1e-5)


def test_hybrid_gradients_reach_attention_and_every_mamba_layer():
    # A backward pass must produce a finite, non-zero gradient for the attention layer AND
    # for at least one parameter of every bimamba block (nothing silently detached).
    torch.manual_seed(0)
    enc = HybridEncoder(order_mode="hilbert", **MP)
    out = enc(torch.rand(2, 100, 2))
    out.pow(2).sum().backward()

    for p in enc.parameters():
        assert p.grad is not None                       # nothing detached from the graph

    def gnorm(module):
        return sum(p.grad.norm().item() for p in module.parameters() if p.grad is not None)

    assert gnorm(enc.attn) > 0                           # attention layer trains
    for blk in enc.blocks:
        assert gnorm(blk) > 0                            # every mamba layer trains


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
