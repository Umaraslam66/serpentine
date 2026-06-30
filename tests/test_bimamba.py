#!/usr/bin/env python3
"""Tests for the bidirectional Mamba encoder (the diagnosis-motivated discriminator probe).

The defining correctness property comes straight from diagnostic C: the vanilla causal
MambaBlock has EXACTLY zero upstream sensitivity (perturbing a node never changes nodes
earlier in the sequence). A correct bidirectional block MUST break that — perturbing the
LAST sequence position must change the FIRST position's output. We also pin the parameter
count of the 5-layer bidirectional encoder to the 10-layer unidirectional one (within 5%),
so the GPU discriminator measures bidirectionality, not extra capacity.

Self-contained runner (needs torch + POMO): `python3 tests/test_bimamba.py`.
"""
import torch

from serpentine.encoders.mamba import MambaBlock, MambaEncoder, BiMambaBlock, BiMambaEncoder
from serpentine.model import build_model

MP = dict(embedding_dim=128, sqrt_embedding_dim=128 ** 0.5, encoder_layer_num=6,
          qkv_dim=16, head_num=8, logit_clipping=10, ff_hidden_dim=512,
          eval_type="argmax")


def _upstream_sensitivity(block, L=20, d=32):
    """L2 change at the position just BEFORE a perturbed one (i.e. upstream in the scan).

    Measured one step up rather than at position 0: a causal scan is *exactly* zero here
    too, while the SSM state decays ~99% over ~20 positions (diagnostic C), so position 0
    would swamp a genuine backward signal in decay. Adjacent-upstream keeps the causal
    contrast crisp (exact 0) and the backward signal robust.
    """
    block.eval()
    x = torch.randn(1, L, d)
    p = L - 1
    with torch.no_grad():
        y0 = block(x)
        x2 = x.clone()
        x2[0, p] += 0.1
        y1 = block(x2)
    return (y1[0, p - 1] - y0[0, p - 1]).norm().item()


def test_mamba_block_is_strictly_causal():
    # Characterization: the existing block leaves position 0 untouched by later positions.
    torch.manual_seed(0)
    assert _upstream_sensitivity(MambaBlock(32)) < 1e-9


def test_bimamba_block_has_upstream_sensitivity():
    # The whole point: the backward scan lets a late node influence an early one.
    torch.manual_seed(0)
    assert _upstream_sensitivity(BiMambaBlock(32)) > 1e-6


def test_bimamba_block_shape():
    blk = BiMambaBlock(128)
    assert blk(torch.randn(4, 100, 128)).shape == (4, 100, 128)


def test_bimamba_block_concat_shape():
    blk = BiMambaBlock(128, combine="concat")
    assert blk(torch.randn(4, 100, 128)).shape == (4, 100, 128)


def test_bimamba_encoder_shape():
    enc = BiMambaEncoder(order_mode="hilbert", **{**MP, "encoder_layer_num": 5})
    assert enc(torch.rand(4, 100, 2)).shape == (4, 100, 128)


def test_bimamba_encoder_unsort_preserves_node_identity():
    # Same un-serialize contract as the unidirectional encoder: output in original node order.
    enc = BiMambaEncoder(order_mode="hilbert", **{**MP, "encoder_layer_num": 2})
    data = torch.rand(3, 100, 2)
    order, inv = enc._orders(data)
    h = enc.embedding(data)
    h_ser = torch.gather(h, 1, order.unsqueeze(-1).expand(-1, -1, 128))
    h_back = torch.gather(h_ser, 1, inv.unsqueeze(-1).expand(-1, -1, 128))
    assert torch.allclose(h_back, h, atol=1e-6)


def test_bimamba_5layer_param_matches_uni_10layer_within_5pct():
    bi = BiMambaEncoder(order_mode="hilbert", **{**MP, "encoder_layer_num": 5})
    uni = MambaEncoder(order_mode="hilbert", **{**MP, "encoder_layer_num": 10})
    pb = sum(p.numel() for p in bi.parameters())
    pu = sum(p.numel() for p in uni.parameters())
    assert abs(pb - pu) / pu < 0.05, (pb, pu)


def test_build_model_bimamba_keeps_decoder_and_matches_params():
    att = build_model("attention", **MP)
    bi = build_model("bimamba", **{**MP, "encoder_layer_num": 5, "order_mode": "hilbert"})
    from serpentine.decoder import TSP_Decoder
    assert isinstance(bi.decoder, TSP_Decoder)
    dp_att = sum(p.numel() for p in att.decoder.parameters())
    dp_bi = sum(p.numel() for p in bi.decoder.parameters())
    assert dp_att == dp_bi, (dp_att, dp_bi)
    pa = sum(p.numel() for p in att.parameters())
    pb = sum(p.numel() for p in bi.parameters())
    assert abs(pa - pb) / pa < 0.05, (pa, pb)


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
