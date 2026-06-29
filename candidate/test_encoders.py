#!/usr/bin/env python3
"""Torch-side tests for the pluggable encoders (run on Leonardo: needs torch + POMO).

Critical correctness check: the serialize -> Mamba -> un-serialize path must return
per-node embeddings in ORIGINAL node order (the decoder indexes nodes by original
index). A bug here would silently scramble nodes -> "verify the mechanism, not the metric".
Self-contained runner: `python3 test_encoders.py`.
"""
import numpy as np
import torch

from encoders import MambaBlock, MambaEncoder, build_model

MP = dict(embedding_dim=128, sqrt_embedding_dim=128 ** 0.5, encoder_layer_num=6,
          qkv_dim=16, head_num=8, logit_clipping=10, ff_hidden_dim=512,
          eval_type="argmax")


def test_mamba_block_shape():
    blk = MambaBlock(128)
    x = torch.randn(4, 100, 128)
    assert blk(x).shape == (4, 100, 128)


def test_mamba_encoder_shape():
    enc = MambaEncoder(order_mode="hilbert", **{**MP, "encoder_layer_num": 10})
    out = enc(torch.rand(4, 100, 2))
    assert out.shape == (4, 100, 128)


def test_encoder_unsort_preserves_node_identity():
    enc = MambaEncoder(order_mode="hilbert", **{**MP, "encoder_layer_num": 2})
    data = torch.rand(3, 100, 2)
    order, inv = enc._orders(data)
    h = enc.embedding(data)
    h_ser = torch.gather(h, 1, order.unsqueeze(-1).expand(-1, -1, 128))
    h_back = torch.gather(h_ser, 1, inv.unsqueeze(-1).expand(-1, -1, 128))
    assert torch.allclose(h_back, h, atol=1e-6)


def test_orders_all_modes_are_permutations():
    for mode in ("hilbert", "sort", "random"):
        enc = MambaEncoder(order_mode=mode, **{**MP, "encoder_layer_num": 1})
        order, _ = enc._orders(torch.rand(2, 100, 2))
        for b in range(2):
            assert sorted(order[b].tolist()) == list(range(100)), mode


def test_build_model_swaps_encoder_keeps_decoder():
    att = build_model("attention", **MP)
    mam = build_model("mamba", **{**MP, "encoder_layer_num": 10, "order_mode": "hilbert"})
    from TSPModel import TSP_Decoder
    assert isinstance(att.decoder, TSP_Decoder) and isinstance(mam.decoder, TSP_Decoder)
    dp_att = sum(p.numel() for p in att.decoder.parameters())
    dp_mam = sum(p.numel() for p in mam.decoder.parameters())
    assert dp_att == dp_mam, (dp_att, dp_mam)


def test_param_counts_matched_within_5pct():
    att = build_model("attention", **MP)
    mam = build_model("mamba", **{**MP, "encoder_layer_num": 10, "order_mode": "hilbert"})
    pa = sum(p.numel() for p in att.parameters())
    pm = sum(p.numel() for p in mam.parameters())
    assert abs(pa - pm) / pa < 0.05, (pa, pm)


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
