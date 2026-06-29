#!/usr/bin/env python3
"""MECHANISM GATE (reviewer-mandated): mamba-ssm kernel == pure-PyTorch reference.

Runs the SAME (u, dt, A, B, C) and the SAME MambaBlock weights through both the
pure-PyTorch selective scan (reference) and the mamba-ssm CUDA kernel; asserts
max abs diff < 1e-3. Mismatch -> exit 1 -> HALT (do not train). Needs a GPU.
"""
import torch
import torch.nn.functional as F

from serpentine.encoders.mamba import MambaBlock

TOL = 1e-3


def main():
    if not torch.cuda.is_available():
        print("GATE FAIL: no CUDA")
        raise SystemExit(1)
    dev = "cuda"
    torch.manual_seed(0)
    b, l, d_model = 2, 100, 128
    blk = MambaBlock(d_model, d_state=16, expand=2).to(dev)
    d, n = blk.d_inner, blk.d_state

    # --- core selective-scan parity on identical inputs ---
    u = torch.randn(b, l, d, device=dev)
    dt = F.softplus(torch.randn(b, l, d, device=dev))
    A = -torch.exp(torch.randn(d, n, device=dev))
    B = torch.randn(b, l, n, device=dev)
    C = torch.randn(b, l, n, device=dev)
    blk.use_kernel = False
    y_ref = blk._selective_scan(u, dt, A, B, C)
    blk.use_kernel = True
    y_ker = blk._selective_scan(u, dt, A, B, C)
    scan_diff = (y_ref - y_ker).abs().max().item()
    print(f"selective_scan  max|Δ| = {scan_diff:.3e}")

    # --- full block forward parity (identical weights) ---
    x = torch.randn(b, l, d_model, device=dev)
    blk.use_kernel = False
    yb_ref = blk(x)
    blk.use_kernel = True
    yb_ker = blk(x)
    blk_diff = (yb_ref - yb_ker).abs().max().item()
    print(f"block.forward   max|Δ| = {blk_diff:.3e}")

    ok = scan_diff < TOL and blk_diff < TOL
    print(f"GATE {'PASS' if ok else 'FAIL'} (tol={TOL})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
