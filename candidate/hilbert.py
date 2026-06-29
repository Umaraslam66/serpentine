#!/usr/bin/env python3
"""Hilbert / random / coordinate-sort serialization of 2D point sets.

Lifts the space-filling-curve recipe used by PointMamba / Point Transformer V3:
quantize points to a 2^bits grid, map each cell to its Hilbert-curve distance,
sort by that distance. numpy core (unit-tested); a torch batched wrapper lives in
encoders.py for GPU use and is cross-checked against this core.
"""
import numpy as np


def hilbert_index(cells, bits):
    """Hilbert-curve distance of integer grid cells.

    cells: (N, 2) int in [0, 2^bits). Returns (N,) int64 distances.
    Vectorized form of the canonical iterative xy2d (Wikipedia).
    """
    cells = np.asarray(cells, dtype=np.int64)
    x = cells[:, 0].copy()
    y = cells[:, 1].copy()
    n = 1 << bits
    d = np.zeros(len(cells), dtype=np.int64)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # rotate quadrant
        ry0 = ry == 0
        flip = ry0 & (rx == 1)
        x[flip] = (n - 1) - x[flip]
        y[flip] = (n - 1) - y[flip]
        xs, ys = x[ry0].copy(), y[ry0].copy()
        x[ry0], y[ry0] = ys, xs
        s >>= 1
    return d


def serialize_order(points, bits=7, mode="hilbert", rng=None):
    """Return a permutation ordering the points along the chosen curve.

    points: (N, 2) floats. modes:
      hilbert - Hilbert-curve order over a 2^bits grid (locality-preserving)
      sort    - lexicographic by (x, then y)  [coordinate-sort ablation]
      random  - random permutation            [order-destroying ablation]
    """
    points = np.asarray(points, dtype=np.float64)
    N = len(points)

    if mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        return rng.permutation(N)

    if mode == "sort":
        return np.lexsort((points[:, 1], points[:, 0]))

    if mode == "hilbert":
        n = 1 << bits
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        span = np.maximum(mx - mn, 1e-9)
        cells = np.floor((points - mn) / span * (n - 1e-9)).astype(np.int64)
        cells = np.clip(cells, 0, n - 1)
        d = hilbert_index(cells, bits)
        return np.argsort(d, kind="stable")

    raise ValueError(f"unknown mode: {mode}")
