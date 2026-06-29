#!/usr/bin/env python3
"""Hilbert-curve serialization of 2D point sets.

Lifts the space-filling-curve recipe from PointMamba / Point Transformer V3: quantize
points to a 2^bits grid, map each cell to its Hilbert-curve distance, sort by it.
The unified `serialize_order` dispatcher lives in the package __init__.
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


def quantize(points, bits):
    """Map float points to integer grid cells in [0, 2^bits)."""
    points = np.asarray(points, dtype=np.float64)
    n = 1 << bits
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    span = np.maximum(mx - mn, 1e-9)
    cells = np.floor((points - mn) / span * (n - 1e-9)).astype(np.int64)
    return np.clip(cells, 0, n - 1)


def hilbert_order(points, bits=7):
    """Permutation ordering points along the Hilbert curve (locality-preserving)."""
    cells = quantize(points, bits)
    return np.argsort(hilbert_index(cells, bits), kind="stable")
