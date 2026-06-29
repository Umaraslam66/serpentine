#!/usr/bin/env python3
"""Z-order (Morton) serialization of 2D point sets.

The other space-filling curve used by PointMamba / PTv3. Cheaper than Hilbert and
locality-preserving, but with occasional long jumps across the quadtree boundary
(Hilbert avoids these). Provided as an additional ordering for the ablation.
"""
import numpy as np

from serpentine.serialization.hilbert import quantize


def zorder_index(cells, bits):
    """Morton code of integer grid cells: interleave the bits of x and y.

    cells: (N, 2) int in [0, 2^bits). Returns (N,) int64 codes.
    """
    cells = np.asarray(cells, dtype=np.int64)
    x = cells[:, 0]
    y = cells[:, 1]
    d = np.zeros(len(cells), dtype=np.int64)
    for i in range(bits):
        d |= ((x >> i) & 1) << (2 * i)
        d |= ((y >> i) & 1) << (2 * i + 1)
    return d


def zorder_order(points, bits=7):
    """Permutation ordering points along the Z-order curve."""
    cells = quantize(points, bits)
    return np.argsort(zorder_index(cells, bits), kind="stable")
