"""Serialization of unordered 2D point sets into 1D sequences.

`serialize_order(points, bits, mode, rng)` returns a permutation. Modes:
  hilbert - Hilbert space-filling curve (locality-preserving; primary)
  zorder  - Z-order / Morton curve (alternative SFC)
  sort    - lexicographic by (x, then y)  [coordinate-sort ablation]
  random  - random permutation            [order-destroying ablation]
"""
import numpy as np

from serpentine.serialization.hilbert import hilbert_index, hilbert_order, quantize
from serpentine.serialization.zorder import zorder_index, zorder_order

__all__ = [
    "serialize_order", "hilbert_index", "hilbert_order",
    "zorder_index", "zorder_order", "quantize",
]


def serialize_order(points, bits=7, mode="hilbert", rng=None):
    points = np.asarray(points, dtype=np.float64)
    if mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        return rng.permutation(len(points))
    if mode == "sort":
        return np.lexsort((points[:, 1], points[:, 0]))
    if mode == "hilbert":
        return hilbert_order(points, bits)
    if mode == "zorder":
        return zorder_order(points, bits)
    raise ValueError(f"unknown mode: {mode}")
