#!/usr/bin/env python3
"""Unit tests for Z-order (Morton) serialization. Runner: `python3 tests/test_zorder.py`."""
import numpy as np

from serpentine.serialization import serialize_order
from serpentine.serialization.zorder import zorder_index


def test_zorder_index_canonical_2x2():
    # Morton interleave (y high bit): (0,0)=0,(1,0)=1,(0,1)=2,(1,1)=3
    cells = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    assert list(zorder_index(cells, bits=1)) == [0, 1, 2, 3], list(zorder_index(cells, 1))


def test_zorder_index_bijection():
    for bits in (2, 3, 4):
        n = 1 << bits
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cells = np.stack([xs.ravel(), ys.ravel()], axis=1)
        d = zorder_index(cells, bits)
        assert sorted(d.tolist()) == list(range(n * n)), bits


def test_serialize_zorder_is_permutation():
    pts = np.random.default_rng(0).random((60, 2))
    order = serialize_order(pts, bits=7, mode="zorder")
    assert sorted(order.tolist()) == list(range(60))


def test_serialize_zorder_more_local_than_random():
    pts = np.random.default_rng(1).random((300, 2))

    def path_len(o):
        seq = pts[o]
        return np.sqrt((np.diff(seq, axis=0) ** 2).sum(1)).sum()

    z = path_len(serialize_order(pts, bits=7, mode="zorder"))
    r = path_len(serialize_order(pts, bits=7, mode="random", rng=np.random.default_rng(2)))
    assert z < 0.6 * r, (z, r)


if __name__ == "__main__":
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
