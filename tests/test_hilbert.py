#!/usr/bin/env python3
"""Unit tests for Hilbert serialization (reviewer-mandated, BEFORE any training).

Self-contained runner (numpy only): `python3 test_hilbert.py`.
The defining correctness property we lean on: consecutive indices along a Hilbert
curve are 4-adjacent grid cells. If that holds on full grids, the index map is a
genuine Hilbert curve, not an accidental space-filling-ish ordering.
"""
import numpy as np

from serpentine.serialization import hilbert_index, serialize_order


def test_xy2d_order1_canonical_sequence():
    # 2x2 grid (1 bit): the Hilbert curve visits (0,0)->(0,1)->(1,1)->(1,0).
    cells = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    d = hilbert_index(cells, bits=1)
    assert list(d) == [0, 1, 2, 3], list(d)


def test_hilbert_index_bijection():
    # Over a full 2^bits x 2^bits grid, the index is a bijection onto [0, n^2).
    for bits in (2, 3, 4):
        n = 1 << bits
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cells = np.stack([xs.ravel(), ys.ravel()], axis=1)
        d = hilbert_index(cells, bits)
        assert sorted(d.tolist()) == list(range(n * n)), bits


def test_hilbert_consecutive_cells_are_adjacent():
    # THE defining property: successive curve positions are Manhattan-distance 1.
    for bits in (3, 4, 5):
        n = 1 << bits
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cells = np.stack([xs.ravel(), ys.ravel()], axis=1)
        d = hilbert_index(cells, bits)
        seq = cells[np.argsort(d)]
        steps = np.abs(np.diff(seq, axis=0)).sum(axis=1)
        assert np.all(steps == 1), (bits, steps[steps != 1][:5])


def test_serialize_order_is_permutation():
    pts = np.random.default_rng(0).random((50, 2))
    for mode in ("hilbert", "sort", "random"):
        order = serialize_order(pts, bits=7, mode=mode, rng=np.random.default_rng(3))
        assert sorted(order.tolist()) == list(range(50)), mode


def test_serialize_hilbert_far_more_local_than_random():
    # Hilbert path through the points must be much shorter than a random path.
    pts = np.random.default_rng(1).random((300, 2))

    def path_len(o):
        seq = pts[o]
        return np.sqrt((np.diff(seq, axis=0) ** 2).sum(1)).sum()

    h = path_len(serialize_order(pts, bits=7, mode="hilbert"))
    r = path_len(serialize_order(pts, bits=7, mode="random", rng=np.random.default_rng(2)))
    assert h < 0.5 * r, (h, r)


def test_serialize_sort_is_lexicographic_x_then_y():
    pts = np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.1], [0.9, 0.9]])
    order = serialize_order(pts, bits=7, mode="sort")
    expected = np.lexsort((pts[:, 1], pts[:, 0]))
    assert list(order) == list(expected), list(order)


def test_serialize_random_is_deterministic_given_seed():
    pts = np.random.default_rng(0).random((30, 2))
    o1 = serialize_order(pts, bits=7, mode="random", rng=np.random.default_rng(5))
    o2 = serialize_order(pts, bits=7, mode="random", rng=np.random.default_rng(5))
    assert list(o1) == list(o2)


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
