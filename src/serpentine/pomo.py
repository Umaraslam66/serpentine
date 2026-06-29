"""Locate and import the pinned POMO TSP code (yd-kwon/POMO @ d7c3d6e).

POMO is an external MIT-licensed dependency (env, attention encoder, pointer decoder,
multi-start forward), kept pristine rather than vendored. Set POMO_ROOT to the clone's
NEW_py_ver/TSP directory, or place the clone at ./POMO. scripts/env_setup.sh clones it
and exports POMO_ROOT.
"""
import os
import sys


def pomo_tsp_root():
    env = os.environ.get("POMO_ROOT")
    candidates = [env] if env else []
    candidates.append(os.path.join(os.getcwd(), "POMO", "NEW_py_ver", "TSP"))
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise RuntimeError(
        "POMO not found. Set POMO_ROOT=<clone>/NEW_py_ver/TSP "
        "(git clone https://github.com/yd-kwon/POMO, checkout d7c3d6e)."
    )


def ensure_pomo_on_path():
    root = pomo_tsp_root()
    for p in (os.path.join(root, "POMO"), root):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
