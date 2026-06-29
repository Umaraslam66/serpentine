"""Baseline attention encoder = POMO's 6-layer self-attention stack (MIT, unchanged).

This is the reference encoder the Hilbert+Mamba candidate must match at N=100.
"""
from serpentine.pomo import ensure_pomo_on_path

ensure_pomo_on_path()
from TSPModel import TSP_Encoder  # noqa: E402

__all__ = ["TSP_Encoder"]
