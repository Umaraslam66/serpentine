"""Routing decoder = POMO's pointer-attention decoder (MIT, unchanged).

Shared identically by baseline and candidate so only the encoder differs.
"""
from serpentine.pomo import ensure_pomo_on_path

ensure_pomo_on_path()
from TSPModel import TSP_Decoder  # noqa: E402

__all__ = ["TSP_Decoder"]
