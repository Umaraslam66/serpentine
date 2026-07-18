"""Assemble the routing model: POMO TSPModel with a pluggable encoder.

build_model reuses POMO's TSPModel (encoder + decoder + multi-start forward) and swaps
ONLY the encoder attribute for the Mamba encoder, so the decoder, forward, and RL
interface stay byte-identical between baseline and candidate.
"""
from serpentine.encoders.mamba import BiMambaEncoder, HybridEncoder, MambaEncoder
from serpentine.pomo import ensure_pomo_on_path


def build_model(encoder="attention", **mp):
    ensure_pomo_on_path()
    from TSPModel import TSPModel
    model = TSPModel(**mp)
    if encoder == "mamba":
        model.encoder = MambaEncoder(**mp)
    elif encoder == "bimamba":
        model.encoder = BiMambaEncoder(**mp)
    elif encoder == "hybrid":
        model.encoder = HybridEncoder(**mp)
    elif encoder != "attention":
        raise ValueError(f"unknown encoder: {encoder}")
    return model
