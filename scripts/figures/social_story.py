"""IMAGE 3 - three-act narrative card (mostly typographic, LinkedIn-friendly).

ACT 1  The kill: vanilla Hilbert+Mamba 7.50 vs attention 2.95
ACT 2  The fix that half-worked: bidirectional scan reaches multistart parity 1.94 vs 1.91
ACT 3  The win: + one attention layer beats attention 1.14 vs 1.61
Footer: github.com/Umaraslam66/serpentine
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from _social_style import (
    apply_rc, DPI, FIGSIZE,
    HYBRID, BIMAMBA, ATTENTION, SURFACE, PLANE, INK, SECOND, MUTED,
)

OUT = "assets/social/social_story.png"
PXW, PXH = FIGSIZE[0] * DPI, FIGSIZE[1] * DPI  # 3200 x 1800
RED = "#c0392b"  # "kill" accent, only on the failure act

# act, title, accent, metric context, big number, comparator text, tag
ACTS = [
    ("ACT 1", "The kill", RED,
     "vanilla Hilbert + Mamba, single-trajectory gap at 250k",
     "7.50", "vs attention 2.95", "uni-directional scan falls far short"),
    ("ACT 2", "The fix that half-worked", BIMAMBA,
     "bidirectional scan, multistart gap at 250k",
     "1.94", "vs attention 1.91", "reaches parity, not a win"),
    ("ACT 3", "The win", HYBRID,
     "plus one attention layer, multistart gap at 500k",
     "1.14", "vs attention 1.61", "80% Mamba beats full attention"),
]

_fig = None
_renderer = None


def wrap(text, fontsize, weight, max_px):
    """Greedy word-wrap using real rendered pixel widths."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        t = _fig.text(0, 0, trial, fontsize=fontsize, fontweight=weight)
        wpx = t.get_window_extent(_renderer).width
        t.remove()
        if wpx > max_px and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def main():
    global _fig, _renderer
    apply_rc()
    _fig = fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(PLANE)
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()

    # header
    fig.text(0.055, 0.955, "How an 80% Mamba encoder learned to route",
             fontsize=23, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.055, 0.858,
             "three acts on TSP-100  ·  POMO RL  ·  matched params and budget  ·  gap vs LKH-3 optimal",
             fontsize=12, color=SECOND, va="top", ha="left")

    col_x = [0.055, 0.373, 0.691]
    col_w = 0.272
    card_y = 0.150
    card_h = 0.585
    pad = 0.024
    inner_px = (col_w - 2 * pad) * PXW  # usable text width in px

    aspect = FIGSIZE[0] / FIGSIZE[1]

    for (act, title, accent, ctx, num, comp, tag), x in zip(ACTS, col_x):
        # card + top accent rule
        fig.add_artist(FancyBboxPatch(
            (x, card_y), col_w, card_h,
            boxstyle="round,pad=0,rounding_size=0.012", mutation_aspect=aspect,
            linewidth=0, facecolor=SURFACE, zorder=1,
            transform=fig.transFigure, clip_on=False))
        fig.add_artist(FancyBboxPatch(
            (x, card_y + card_h - 0.013), col_w, 0.013,
            boxstyle="round,pad=0,rounding_size=0.005", mutation_aspect=aspect,
            linewidth=0, facecolor=accent, zorder=2,
            transform=fig.transFigure, clip_on=False))

        ix = x + pad
        top = card_y + card_h - 0.058

        fig.text(ix, top, act, fontsize=13, fontweight="bold",
                 color=accent, va="top", ha="left")

        # wrapped title (up to 2 lines)
        y = top - 0.050
        for ln in wrap(title, 18, "bold", inner_px):
            fig.text(ix, y, ln, fontsize=18, fontweight="bold",
                     color=INK, va="top", ha="left")
            y -= 0.050

        # wrapped context
        y -= 0.014
        for ln in wrap(ctx, 11, "normal", inner_px):
            fig.text(ix, y, ln, fontsize=11, color=SECOND, va="top", ha="left")
            y -= 0.036

        # big number (fixed lower zone), % attached
        fig.text(ix, card_y + 0.205, f"{num}%", fontsize=46, fontweight="bold",
                 color=accent, va="center", ha="left")
        fig.text(ix, card_y + 0.108, comp, fontsize=13, fontweight="bold",
                 color=INK, va="center", ha="left")
        fig.text(ix, card_y + 0.058, tag, fontsize=10.5, color=MUTED,
                 va="center", ha="left")

    # connective chevrons between cards
    for x in (col_x[0] + col_w, col_x[1] + col_w):
        fig.text(x + 0.023, card_y + card_h / 2, "›",  # single ›
                 fontsize=34, color=MUTED, va="center", ha="center")

    # footer
    fig.text(0.055, 0.070, "github.com/Umaraslam66/serpentine",
             fontsize=13, fontweight="bold", color=HYBRID, va="center", ha="left")
    fig.text(0.945, 0.070, "One attention layer is enough",
             fontsize=12, color=SECOND, va="center", ha="right")

    fig.savefig(OUT, dpi=DPI, facecolor=PLANE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
