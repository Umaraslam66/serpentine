"""IMAGE 1 - social hero: two-panel training curves (single-traj + multistart).

Headline: One attention layer is enough
Sub: an encoder that is 80% Mamba beats full attention on multistart tour quality
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from _social_style import (
    apply_rc, load_curve, DPI, FIGSIZE,
    HYBRID, BIMAMBA, ATTENTION, SURFACE, PLANE, INK, SECOND, MUTED, GRID, AXISLN,
)

OUT = "assets/social/social_hero.png"

SERIES = [
    # name, model, color, dashed
    ("Attention", "attention", ATTENTION, True),
    ("BiMamba",   "bimamba",   BIMAMBA,   False),
    ("Hybrid",    "hybrid",    HYBRID,    False),
]


def panel(ax, idx, title):
    """idx 0 = single-traj, 1 = multistart."""
    ax.set_facecolor(SURFACE)
    endpoints = []
    for name, model, color, dashed in SERIES:
        steps, single, multi = load_curve(model)
        y = single if idx == 0 else multi
        xs = [s / 1000.0 for s in steps]
        lw = 3.2 if name == "Hybrid" else 2.4
        ax.plot(xs, y, color=color, lw=lw,
                ls=(0, (5, 2.4)) if dashed else "-",
                solid_capstyle="round", dash_capstyle="round",
                zorder=4 if name == "Hybrid" else 3)
        endpoints.append((name, color, xs[-1], y[-1]))

    # gridlines: recessive hairlines
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXISLN)
    ax.spines["bottom"].set_color(AXISLN)

    ax.set_xlim(0, 760)          # gutter past 500 holds the endpoint labels
    ax.set_xticks([0, 100, 250, 500])
    ax.set_xticklabels(["0", "100k", "250k", "500k"], fontsize=9.5, color=MUTED)
    ax.tick_params(length=0)
    ax.set_xlabel("training steps", fontsize=10, color=MUTED, labelpad=5)

    ax.set_title(title, fontsize=14, color=INK, fontweight="bold",
                 loc="left", pad=7)

    # y limit per panel
    if idx == 0:
        ax.set_ylim(2, 11)
        ax.set_yticks([2, 5, 8, 11])
    else:
        ax.set_ylim(0.8, 4.6)
        ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels([f"{t:g}" for t in ax.get_yticks()], fontsize=9.5, color=MUTED)

    # BIG endpoint labels, collision-free vertical nudging
    endpoints.sort(key=lambda e: e[3])
    placed = []
    span = ax.get_ylim()
    minsep = (span[1] - span[0]) * 0.11
    ypos = []
    for _, _, _, yv in endpoints:
        y = yv
        for py in ypos:
            if abs(y - py) < minsep:
                y = py + minsep
        ypos.append(y)
    for (name, color, xv, yv), ytxt in zip(endpoints, ypos):
        ax.plot([xv], [yv], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=6)
        val = f"{yv:.2f}"
        ax.annotate(f"{name}  {val}", xy=(xv, yv), xytext=(xv + 24, ytxt),
                    fontsize=13.5, fontweight="bold", color=color,
                    va="center", ha="left", zorder=7,
                    annotation_clip=False)


def main():
    apply_rc()
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(PLANE)

    # headline (figure fractions, y measured from bottom)
    fig.text(0.055, 0.955, "One attention layer is enough",
             fontsize=30, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.055, 0.815,
             "an encoder that is 80% Mamba beats full attention on multistart tour quality",
             fontsize=14.5, color=SECOND, va="top", ha="left")

    # two panels, explicitly placed so nothing collides
    axL = fig.add_axes([0.055, 0.165, 0.400, 0.545])
    axR = fig.add_axes([0.560, 0.165, 0.400, 0.545])
    panel(axL, 0, "Single-trajectory gap  (%)")
    panel(axR, 1, "Multistart gap  (%)")

    fig.text(0.055, 0.085,
             "TSP-100  ·  POMO RL  ·  matched params and budget  ·  gap vs LKH-3 optimal",
             fontsize=11, color=SECOND, va="top", ha="left")
    fig.text(0.055, 0.038,
             "curves: seed 1  ·  lower is better  ·  hybrid = 4 bidirectional Mamba layers + 1 attention layer",
             fontsize=9.5, color=MUTED, va="top", ha="left")

    fig.savefig(OUT, dpi=DPI, facecolor=PLANE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
