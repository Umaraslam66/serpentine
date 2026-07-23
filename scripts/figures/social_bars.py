"""IMAGE 2 - social bars: multistart optimality gap at 500k steps (3 seeds).

Headline: Lower is better: multistart optimality gap at 500k steps
Annotation: worst hybrid seed 1.365 < best attention seed 1.589
Hybrid highlighted as the hero with a value callout.
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(__file__))
from _social_style import (
    apply_rc, DPI, FIGSIZE,
    HYBRID, BIMAMBA, ATTENTION, SURFACE, PLANE, INK, SECOND, MUTED, GRID, AXISLN,
    HYBRID_WASH,
)

OUT = "assets/social/social_bars.png"

# name, value, err (None = single seed), color, is_hero, sublabel
BARS = [
    ("Hybrid",    1.14, 0.19, HYBRID,    True,  "4 BiMamba layers + 1 attention  ·  4.5% fewer params"),
    ("BiMamba",   1.54, None, BIMAMBA,   False, "5 bidirectional Mamba layers  ·  seed 1"),
    ("Attention", 1.61, 0.02, ATTENTION, False, "full self-attention encoder"),
]


def rounded_bar(ax, y, width, height, color, r=0.045):
    """Horizontal bar grown from x=0 with a rounded data-end."""
    patch = FancyBboxPatch(
        (0, y - height / 2), width, height,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=1.0,
        linewidth=0, facecolor=color, zorder=4,
        clip_on=False,
    )
    ax.add_patch(patch)
    # square off the baseline (left) end
    ax.add_patch(Rectangle((0, y - height / 2), r, height,
                           linewidth=0, facecolor=color, zorder=4, clip_on=False))


def main():
    apply_rc()
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(PLANE)

    fig.text(0.055, 0.955, "Lower is better",
             fontsize=30, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.055, 0.815, "multistart optimality gap at 500k steps",
             fontsize=15, color=SECOND, va="top", ha="left")

    ax = fig.add_axes([0.055, 0.205, 0.905, 0.500])
    ax.set_facecolor(PLANE)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.5, len(BARS) - 0.5)
    ax.invert_yaxis()  # first entry (Hybrid) on top

    # recessive vertical gridlines
    for xg in (0.5, 1.0, 1.5, 2.0):
        ax.axvline(xg, color=GRID, lw=0.9, zorder=0)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0"], fontsize=10, color=MUTED)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXISLN)
    ax.set_xlabel("gap vs LKH-3 optimal  (%)", fontsize=11, color=MUTED, labelpad=6)

    bar_h = 0.44
    for i, (name, val, err, color, hero, sub) in enumerate(BARS):
        # hero highlight wash spanning the row
        if hero:
            ax.add_patch(Rectangle((0, i - 0.5), 2.0, 1.0,
                                   facecolor=HYBRID_WASH, edgecolor="none",
                                   zorder=1, clip_on=True))
        rounded_bar(ax, i, val, bar_h, color)

        # error bar
        if err is not None:
            ax.errorbar(val, i, xerr=err, fmt="none", ecolor=INK,
                        elinewidth=2.0, capsize=7, capthick=2.0, zorder=6)

        # category name inside the bar (white on the colored fill)
        ax.text(0.03, i - 0.045, name, fontsize=17 if hero else 15,
                fontweight="bold", color="#ffffff",
                va="center", ha="left", zorder=7)
        # sublabel BELOW the bar, dark ink over the row
        ax.text(0.03, i + 0.345, sub, fontsize=10,
                color=SECOND, va="center", ha="left", zorder=7)

        # value callout at the tip
        tipx = val + (err or 0) + 0.06
        vfs = 26 if hero else 17
        vcol = HYBRID if hero else INK
        ax.text(tipx, i, f"{val:.2f}", fontsize=vfs, fontweight="bold",
                color=vcol, va="center", ha="left", zorder=7)
        if err is not None:
            ax.text(tipx + (0.30 if hero else 0.19), i,
                    f"± {err:.2f}", fontsize=11 if hero else 10,
                    color=MUTED, va="center", ha="left", zorder=7)

    # annotation callout
    fig.text(0.055, 0.115,
             "worst hybrid seed  1.365   <   best attention seed  1.589",
             fontsize=13.5, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.055, 0.058,
             "TSP-100  ·  POMO RL  ·  matched params and budget  ·  lower gap = better tours",
             fontsize=10.5, color=MUTED, va="top", ha="left")

    fig.savefig(OUT, dpi=DPI, facecolor=PLANE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
