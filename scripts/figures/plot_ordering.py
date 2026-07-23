#!/usr/bin/env python3
"""Figure 1: ordering ablation curves (TSP-100 single-trajectory optimality gap).

Regenerates results/gate0/ablation/fig_ordering_curves.png.

Design note: NO em dashes anywhere in rendered text. Use middle dots, colons,
or commas only. Colors follow the dataviz reference palette (light mode):
  blue #2a78d6, aqua #1baf7a, red #e34948, neutral-dark #444442.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "gate0", "ablation", "fig_ordering_curves.png")

# Palette (dataviz reference, light surface)
C_BLUE = "#2a78d6"    # Mamba random
C_AQUA = "#1baf7a"    # Mamba sort
C_RED = "#e34948"     # Mamba Hilbert
C_DARK = "#444442"    # Attention (neutral dark, dashed reference)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_curve(path, field, xmin=None, xmax=None):
    """Load JSONL curve. Dedupe by step keep-last, drop step-0 outliers."""
    rows = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["step"]] = r[field]  # keep-last dedupe
    xs, ys = [], []
    for step in sorted(rows):
        if step == 0:
            continue  # drop step-0 outlier
        if xmin is not None and step < xmin:
            continue
        if xmax is not None and step > xmax:
            continue
        xs.append(step)
        ys.append(rows[step])
    return xs, ys


def main():
    abl = os.path.join(REPO, "results", "gate0", "ablation")
    ext = os.path.join(REPO, "results", "gate0", "calibration", "extension")

    series = [
        ("Mamba random", os.path.join(abl, "calib_mamba_random_s1.curve.jsonl"), C_BLUE),
        ("Mamba sort", os.path.join(abl, "calib_mamba_sort_s1.curve.jsonl"), C_AQUA),
        ("Mamba Hilbert", os.path.join(ext, "calib_mamba_hilbert_s1.curve.jsonl"), C_RED),
        ("Attention", os.path.join(ext, "calib_attention_hilbert_s1.curve.jsonl"), C_DARK),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.edgecolor": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    fig.subplots_adjust(left=0.075, right=0.70, top=0.80, bottom=0.12)

    endpoints = {}
    for name, path, color in series:
        # Attention plotted as reference only up to 250000 for this figure.
        xs, ys = load_curve(path, "single_traj_gap_pct", xmin=20000, xmax=250000)
        dashed = name == "Attention"
        ax.plot(
            xs, ys,
            color=color,
            linewidth=2.4 if not dashed else 2.0,
            linestyle=(0, (5, 3)) if dashed else "-",
            zorder=3,
            solid_capstyle="round",
        )
        ax.plot(xs[-1], ys[-1], "o", color=color, markersize=6, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        endpoints[name] = (xs[-1], ys[-1])

    # Direct endpoint labels (no legend box), placed to the right of the line ends.
    label_y = {
        "Mamba Hilbert": endpoints["Mamba Hilbert"][1],
        "Mamba sort": endpoints["Mamba sort"][1],
        "Mamba random": endpoints["Mamba random"][1],
        "Attention": endpoints["Attention"][1],
    }
    for name, (x, y) in endpoints.items():
        color = dict((n, c) for n, _, c in series)[name]
        ax.annotate(
            f"{name}  {y:.2f}",
            xy=(x, y),
            xytext=(x + 5000, label_y[name]),
            va="center", ha="left",
            fontsize=11, fontweight="bold", color=color,
            annotation_clip=False,
        )

    ax.set_xlim(20000, 250000)
    ax.set_ylim(2.0, 9.2)
    ax.set_xticks([20000, 60000, 100000, 140000, 180000, 220000, 250000])
    ax.set_xticklabels(["20k", "60k", "100k", "140k", "180k", "220k", "250k"])
    ax.set_xlabel("training steps", fontsize=11.5, color=INK2)
    ax.set_ylabel("single-trajectory gap (%)", fontsize=11.5, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=10.5)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")

    # Title and subtitle (no em dashes).
    fig.text(0.075, 0.945, "The locality prior did not help",
             fontsize=17, fontweight="bold", color=INK, ha="left")
    subtitle = ("TSP-100 optimality gap, seed 1 curves. Across 3 seeds the Hilbert and "
                "random\nranges overlap: the prior buys nothing for a causal scan. "
                "Attention shown as reference.")
    fig.text(0.075, 0.855, subtitle, fontsize=9.5, color=INK2, ha="left",
             linespacing=1.35)

    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")
    for name, (x, y) in endpoints.items():
        print(f"  {name}: step {x} -> {y:.4f}")


if __name__ == "__main__":
    main()
