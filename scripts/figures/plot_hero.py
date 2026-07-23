#!/usr/bin/env python3
"""Figure 2: hybrid hero (TSP-100, single-trajectory and multistart gap to 500k).

Regenerates results/gate0/incoming/fig_hybrid_hero.png.

Design note: NO em dashes anywhere in rendered text. Use middle dots, colons,
or commas only. Colors follow the dataviz reference palette (light mode):
  neutral-dark #444442 (Attention reference), orange #eb6834 (BiMamba),
  blue #2a78d6 (Hybrid, hero).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results", "gate0", "incoming", "fig_hybrid_hero.png")

C_DARK = "#444442"    # Attention reference (dashed)
C_ORANGE = "#eb6834"  # BiMamba
C_BLUE = "#2a78d6"    # Hybrid (hero)
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
            continue
        if xmin is not None and step < xmin:
            continue
        if xmax is not None and step > xmax:
            continue
        xs.append(step)
        ys.append(rows[step])
    return xs, ys


def main():
    inc = os.path.join(REPO, "results", "gate0", "incoming")
    paths = {
        "Attention": os.path.join(inc, "calib_attention_hilbert_s1.curve.jsonl"),
        "BiMamba": os.path.join(inc, "calib_bimamba_hilbert_s1.curve.jsonl"),
        "Hybrid": os.path.join(inc, "calib_hybrid_hilbert_s1.curve.jsonl"),
    }
    colors = {"Attention": C_DARK, "BiMamba": C_ORANGE, "Hybrid": C_BLUE}
    order = ["Attention", "BiMamba", "Hybrid"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.edgecolor": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9, 4.5), dpi=200)
    fig.subplots_adjust(left=0.055, right=0.845, top=0.78, bottom=0.12, wspace=0.46)

    def draw(ax, field, label_offsets):
        endpoints = {}
        for name in order:
            xs, ys = load_curve(paths[name], field, xmin=20000, xmax=500000)
            dashed = name == "Attention"
            ax.plot(
                xs, ys,
                color=colors[name],
                linewidth=2.6 if name == "Hybrid" else (2.0 if dashed else 2.2),
                linestyle=(0, (5, 3)) if dashed else "-",
                zorder=3,
                solid_capstyle="round",
            )
            ax.plot(xs[-1], ys[-1], "o", color=colors[name], markersize=6, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            endpoints[name] = (xs[-1], ys[-1])
        for name in order:
            x, y = endpoints[name]
            ax.annotate(
                f"{name}  {y:.2f}",
                xy=(x, y),
                xytext=(x + 9000, y + label_offsets[name]),
                va="center", ha="left",
                fontsize=10, fontweight="bold", color=colors[name],
                annotation_clip=False,
            )
        return endpoints

    # Left panel: single-trajectory gap. Endpoints: BiMamba 4.89, Hybrid 3.01, Attention 2.45.
    epL = draw(axl, "single_traj_gap_pct",
               {"BiMamba": 0.0, "Hybrid": 0.0, "Attention": -0.02})
    axl.set_ylim(2.0, 10.0)
    axl.set_title("single-trajectory gap (%)", fontsize=12, color=INK2, pad=6)

    # Right panel: multistart gap. Endpoints: Attention 1.59, BiMamba 1.54, Hybrid 1.02.
    # Attention and BiMamba endpoints nearly coincide, so nudge labels apart.
    epR = draw(axr, "multistart_gap_pct",
               {"Attention": 0.14, "BiMamba": -0.14, "Hybrid": 0.0})
    axr.set_ylim(0.8, 4.4)
    axr.set_title("multistart gap (%)", fontsize=12, color=INK2, pad=6)

    # Annotate the crossover: hybrid multistart drops below full attention (~120k).
    axr.annotate(
        "hybrid drops below\nfull attention",
        xy=(130000, 2.30),
        xytext=(215000, 3.35),
        fontsize=10, color=INK, ha="left", va="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4,
                        connectionstyle="arc3,rad=-0.25"),
        annotation_clip=False,
    )

    for ax in (axl, axr):
        ax.set_xlim(20000, 500000)
        ax.set_xticks([20000, 100000, 200000, 300000, 400000, 500000])
        ax.set_xticklabels(["20k", "100k", "200k", "300k", "400k", "500k"])
        ax.set_xlabel("training steps", fontsize=11, color=INK2)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")

    fig.text(0.055, 0.94, "One attention layer is enough",
             fontsize=17, fontweight="bold", color=INK, ha="left")
    subtitle = ("TSP-100 · POMO RL · matched params and budget. An encoder that is 80% "
                "linear-scan Mamba\nbeats full attention on multistart tour quality "
                "(seed 1 curves; 3-seed result in text).")
    fig.text(0.055, 0.845, subtitle, fontsize=9.5, color=INK2, ha="left",
             linespacing=1.35)

    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")
    print("left (single_traj):")
    for name in order:
        x, y = epL[name]
        print(f"  {name}: step {x} -> {y:.4f}")
    print("right (multistart):")
    for name in order:
        x, y = epR[name]
        print(f"  {name}: step {x} -> {y:.4f}")


if __name__ == "__main__":
    main()
