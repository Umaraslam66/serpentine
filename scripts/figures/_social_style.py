"""Shared style for the three social cards.

Palette (validated via dataviz skill validator, blue/orange CVD dE 46.4):
  hybrid    -> blue   #2a78d6   (the hero)
  bimamba   -> orange #eb6834
  attention -> neutral dark #52514e (always dashed + direct-labelled: secondary encoding)

Constraint: no em dashes anywhere in any rendered string.
Output: 1600x900 logical, rendered at 2x (3200x1800) for phone-crisp text.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl

# --- palette -----------------------------------------------------------------
HYBRID    = "#2a78d6"   # blue, hero
BIMAMBA   = "#eb6834"   # orange
ATTENTION = "#52514e"   # neutral dark

SURFACE   = "#fcfcfb"   # chart / card surface (light)
PLANE     = "#f4f3f0"   # page plane behind panels
INK       = "#0b0b0b"   # primary text
SECOND    = "#52514e"   # secondary text
MUTED     = "#898781"   # axis / muted labels
GRID      = "#e1e0d9"   # hairline gridline
AXISLN    = "#c3c2b7"   # baseline / axis
HYBRID_WASH = "#eaf1fb"  # pale blue for hero highlight

# --- typography (large, phone-readable) --------------------------------------
def apply_rc():
    mpl.rcParams.update({
        "font.family": "Helvetica",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.edgecolor": AXISLN,
        "axes.labelcolor": SECOND,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "none",
        "figure.dpi": 400,          # 8x4.5in * 400 = 3200x1800 = 2x of 1600x900
    })

DPI = 400          # 2x of 100dpi -> 1600x900 logical at 2x
FIGSIZE = (8.0, 4.5)


def load_curve(model):
    """Return (steps, single_pct, multi_pct) deduped by step, step 0 dropped."""
    pts = {}
    path = f"results/gate0/incoming/calib_{model}_hilbert_s1.curve.jsonl"
    for line in open(path):
        d = json.loads(line)
        if d["step"] == 0:
            continue
        pts[d["step"]] = (d["single_traj_gap_pct"], d["multistart_gap_pct"])
    steps = sorted(pts)
    return steps, [pts[s][0] for s in steps], [pts[s][1] for s in steps]
