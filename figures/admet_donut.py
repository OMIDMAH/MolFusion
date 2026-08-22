"""
ADMET benchmark composition — nested donut (sunburst) chart.

Outer ring : ADMET category (with share of the 22 datasets)
Inner ring : the individual TDC datasets belonging to each category
Center     : small white hole (radius = HOLE_RADIUS)

Print-quality output: 600 dpi PNG + vector PDF/SVG.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
CATEGORIES = [
    ("Absorption",   "#4DC8B4", ["Caco2_Wang", "HIA_Hou", "Pgp_Broccatelli",
                                 "Bioavailability_Ma", "Lipophilicity_AstraZeneca",
                                 "Solubility_AqSolDB"]),
    ("Distribution", "#FCD860", ["BBB_Martins", "PPBR_AZ", "VDss_Lombardo"]),
    ("Metabolism",   "#B49BE4", ["CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
                                 "CYP2C9_Substrate", "CYP2D6_Substrate",
                                 "CYP3A4_Substrate"]),
    ("Excretion",    "#F7716A", ["Clearance_Microsome", "Clearance_Hepatocyte",
                                 "Half-Life_Obach"]),
    ("Toxicity",     "#6D96D6", ["LD50_Zhu", "hERG", "AMES", "DILI"]),
]

# ----------------------------------------------------------------------
# 2. Geometry  ->  HOLE_RADIUS is the white circle in the middle
# ----------------------------------------------------------------------
HOLE_RADIUS  = 0.20          # <-- smaller white centre (was ~0.35 in the original)
INNER_R0     = HOLE_RADIUS   # inner ring: datasets
INNER_R1     = 0.78
OUTER_R0     = 0.80          # outer ring: categories
OUTER_R1     = 1.00

START_ANGLE  = 90.0          # first slice starts at 12 o'clock
DIRECTION    = -1            # -1 = clockwise, +1 = counter-clockwise

GAP_COLOR    = "white"
GAP_WIDTH    = 2.6

mpl.rcParams.update({
    "font.family":      "DejaVu Sans",
    "pdf.fonttype":     42,     # embed real TrueType glyphs (editable in Illustrator)
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def upright(angle_deg):
    """Normalise an angle to (-180, 180]."""
    a = (angle_deg + 180.0) % 360.0 - 180.0
    return a


def draw(ax):
    n_total = sum(len(d) for _, _, d in CATEGORIES)
    theta = START_ANGLE

    for name, color, datasets in CATEGORIES:
        span = 360.0 * len(datasets) / n_total
        t0, t1 = sorted((theta, theta + DIRECTION * span))

        # ---------------- outer ring: the category -------------------
        ax.add_patch(Wedge((0, 0), OUTER_R1, t0, t1, width=OUTER_R1 - OUTER_R0,
                           facecolor=color, edgecolor=GAP_COLOR,
                           linewidth=GAP_WIDTH, zorder=2))

        mid = (t0 + t1) / 2.0
        # tangential text, always kept upright (never mirrored)
        rot = next(c for c in (upright(mid - 90), upright(mid + 90))
                   if -90 < c <= 90)
        ax.text(((OUTER_R0 + OUTER_R1) / 2) * _cos(mid),
                ((OUTER_R0 + OUTER_R1) / 2) * _sin(mid),
                f"{name} ({100 * len(datasets) / n_total:.1f}%)",
                rotation=rot, rotation_mode="anchor",
                ha="center", va="center",
                fontsize=13, fontweight="bold", color="#111111", zorder=4)

        # ---------------- inner ring: the datasets -------------------
        sub = span / len(datasets)
        for i, ds in enumerate(datasets):
            a0 = theta + DIRECTION * sub * i
            a1 = theta + DIRECTION * sub * (i + 1)
            w0, w1 = sorted((a0, a1))
            ax.add_patch(Wedge((0, 0), INNER_R1, w0, w1, width=INNER_R1 - INNER_R0,
                               facecolor=color, edgecolor=GAP_COLOR,
                               linewidth=GAP_WIDTH, zorder=2))

            m = (w0 + w1) / 2.0
            r = INNER_R1 - 0.035                    # anchor near the outer edge
            a = upright(m)
            if -90 < a <= 90:                       # right half -> read outward
                rot, ha = a, "right"
            else:                                   # left half -> flip 180°
                rot, ha = a + 180, "left"
            ax.text(r * _cos(m), r * _sin(m), ds,
                    rotation=rot, rotation_mode="anchor",
                    ha=ha, va="center",
                    fontsize=11, fontweight="bold", color="#111111", zorder=4)

        theta += DIRECTION * span

    # ---------------- the white hole in the middle -------------------
    ax.add_patch(Circle((0, 0), HOLE_RADIUS, facecolor="white",
                        edgecolor="#E6E1F0", linewidth=1.4, zorder=3))


def _cos(deg):
    from math import cos, radians
    return cos(radians(deg))


def _sin(deg):
    from math import sin, radians
    return sin(radians(deg))


fig, ax = plt.subplots(figsize=(9, 9))
draw(ax)
ax.set_xlim(-1.08, 1.08)
ax.set_ylim(-1.08, 1.08)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0.2)

# ---- print-quality export: 600 dpi raster + vector ----
# (add ("tiff", dict(dpi=600)) below if the journal demands TIFF)
for ext, kw in (("png", dict(dpi=600)), ("pdf", {}), ("svg", {})):
    fig.savefig(f"admet_datasets_donut.{ext}",
                bbox_inches="tight", pad_inches=0.05, **kw)

plt.show()
