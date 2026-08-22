"""
ADMET benchmark composition — nested donut (sunburst) chart.

Outer ring : ADMET category, written as *curved* text that follows the ring
             (every glyph stands along the radius, exactly like the original)
Inner ring : the individual TDC datasets, written as straight radial text
Center     : small white hole (radius = HOLE_RADIUS)

Print-quality output: 600 dpi PNG + vector PDF/SVG.
"""

from math import cos, sin, radians, degrees

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.font_manager import FontProperties

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
HOLE_RADIUS = 0.20          # <-- smaller white centre (was ~0.35 in the original)
INNER_R0    = HOLE_RADIUS   # inner ring: datasets
INNER_R1    = 0.78
OUTER_R0    = 0.80          # outer ring: categories
OUTER_R1    = 1.00

START_ANGLE = 90.0          # first slice starts at 12 o'clock
DIRECTION   = -1            # -1 = clockwise, +1 = counter-clockwise

GAP_COLOR = "white"
GAP_WIDTH = 2.6

FONT_FAMILY = "DejaVu Sans"
CAT_FONTSIZE = 13
DS_FONTSIZE = 11

mpl.rcParams.update({
    "font.family":       FONT_FAMILY,
    "pdf.fonttype":      42,    # embed real TrueType glyphs (editable in Illustrator)
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
})


# ----------------------------------------------------------------------
# 3. Helpers
# ----------------------------------------------------------------------
def upright(angle_deg):
    """Normalise an angle to (-180, 180]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def px_per_data_unit(ax):
    """Pixels per data unit along x (aspect is 'equal', so y is identical)."""
    (x0, _), (x1, _) = ax.transData.transform([(0, 0), (1, 0)])
    return x1 - x0


def curved_text(fig, ax, ax_scale, s, r_center, mid_deg,
                fontsize=CAT_FONTSIZE, weight="bold", color="#111111", zorder=4):
    """Draw `s` bent along the circle of radius `r_center`, centred on `mid_deg`.

    Every glyph is placed individually and rotated so that its baseline is
    tangential and its stems point along the radius. The word is flipped as a
    whole (glyph heads inwards instead of outwards) whenever that is what keeps
    it readable, which is what the original figure does.
    """
    fp = FontProperties(family=FONT_FAMILY, weight=weight, size=fontsize)
    renderer = fig.canvas.get_renderer()

    def measure(t):
        """(width, height, descent) of `t` in data units."""
        w, h, d = renderer.get_text_width_height_descent(t, fp, False)
        return w / ax_scale, h / ax_scale, d / ax_scale

    total_w, h, desc = measure(s)

    # Glyph heads outwards (rotation = angle - 90) when that keeps the word
    # upright; otherwise heads inwards (rotation = angle + 90).
    heads_out = -90 < upright(mid_deg - 90) <= 90
    quarter = -90 if heads_out else +90
    # reading direction: clockwise for heads-out, counter-clockwise otherwise
    step = -1 if heads_out else +1

    # Put the optical middle of the glyphs on r_center, not their baseline.
    r_base = r_center - (h / 2 - desc) * (1 if heads_out else -1)

    span = degrees(total_w / r_base)                 # angular length of the word
    start = mid_deg - step * span / 2                # angle of the word's start

    for i, ch in enumerate(s):
        if ch == " ":
            continue
        w_before, _, _ = measure(s[:i])
        w_upto, _, _ = measure(s[:i + 1])
        offset = (w_before + w_upto) / 2             # centre of this glyph
        ang = start + step * degrees(offset / r_base)
        ax.text(r_base * cos(radians(ang)), r_base * sin(radians(ang)), ch,
                rotation=ang + quarter, rotation_mode="anchor",
                ha="center", va="baseline",
                fontsize=fontsize, fontweight=weight, color=color, zorder=zorder)


def radial_text(ax, s, r_outer, mid_deg,
                fontsize=DS_FONTSIZE, weight="bold", color="#111111", zorder=4):
    """Straight label running along the radius, anchored at the ring's outer edge."""
    a = upright(mid_deg)
    if -90 < a <= 90:                 # right half -> read outwards
        rot, ha = a, "right"
    else:                             # left half -> flip by 180 deg
        rot, ha = a + 180, "left"
    ax.text(r_outer * cos(radians(mid_deg)), r_outer * sin(radians(mid_deg)), s,
            rotation=rot, rotation_mode="anchor", ha=ha, va="center",
            fontsize=fontsize, fontweight=weight, color=color, zorder=zorder)


# ----------------------------------------------------------------------
# 4. Figure
# ----------------------------------------------------------------------
def build():
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()                     # freeze the layout before measuring text
    ax_scale = px_per_data_unit(ax)

    n_total = sum(len(d) for _, _, d in CATEGORIES)
    theta = START_ANGLE

    for name, color, datasets in CATEGORIES:
        span = 360.0 * len(datasets) / n_total
        t0, t1 = sorted((theta, theta + DIRECTION * span))

        # ---------------- outer ring: the category -------------------
        ax.add_patch(Wedge((0, 0), OUTER_R1, t0, t1, width=OUTER_R1 - OUTER_R0,
                           facecolor=color, edgecolor=GAP_COLOR,
                           linewidth=GAP_WIDTH, zorder=2))
        curved_text(fig, ax, ax_scale,
                    f"{name} ({100 * len(datasets) / n_total:.1f}%)",
                    (OUTER_R0 + OUTER_R1) / 2, (t0 + t1) / 2)

        # ---------------- inner ring: the datasets -------------------
        sub = span / len(datasets)
        for i, ds in enumerate(datasets):
            w0, w1 = sorted((theta + DIRECTION * sub * i,
                             theta + DIRECTION * sub * (i + 1)))
            ax.add_patch(Wedge((0, 0), INNER_R1, w0, w1, width=INNER_R1 - INNER_R0,
                               facecolor=color, edgecolor=GAP_COLOR,
                               linewidth=GAP_WIDTH, zorder=2))
            radial_text(ax, ds, INNER_R1 - 0.035, (w0 + w1) / 2)

        theta += DIRECTION * span

    # ---------------- the white hole in the middle -------------------
    ax.add_patch(Circle((0, 0), HOLE_RADIUS, facecolor="white",
                        edgecolor="#E6E1F0", linewidth=1.4, zorder=3))
    return fig


fig = build()

# ---- print-quality export: 600 dpi raster + vector ----
# (add ("tiff", dict(dpi=600)) below if the journal demands TIFF)
for ext, kw in (("png", dict(dpi=600)), ("pdf", {}), ("svg", {})):
    fig.savefig(f"admet_datasets_donut.{ext}",
                bbox_inches="tight", pad_inches=0.05, **kw)

plt.show()
