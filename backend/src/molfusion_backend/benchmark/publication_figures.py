"""Deterministic SVG figures, rendered without a plotting dependency.

Two reasons this emits SVG directly rather than calling a plotting library.

The project adds dependencies only after individual approval (see
pyproject.toml), and Phase 6B was not authorised to add one. More usefully,
hand-emitted SVG is *byte-deterministic*: the same frozen inputs produce
the same file on any machine, with no dependence on library version,
backend or installed fonts. Phase 6B requires figures that can be
regenerated exactly, and a rendering stack that silently changes its output
between versions would undermine that.

Every figure also writes its underlying table, so a figure can be
independently redrawn in any tool without re-deriving anything. The SVG is
a rendering of the CSV, never a separate calculation.

Coordinates are rounded to a fixed precision so float formatting cannot
introduce spurious diffs between runs.
"""

from collections.abc import Sequence
from typing import Any

FIGURE_SCRIPT_VERSION = "6B.1"

_PALETTE = {
    "rdkit_physchem_descriptors": "#1b6ca8",
    "smiles_tfidf_4096": "#c1441e",
    "morgan_ecfp4_1024": "#2e7d5b",
    "avalon_1024": "#7a5195",
    "maccs_keys_167": "#bc8034",
    "erg_reduced_graph_315": "#6b7280",
    "rdkit_fragment_descriptors": "#94708c",
}
_INK = "#1a1a1a"
_GRID = "#d8d8d8"
_MUTED = "#6b7280"


def _n(value: float) -> str:
    """Fixed-precision coordinate, so output cannot drift between runs."""
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _short(name: str) -> str:
    return {
        "rdkit_physchem_descriptors": "physchem (217)",
        "smiles_tfidf_4096": "SMILES TF-IDF (4096)",
        "morgan_ecfp4_1024": "Morgan ECFP4 (1024)",
        "avalon_1024": "Avalon (1024)",
        "maccs_keys_167": "MACCS (167)",
        "erg_reduced_graph_315": "ErG (315)",
        "rdkit_fragment_descriptors": "fragments (85)",
    }.get(name, name)


def _document(width: int, height: int, body: Sequence[str], title: str) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
        f"  <title>{_esc(title)}</title>",
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        *(f"  {line}" for line in body),
        "</svg>",
        "",
    ])


def _text(x, y, content, *, size=11, anchor="start", fill=_INK, weight="normal"):
    return (f'<text x="{_n(x)}" y="{_n(y)}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">'
            f"{_esc(content)}</text>")


# ---------------------------------------------------------------------------
# Figure 1 -- endpoint x representation rank heatmap, one panel per probe
# ---------------------------------------------------------------------------


def figure_rank_heatmap(rows: Sequence[dict[str, Any]], *, probes: Sequence[str],
                        representations: Sequence[str],
                        endpoints: Sequence[str],
                        low_stability: Sequence[str]) -> str:
    """Direction-normalised ranks. Raw MAE and AUROC are never averaged."""
    cell_w, cell_h = 74, 19
    left, top, gap = 210, 62, 46
    panel_w = cell_w * len(representations)
    width = left + panel_w * len(probes) + gap * (len(probes) - 1) + 30
    height = top + cell_h * len(endpoints) + 74

    lookup = {(r["endpoint"], r["probe"], r["representation"]): float(r["rank"])
              for r in rows}
    body: list[str] = []

    for panel, probe in enumerate(probes):
        x0 = left + panel * (panel_w + gap)
        body.append(_text(x0 + panel_w / 2, 26, f"{probe} probe",
                          size=13, anchor="middle", weight="bold"))
        for index, representation in enumerate(representations):
            cx = x0 + index * cell_w + cell_w / 2
            body.append(f'<g transform="translate({_n(cx)},{_n(top - 8)}) rotate(-38)">'
                        f'{_text(0, 0, _short(representation), size=9, anchor="start", fill=_MUTED)}'
                        "</g>")

        for row_index, endpoint in enumerate(endpoints):
            y0 = top + row_index * cell_h
            if panel == 0:
                flagged = endpoint in low_stability
                label = f"{endpoint} *" if flagged else endpoint
                body.append(_text(left - 10, y0 + cell_h * 0.72, label, size=9.5,
                                  anchor="end", fill=_MUTED if flagged else _INK))
            for index, representation in enumerate(representations):
                rank = lookup.get((endpoint, probe, representation))
                x = x0 + index * cell_w
                if rank is None:
                    fill, label = "#f2f2f2", ""
                else:
                    # rank 1 (best) darkest; linear ramp over 7 positions.
                    t = (rank - 1) / 6
                    shade = int(round(60 + t * 185))
                    fill = f"rgb({shade},{min(255, shade + 12)},{min(255, shade + 30)})"
                    label = f"{rank:.1f}".rstrip("0").rstrip(".")
                body.append(f'<rect x="{_n(x)}" y="{_n(y0)}" width="{cell_w}" '
                            f'height="{cell_h}" fill="{fill}" stroke="#ffffff" '
                            f'stroke-width="1"/>')
                if label:
                    ink = "#ffffff" if rank <= 2.5 else _INK
                    body.append(_text(x + cell_w / 2, y0 + cell_h * 0.71, label,
                                      size=9, anchor="middle", fill=ink))

    body.append(_text(left, height - 40,
                      "Within-endpoint rank, 1 = best. Direction-aware: lower is "
                      "better after orienting each endpoint's primary metric.",
                      size=9.5, fill=_MUTED))
    body.append(_text(left, height - 25,
                      "* endpoint whose ranking did not survive repartitioning "
                      "(Kendall's W < 0.35 on the weaker probe); no per-endpoint "
                      "claim is drawn from these rows.",
                      size=9.5, fill=_MUTED))
    return _document(width, height, body, "Figure 1 - representation rank by endpoint")


# ---------------------------------------------------------------------------
# Figure 2 -- mean rank with bootstrap CI
# ---------------------------------------------------------------------------


def figure_mean_rank_ci(rows: Sequence[dict[str, Any]], *,
                        probes: Sequence[str]) -> str:
    """Equal x-scale in both panels, so linear differences cannot be inflated."""
    panel_w, row_h = 400, 34
    left, top, gap = 200, 66, 96
    width = left + panel_w * len(probes) + gap + 40
    height = top + row_h * 7 + 92
    lo, hi = 1.0, 7.0

    def sx(x0, rank):
        return x0 + (rank - lo) / (hi - lo) * panel_w

    body: list[str] = []
    for panel, probe in enumerate(probes):
        x0 = left + panel * (panel_w + gap)
        subset = sorted((r for r in rows if r["probe"] == probe),
                        key=lambda r: float(r["mean_rank"]))
        body.append(_text(x0 + panel_w / 2, 30, f"{probe} probe",
                          size=13, anchor="middle", weight="bold"))

        for tick in range(1, 8):
            x = sx(x0, tick)
            body.append(f'<line x1="{_n(x)}" y1="{_n(top - 12)}" x2="{_n(x)}" '
                        f'y2="{_n(top + row_h * 7 - 8)}" stroke="{_GRID}" stroke-width="1"/>')
            body.append(_text(x, top + row_h * 7 + 12, str(tick), size=9.5,
                              anchor="middle", fill=_MUTED))

        leader_upper = float(subset[0]["ci_upper_95"])
        for index, row in enumerate(subset):
            y = top + index * row_h + row_h / 2
            name = row["representation"]
            mean = float(row["mean_rank"])
            ci_lo, ci_hi = float(row["ci_lower_95"]), float(row["ci_upper_95"])
            colour = _PALETTE.get(name, _MUTED)
            if panel == 0:
                body.append(_text(left - 12, y + 4, _short(name), size=10.5, anchor="end"))
            body.append(f'<line x1="{_n(sx(x0, ci_lo))}" y1="{_n(y)}" '
                        f'x2="{_n(sx(x0, ci_hi))}" y2="{_n(y)}" stroke="{colour}" '
                        f'stroke-width="2.5" stroke-linecap="round"/>')
            for edge in (ci_lo, ci_hi):
                body.append(f'<line x1="{_n(sx(x0, edge))}" y1="{_n(y - 5)}" '
                            f'x2="{_n(sx(x0, edge))}" y2="{_n(y + 5)}" '
                            f'stroke="{colour}" stroke-width="2"/>')
            body.append(f'<circle cx="{_n(sx(x0, mean))}" cy="{_n(y)}" r="4.5" '
                        f'fill="{colour}"/>')
            body.append(_text(sx(x0, ci_hi) + 9, y + 4,
                              f"{mean:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]",
                              size=9, fill=_MUTED))

        if index := len(subset):
            separated = all(float(r["ci_lower_95"]) > leader_upper for r in subset[1:])
            note = ("leader CI clear of every competitor" if separated
                    else "leader CI overlaps competitors")
            body.append(_text(x0, top + row_h * 7 + 34, note, size=9.5,
                              fill=_INK if separated else _MUTED,
                              weight="bold" if separated else "normal"))

    body.append(_text(left, height - 22,
                      "Mean rank over 22 endpoints with 95% bootstrap intervals "
                      "(10,000 resamples; resampling unit = endpoint). Both panels "
                      "share one scale. Intervals are marginal, not simultaneous.",
                      size=9.5, fill=_MUTED))
    return _document(width, height, body, "Figure 2 - mean rank with bootstrap CI")


# ---------------------------------------------------------------------------
# Figure 3 -- A1 -> A2 slopegraph
# ---------------------------------------------------------------------------


def figure_rank_slopegraph(rows: Sequence[dict[str, Any]], *,
                           probes: Sequence[str]) -> str:
    panel_w, row_h = 190, 46
    left, top, gap = 190, 70, 210
    width = left + (panel_w + gap) * len(probes) + 60
    height = top + row_h * 7 + 60

    body: list[str] = []
    for panel, probe in enumerate(probes):
        x0 = left + panel * (panel_w + gap)
        subset = [r for r in rows if r["probe"] == probe and r["subset"] == "all"]
        body.append(_text(x0 + panel_w / 2, 30, f"{probe} probe",
                          size=13, anchor="middle", weight="bold"))
        body.append(_text(x0, top - 16, "A1", size=11, anchor="middle",
                          fill=_MUTED, weight="bold"))
        body.append(_text(x0 + panel_w, top - 16, "A2", size=11, anchor="middle",
                          fill=_MUTED, weight="bold"))

        for row in subset:
            name = row["representation"]
            a1 = int(row["a1_position"])
            a2 = int(row["a2_position"])
            y1 = top + (a1 - 1) * row_h + row_h / 2
            y2 = top + (a2 - 1) * row_h + row_h / 2
            colour = _PALETTE.get(name, _MUTED)
            moved = a1 != a2
            stroke_width = "2.6" if moved else "1.4"
            dash = "" if moved else ' stroke-dasharray="4 3"'
            body.append(f'<line x1="{_n(x0)}" y1="{_n(y1)}" x2="{_n(x0 + panel_w)}" '
                        f'y2="{_n(y2)}" stroke="{colour}" '
                        f'stroke-width="{stroke_width}"{dash} opacity="0.9"/>')
            body.append(f'<circle cx="{_n(x0)}" cy="{_n(y1)}" r="4" fill="{colour}"/>')
            body.append(f'<circle cx="{_n(x0 + panel_w)}" cy="{_n(y2)}" r="4" '
                        f'fill="{colour}"/>')
            body.append(_text(x0 - 10, y1 + 4, f"{a1}. {_short(name)}", size=9.5,
                              anchor="end"))
            delta = a1 - a2
            marker = "" if delta == 0 else f"  ({delta:+d})"
            body.append(_text(x0 + panel_w + 10, y2 + 4, f"{a2}{marker}", size=9.5,
                              fill=_MUTED))

    body.append(_text(left - 100, height - 22,
                      "Position by mean rank, 1 = best. A1 = official TDC partition "
                      "(primary); A2 = MolFusion scaffold repartitioning "
                      "(supplementary). Dashed = unchanged.",
                      size=9.5, fill=_MUTED))
    return _document(width, height, body, "Figure 3 - A1 to A2 rank robustness")


# ---------------------------------------------------------------------------
# Figure 4 -- performance versus cost
# ---------------------------------------------------------------------------


def figure_rank_vs_cost(rows: Sequence[dict[str, Any]]) -> str:
    """Two axes, never combined. No composite efficiency score is defined."""
    width, height = 760, 500
    left, right, top, bottom = 96, 40, 60, 92
    plot_w, plot_h = width - left - right, height - top - bottom

    costs = [float(r["nonlinear_model_seconds"]) / 3600 for r in rows]
    cmin, cmax = min(costs), max(costs)
    span = cmax - cmin or 1.0

    def sx(cost_hours):
        return left + (cost_hours - cmin) / span * plot_w

    def sy(rank):
        return top + (rank - 1) / 6 * plot_h

    body: list[str] = []
    body.append(_text(width / 2, 28, "Nonlinear rank versus nonlinear compute",
                      size=13, anchor="middle", weight="bold"))

    for tick in range(1, 8):
        y = sy(tick)
        body.append(f'<line x1="{_n(left)}" y1="{_n(y)}" x2="{_n(left + plot_w)}" '
                    f'y2="{_n(y)}" stroke="{_GRID}" stroke-width="1"/>')
        body.append(_text(left - 10, y + 4, str(tick), size=9.5, anchor="end", fill=_MUTED))

    steps = 5
    for step in range(steps + 1):
        hours = cmin + span * step / steps
        x = sx(hours)
        body.append(f'<line x1="{_n(x)}" y1="{_n(top + plot_h)}" x2="{_n(x)}" '
                    f'y2="{_n(top + plot_h + 5)}" stroke="{_MUTED}" stroke-width="1"/>')
        body.append(_text(x, top + plot_h + 19, f"{hours:.1f}", size=9.5,
                          anchor="middle", fill=_MUTED))

    body.append(_text(left + plot_w / 2, top + plot_h + 42,
                      "nonlinear model compute (hours, 22 endpoints x 5 seeds)",
                      size=10.5, anchor="middle", fill=_MUTED))
    body.append(f'<g transform="translate(26,{_n(top + plot_h / 2)}) rotate(-90)">'
                f'{_text(0, 0, "mean nonlinear rank (1 = best)", size=10.5, anchor="middle", fill=_MUTED)}'
                "</g>")

    annotate = {"rdkit_physchem_descriptors", "smiles_tfidf_4096"}
    for row in rows:
        name = row["representation"]
        x = sx(float(row["nonlinear_model_seconds"]) / 3600)
        y = sy(float(row["nonlinear_mean_rank"]))
        colour = _PALETTE.get(name, _MUTED)
        emphasised = name in annotate
        body.append(f'<circle cx="{_n(x)}" cy="{_n(y)}" r="{7 if emphasised else 5}" '
                    f'fill="{colour}" opacity="{1 if emphasised else 0.8}"/>')
        anchor = "end" if x > left + plot_w * 0.7 else "start"
        offset = -11 if anchor == "end" else 11
        body.append(_text(x + offset, y + 4, _short(name),
                          size=10 if emphasised else 9,
                          anchor=anchor, fill=_INK if emphasised else _MUTED,
                          weight="bold" if emphasised else "normal"))

    body.append(_text(left, height - 30,
                      "Lower and further left is better on both axes. Cost and "
                      "predictive rank are reported separately; no composite "
                      "efficiency score is defined.",
                      size=9.5, fill=_MUTED))
    return _document(width, height, body, "Figure 4 - rank versus cost")


# ---------------------------------------------------------------------------
# Figure 5 -- endpoint stability
# ---------------------------------------------------------------------------


def figure_endpoint_stability(rows: Sequence[dict[str, Any]]) -> str:
    bar_h, width = 22, 720
    left, top = 250, 66
    plot_w = width - left - 90
    height = top + bar_h * len(rows) + 76
    threshold = 0.35

    body: list[str] = []
    body.append(_text(width / 2, 28,
                      "Rank agreement across the five A2 partitions (Kendall's W)",
                      size=13, anchor="middle", weight="bold"))

    ordered = sorted(rows, key=lambda r: float(r["kendall_w_min"]))
    for index, row in enumerate(ordered):
        y = top + index * bar_h
        flagged = row["endpoint_stability_flag"] != "OK"
        for probe, colour, offset in ((("kendall_w_linear"), "#8aa9c9", 1),
                                      (("kendall_w_nonlinear"), "#1b6ca8", 10)):
            value = float(row[probe])
            body.append(f'<rect x="{_n(left)}" y="{_n(y + offset)}" '
                        f'width="{_n(value * plot_w)}" height="8" fill="{colour}"/>')
        label = row["endpoint"]
        body.append(_text(left - 10, y + 15, label, size=9.5, anchor="end",
                          fill="#a33" if row["endpoint_stability_flag"] == "LOW"
                          else _MUTED if flagged else _INK))
        body.append(_text(left + plot_w + 8, y + 15,
                          f"{float(row['kendall_w_min']):.2f}", size=9, fill=_MUTED))

    x = left + threshold * plot_w
    body.append(f'<line x1="{_n(x)}" y1="{_n(top - 6)}" x2="{_n(x)}" '
                f'y2="{_n(top + bar_h * len(ordered))}" stroke="#a33" '
                f'stroke-width="1.5" stroke-dasharray="5 3"/>')
    body.append(_text(x + 5, top - 10, "W = 0.35", size=9, fill="#a33"))

    body.append(_text(left - 200, height - 40,
                      "Light bar = linear probe, dark bar = nonlinear probe. "
                      "Endpoints below the threshold on their weaker probe "
                      "(red labels) do not support per-endpoint claims;",
                      size=9.5, fill=_MUTED))
    body.append(_text(left - 200, height - 25,
                      "they are retained in all cross-endpoint analyses.",
                      size=9.5, fill=_MUTED))
    return _document(width, height, body, "Figure 5 - endpoint rank stability")


__all__ = [
    "FIGURE_SCRIPT_VERSION",
    "figure_endpoint_stability",
    "figure_mean_rank_ci",
    "figure_rank_heatmap",
    "figure_rank_slopegraph",
    "figure_rank_vs_cost",
]
