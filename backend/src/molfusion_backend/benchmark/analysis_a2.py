"""Track A2 analysis: repartitioning stability, and the A1-versus-A2 test.

Reuses the Phase 6A.3 analysis primitives -- direction-aware ranking,
seed aggregation, Friedman, Holm-corrected Wilcoxon, rank-biserial,
endpoint-level bootstrap -- because A2's cross-endpoint claims must be
computed exactly the way A1's were, or the comparison is between methods
rather than between tracks.

Two things are genuinely new here.

**Between-split stability.** A1's five values shared one test set, so their
spread measured only model-selection variability. A2's five values come from
five different test sets, so the spread now includes which molecules are
evaluated. That makes a stability question askable for the first time, and
it is measured at the seed level -- while the *statistical* unit stays the
endpoint, so n is never inflated from 22 to 110.

**The A1 comparison.** Each pre-registered hypothesis is evaluated by
putting the A1 and A2 numbers side by side and classifying the outcome, not
by re-testing A1.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from molfusion_backend.benchmark import analysis, metrics, protocol

ANALYSIS_VERSION = "6A.4.1"


# ---------------------------------------------------------------------------
# per-seed ranking: the stability question A1 could not ask
# ---------------------------------------------------------------------------


def per_seed_scores(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, float]]:
    """Oriented primary-metric score per (endpoint, probe, seed, representation)."""
    out: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for row in analysis.primary_metric_rows(rows):
        key = (row["endpoint"], row["probe"], int(row["seed"]))
        out[key][row["representation"]] = metrics.orient(row["metric"], row["metric_value"])
    return dict(out)


def per_seed_ranks(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, float]]:
    """Rank the representations within every endpoint x probe x seed.

    Used only for stability description. These are NOT fed to the omnibus:
    five partitions of one endpoint are five views of the same endpoint, and
    treating them as independent would inflate n from 22 to 110.
    """
    ranks: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, scores in per_seed_scores(rows).items():
        names = sorted(scores)
        values = np.array([scores[n] for n in names], dtype=float)
        order = (-values).argsort(kind="stable")
        assigned = np.empty(len(names), dtype=float)
        assigned[order] = np.arange(1, len(names) + 1, dtype=float)
        for value in np.unique(values):
            mask = values == value
            if mask.sum() > 1:
                assigned[mask] = assigned[mask].mean()
        ranks[key] = {n: float(r) for n, r in zip(names, assigned)}
    return ranks


def split_stability(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """How much each representation's rank moves across the five partitions.

    Reported per endpoint x probe x representation: the SD and full range of
    that representation's rank over the five independent splits. A large
    value means the endpoint cannot support a claim about that
    representation regardless of what the mean says.
    """
    ranks = per_seed_ranks(rows)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (endpoint, probe, _seed), row in ranks.items():
        for representation, rank in row.items():
            grouped[(endpoint, probe, representation)].append(rank)

    out = []
    for (endpoint, probe, representation), values in sorted(grouped.items()):
        array = np.array(values, dtype=float)
        out.append(
            {
                "endpoint": endpoint,
                "probe": probe,
                "representation": representation,
                "n_splits": len(array),
                "mean_rank": float(array.mean()),
                "rank_sd_across_splits": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "rank_min": float(array.min()),
                "rank_max": float(array.max()),
                "rank_range": float(array.max() - array.min()),
            }
        )
    return out


def kendall_w(rows: Sequence[dict[str, Any]], *, probe: str) -> list[dict[str, Any]]:
    """Kendall's W across the five partitions, per endpoint.

    W = 1 means the five splits ordered the seven representations
    identically; W near 0 means the ordering is essentially resampled each
    time. This is the cleanest single number for "did repartitioning change
    the answer on this endpoint?".
    """
    ranks = per_seed_ranks(rows)
    by_endpoint: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (endpoint, cell_probe, _seed), row in ranks.items():
        if cell_probe == probe:
            by_endpoint[endpoint].append(row)

    out = []
    for endpoint, per_split in sorted(by_endpoint.items()):
        names = sorted(per_split[0])
        matrix = np.array([[split[n] for n in names] for split in per_split], dtype=float)
        m, n = matrix.shape                      # m raters (splits), n items
        totals = matrix.sum(axis=0)
        s = float(((totals - totals.mean()) ** 2).sum())
        denominator = m**2 * (n**3 - n) / 12.0
        out.append(
            {
                "endpoint": endpoint,
                "probe": probe,
                "n_splits": m,
                "n_representations": n,
                "kendall_w": float(s / denominator) if denominator else float("nan"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# A1 versus A2
# ---------------------------------------------------------------------------


def classify(a1_value: float, a2_value: float, *, tolerance: float, lower_is_better: bool = True) -> str:
    """Label an A1 claim as reproduced / weakened / contradicted.

    Deliberately coarse and stated up front, so the label is not chosen
    after seeing which way the number moved.
    """
    delta = a2_value - a1_value
    if not lower_is_better:
        delta = -delta
    if abs(delta) <= tolerance:
        return "reproduced"
    return "weakened" if delta > 0 else "strengthened"


def compare_rankings(
    a1_summary: Sequence[dict[str, Any]],
    a2_summary: Sequence[dict[str, Any]],
    *,
    probe: str,
    subset: str = "all",
) -> list[dict[str, Any]]:
    """Side-by-side mean rank and position for one probe family."""
    def index(summary):
        rows = [r for r in summary if r["probe"] == probe and r.get("subset", "all") == subset]
        ordered = sorted(rows, key=lambda r: r["mean_rank"])
        return {r["representation"]: (position, r) for position, r in enumerate(ordered, 1)}

    first, second = index(a1_summary), index(a2_summary)
    out = []
    for representation in sorted(set(first) | set(second)):
        a1_position, a1_row = first.get(representation, (None, {}))
        a2_position, a2_row = second.get(representation, (None, {}))
        out.append(
            {
                "probe": probe,
                "subset": subset,
                "representation": representation,
                "a1_mean_rank": a1_row.get("mean_rank"),
                "a2_mean_rank": a2_row.get("mean_rank"),
                "a1_position": a1_position,
                "a2_position": a2_position,
                "position_change": (
                    a1_position - a2_position
                    if a1_position is not None and a2_position is not None else None
                ),
                "a1_wins": a1_row.get("wins"),
                "a2_wins": a2_row.get("wins"),
                "a1_top3": a1_row.get("top3"),
                "a2_top3": a2_row.get("top3"),
            }
        )
    return sorted(out, key=lambda r: r["a2_mean_rank"] if r["a2_mean_rank"] is not None else 99)


def leader(summary: Sequence[dict[str, Any]], *, probe: str, subset: str = "all") -> str | None:
    rows = [r for r in summary if r["probe"] == probe and r.get("subset", "all") == subset]
    return min(rows, key=lambda r: r["mean_rank"])["representation"] if rows else None


def reproduced_contrasts(
    a1_tests: Sequence[dict[str, Any]], a2_tests: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Which A1 significant pairs remain significant under A2.

    A pair that loses significance is reported as "not reproduced", never as
    "shown equal": failing to reject is not evidence of no difference, and
    A2's independent partitions carry their own noise.
    """
    def key(row):
        return (row["probe"], row["task_type"], row["a"], row["b"])

    second = {key(r): r for r in a2_tests}
    out = []
    for row in a1_tests:
        if not row.get("significant_after_holm"):
            continue
        match = second.get(key(row))
        out.append(
            {
                "probe": row["probe"],
                "task_type": row["task_type"],
                "a": row["a"],
                "b": row["b"],
                "a1_p_holm": row["p_holm"],
                "a1_effect": row["effect_size_rank_biserial"],
                "a2_p_holm": match["p_holm"] if match else None,
                "a2_effect": match["effect_size_rank_biserial"] if match else None,
                "a2_tested": match is not None,
                "reproduced": bool(match and match["significant_after_holm"]),
                "effect_direction_preserved": bool(
                    match
                    and np.sign(match["effect_size_rank_biserial"])
                    == np.sign(row["effect_size_rank_biserial"])
                ),
            }
        )
    return out


__all__ = [
    "ANALYSIS_VERSION",
    "classify",
    "compare_rankings",
    "kendall_w",
    "leader",
    "per_seed_ranks",
    "per_seed_scores",
    "reproduced_contrasts",
    "split_stability",
]
