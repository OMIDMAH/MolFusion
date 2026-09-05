"""Track A1 scientific analysis: aggregation, ranking, statistics.

Reads the completed raw result matrix and produces derived tables. The raw
matrix is treated as strictly read-only -- this module never writes to it,
and :func:`verify_raw_results` refuses to proceed if its scientific identity
has moved.

Three decisions shape everything downstream, and each is made once here
rather than implicitly in a dozen places:

**The statistical unit is the endpoint.** Track A1 has five
train/validation realizations scored on ONE fixed test set, so the five
values are not five independent observations -- treating them as such would
be pseudoreplication and would inflate every test by a factor of five. They
are aggregated to a single value per endpoint x representation x probe
before any cross-endpoint comparison.

**Linear and nonlinear probes are never pooled.** How much information a
representation makes linearly accessible and how much a flexible model can
extract from it are different questions, and averaging them answers
neither.

**Raw metric values are never averaged across endpoints.** AUROC, AUPRC,
MAE and Spearman have different units, directions and attainable ranges.
Only ranks -- computed within an endpoint, direction-aware -- cross endpoint
boundaries.
"""

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from molfusion_backend.benchmark import metrics, protocol

ANALYSIS_VERSION = "6A.3.1"

#: How the five per-seed test scores become one endpoint-level value. The
#: mean is used rather than the median because five is a small sample and
#: the median of five discards most of it; the spread is reported alongside
#: rather than folded in.
SEED_AGGREGATION = "mean of the 5 train/validation realizations on the fixed official test set"

BOOTSTRAP_SEED = 0


class AnalysisError(RuntimeError):
    """The raw results are not the results this analysis was written for."""


# ---------------------------------------------------------------------------
# loading and immutability
# ---------------------------------------------------------------------------

NUMERIC_COLUMNS = (
    "seed", "metric_value", "n_train", "n_valid", "n_test",
    "feature_dim", "feature_failures", "feature_seconds",
    "fit_seconds", "selection_seconds", "predict_seconds",
    "validation_predict_seconds",
)


def load_raw_results(path: Path) -> list[dict[str, Any]]:
    """Read the raw long-format result rows, typed but otherwise untouched."""
    with open(path, encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        for column in NUMERIC_COLUMNS:
            if column in row and row[column] != "":
                row[column] = float(row[column]) if "." in row[column] or "e" in row[column].lower() else int(row[column])
        row["metric_value"] = float(row["metric_value"])
        row["seed"] = int(row["seed"])
    return rows


def verify_raw_results(
    rows: Sequence[dict[str, Any]],
    *,
    expected_identity: str,
    expected_rows: int,
) -> dict[str, Any]:
    """Refuse to analyse a result set that is not the one that was audited."""
    from molfusion_backend.benchmark import runner

    identity = runner.scientific_identity(rows)
    if identity != expected_identity:
        raise AnalysisError(
            f"raw scientific identity {identity} != expected {expected_identity}; "
            "the completed result matrix has changed and the analysis is void"
        )
    if len(rows) != expected_rows:
        raise AnalysisError(f"{len(rows)} raw rows, expected {expected_rows}")
    if list(rows[0]) != list(runner.RESULT_COLUMNS):
        raise AnalysisError("raw result schema does not match the frozen column order")

    endpoints = {r["endpoint"] for r in rows}
    representations = {r["representation"] for r in rows}
    probes = {r["probe"] for r in rows}
    seeds = {r["seed"] for r in rows}
    return {
        "scientific_identity": identity,
        "rows": len(rows),
        "endpoints": len(endpoints),
        "representations": len(representations),
        "probes": sorted(probes),
        "seeds": sorted(seeds),
        "tracks": sorted({r["track"] for r in rows}),
    }


# ---------------------------------------------------------------------------
# endpoint-level aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EndpointScore:
    """One endpoint x representation x probe, aggregated over the five runs."""

    endpoint: str
    task_type: str
    representation: str
    probe: str
    metric: str
    mean: float
    std: float
    median: float
    minimum: float
    maximum: float
    n_runs: int
    oriented_mean: float

    def as_row(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "task_type": self.task_type,
            "representation": self.representation,
            "probe": self.probe,
            "metric": self.metric,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.minimum,
            "max": self.maximum,
            "n_runs": self.n_runs,
            "oriented_mean": self.oriented_mean,
        }


def primary_metric_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only each endpoint's frozen primary metric.

    The metric is read from the row's own ``molfusion_primary_metric``
    column, which the runner wrote from the frozen protocol before any score
    existed. It is never chosen here, after the numbers are visible.
    """
    return [r for r in rows if r["metric"] == r["molfusion_primary_metric"]]


def aggregate_seeds(rows: Sequence[dict[str, Any]]) -> list[EndpointScore]:
    """Collapse the five realizations into one value per cell.

    This is where pseudoreplication is prevented: everything downstream sees
    one number per endpoint x representation x probe, so no cross-endpoint
    test can mistake five correlated re-runs for five independent results.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in primary_metric_rows(rows):
        grouped[(row["endpoint"], row["representation"], row["probe"])].append(row)

    scores: list[EndpointScore] = []
    for (endpoint, representation, probe), group in sorted(grouped.items()):
        values = np.array([g["metric_value"] for g in group], dtype=float)
        metric = group[0]["metric"]
        mean = float(values.mean())
        scores.append(
            EndpointScore(
                endpoint=endpoint,
                task_type=group[0]["task_type"],
                representation=representation,
                probe=probe,
                metric=metric,
                mean=mean,
                std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                median=float(np.median(values)),
                minimum=float(values.min()),
                maximum=float(values.max()),
                n_runs=len(values),
                oriented_mean=metrics.orient(metric, mean),
            )
        )
    return scores


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


def rank_endpoint(scores: Sequence[EndpointScore]) -> dict[str, float]:
    """Rank representations within one endpoint, 1 = best, average ties.

    Ties take the average rank rather than being broken. Breaking them --
    by representation name, or by any other property of the contestant --
    would manufacture an ordering the data does not contain.
    """
    if not scores:
        return {}
    oriented = np.array([s.oriented_mean for s in scores], dtype=float)
    order = (-oriented).argsort(kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    # average ties
    for value in np.unique(oriented):
        mask = oriented == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return {s.representation: float(r) for s, r in zip(scores, ranks)}


def rank_table(scores: Sequence[EndpointScore]) -> dict[tuple[str, str], dict[str, float]]:
    """Endpoint-level ranks, keyed by (endpoint, probe)."""
    grouped: dict[tuple[str, str], list[EndpointScore]] = defaultdict(list)
    for score in scores:
        grouped[(score.endpoint, score.probe)].append(score)
    return {key: rank_endpoint(group) for key, group in sorted(grouped.items())}


def summarise_ranks(
    ranks: dict[tuple[str, str], dict[str, float]],
    *,
    probe: str,
    endpoints: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Cross-endpoint rank summary for one probe family."""
    keep = set(endpoints) if endpoints is not None else None
    per_representation: dict[str, list[float]] = defaultdict(list)
    for (endpoint, cell_probe), row in ranks.items():
        if cell_probe != probe or (keep is not None and endpoint not in keep):
            continue
        for representation, rank in row.items():
            per_representation[representation].append(rank)

    summary = []
    for representation, values in per_representation.items():
        array = np.array(values, dtype=float)
        summary.append(
            {
                "probe": probe,
                "representation": representation,
                "n_endpoints": len(array),
                "mean_rank": float(array.mean()),
                "median_rank": float(np.median(array)),
                "rank_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "wins": int((array == 1.0).sum()),
                "top2": int((array <= 2.0).sum()),
                "top3": int((array <= 3.0).sum()),
                "worst_rank": float(array.max()),
            }
        )
    return sorted(summary, key=lambda r: r["mean_rank"])


# ---------------------------------------------------------------------------
# pairwise descriptive comparison
# ---------------------------------------------------------------------------


def pairwise_wins(
    scores: Sequence[EndpointScore],
    *,
    probe: str,
    task_type: str | None = None,
    endpoints: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Endpoint-level win/loss/tie counts for every representation pair."""
    keep = set(endpoints) if endpoints is not None else None
    by_endpoint: dict[str, dict[str, float]] = defaultdict(dict)
    for score in scores:
        if score.probe != probe:
            continue
        if task_type is not None and score.task_type != task_type:
            continue
        if keep is not None and score.endpoint not in keep:
            continue
        by_endpoint[score.endpoint][score.representation] = score.oriented_mean

    representations = sorted(protocol.TRACK_A_REPRESENTATIONS)
    out = []
    for i, a in enumerate(representations):
        for b in representations[i + 1:]:
            wins = losses = ties = 0
            for values in by_endpoint.values():
                if a not in values or b not in values:
                    continue
                if values[a] > values[b]:
                    wins += 1
                elif values[b] > values[a]:
                    losses += 1
                else:
                    ties += 1
            total = wins + losses + ties
            out.append(
                {
                    "probe": probe,
                    "task_type": task_type or "all",
                    "a": a,
                    "b": b,
                    "a_better": wins,
                    "b_better": losses,
                    "ties": ties,
                    "n_endpoints": total,
                    "a_win_rate": wins / total if total else float("nan"),
                }
            )
    return out


# ---------------------------------------------------------------------------
# nonlinear gain
# ---------------------------------------------------------------------------


def nonlinear_gain(scores: Sequence[EndpointScore]) -> list[dict[str, Any]]:
    """How much each representation gains from a flexible model.

    Two transparent measures, because neither alone is enough:

    ``normalised_gain``  within each endpoint, all 14 (representation, probe)
                         oriented scores are min-max scaled to [0, 1], and the
                         gain is the nonlinear value minus the linear one.
                         Scaling *within* the endpoint is what makes values
                         from AUROC and MAE endpoints comparable at all;
                         scaling across all 14 rather than per probe keeps the
                         two probes on one common scale.
    ``rank_gain``        linear rank minus nonlinear rank, positive meaning
                         the representation is placed better under the
                         nonlinear probe. Immune to metric scale, but only
                         measures position relative to competitors.
    """
    by_endpoint: dict[str, list[EndpointScore]] = defaultdict(list)
    for score in scores:
        by_endpoint[score.endpoint].append(score)

    ranks = rank_table(scores)
    out = []
    for endpoint, group in sorted(by_endpoint.items()):
        oriented = np.array([s.oriented_mean for s in group], dtype=float)
        low, high = float(oriented.min()), float(oriented.max())
        span = high - low
        normalised = {
            (s.representation, s.probe): ((s.oriented_mean - low) / span if span else 0.5)
            for s in group
        }
        for representation in protocol.TRACK_A_REPRESENTATIONS:
            linear = normalised.get((representation, protocol.PROBE_LINEAR))
            nonlinear = normalised.get((representation, protocol.PROBE_NONLINEAR))
            if linear is None or nonlinear is None:
                continue
            linear_rank = ranks[(endpoint, protocol.PROBE_LINEAR)][representation]
            nonlinear_rank = ranks[(endpoint, protocol.PROBE_NONLINEAR)][representation]
            out.append(
                {
                    "endpoint": endpoint,
                    "task_type": group[0].task_type,
                    "representation": representation,
                    "linear_normalised": linear,
                    "nonlinear_normalised": nonlinear,
                    "normalised_gain": nonlinear - linear,
                    "linear_rank": linear_rank,
                    "nonlinear_rank": nonlinear_rank,
                    "rank_gain": linear_rank - nonlinear_rank,
                }
            )
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def friedman(
    scores: Sequence[EndpointScore], *, probe: str, task_type: str | None = None
) -> dict[str, Any]:
    """Omnibus test across the seven representations, on endpoint-level values.

    One observation per endpoint per representation -- never one per seed.
    """
    from scipy.stats import friedmanchisquare

    representations = list(protocol.TRACK_A_REPRESENTATIONS)
    by_endpoint: dict[str, dict[str, float]] = defaultdict(dict)
    for score in scores:
        if score.probe != probe:
            continue
        if task_type is not None and score.task_type != task_type:
            continue
        by_endpoint[score.endpoint][score.representation] = score.oriented_mean

    complete = [e for e, v in sorted(by_endpoint.items()) if len(v) == len(representations)]
    if len(complete) < 3:
        return {"probe": probe, "task_type": task_type or "all",
                "n_endpoints": len(complete), "runnable": False}

    columns = [[by_endpoint[e][r] for e in complete] for r in representations]
    statistic, p_value = friedmanchisquare(*columns)
    return {
        "probe": probe,
        "task_type": task_type or "all",
        "n_endpoints": len(complete),
        "n_representations": len(representations),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "reject_at_alpha": bool(p_value < protocol.ALPHA),
        "alpha": protocol.ALPHA,
        "runnable": True,
    }


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, order preserved."""
    indexed = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for step, index in enumerate(indexed):
        candidate = (total - step) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def rank_biserial(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation.

    (W+ - W-) / (W+ + W-) over the signed ranks of the non-zero differences:
    +1 means the first member of the pair was better everywhere, -1 the
    reverse, 0 an even split.
    """
    from scipy.stats import rankdata

    values = np.array([d for d in differences if d != 0.0], dtype=float)
    if values.size == 0:
        return 0.0
    ranks = rankdata(np.abs(values))
    positive = ranks[values > 0].sum()
    negative = ranks[values < 0].sum()
    total = positive + negative
    return float((positive - negative) / total) if total else 0.0


def pairwise_tests(
    scores: Sequence[EndpointScore], *, probe: str, task_type: str | None = None
) -> list[dict[str, Any]]:
    """Holm-corrected paired Wilcoxon tests, each with an effect size.

    Run only when the omnibus rejects; the caller enforces that. No p-value
    is returned without its effect size, because a detectable ordering and
    an ordering worth acting on are different claims.
    """
    from scipy.stats import wilcoxon

    representations = sorted(protocol.TRACK_A_REPRESENTATIONS)
    by_endpoint: dict[str, dict[str, float]] = defaultdict(dict)
    for score in scores:
        if score.probe != probe:
            continue
        if task_type is not None and score.task_type != task_type:
            continue
        by_endpoint[score.endpoint][score.representation] = score.oriented_mean

    endpoints = [e for e, v in sorted(by_endpoint.items()) if len(v) == len(representations)]
    results = []
    for i, a in enumerate(representations):
        for b in representations[i + 1:]:
            differences = [by_endpoint[e][a] - by_endpoint[e][b] for e in endpoints]
            array = np.array(differences, dtype=float)
            if np.all(array == 0):
                statistic, p_value = float("nan"), 1.0
            else:
                statistic, p_value = wilcoxon(array, zero_method="wilcox")
                statistic, p_value = float(statistic), float(p_value)
            results.append(
                {
                    "probe": probe,
                    "task_type": task_type or "all",
                    "a": a,
                    "b": b,
                    "n_endpoints": len(endpoints),
                    "median_difference": float(np.median(array)),
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_value,
                    "effect_size_rank_biserial": rank_biserial(differences),
                }
            )

    adjusted = holm([r["p_raw"] for r in results])
    for row, value in zip(results, adjusted):
        row["p_holm"] = value
        row["significant_after_holm"] = bool(value < protocol.ALPHA)
    return results


def bootstrap_mean_rank(
    ranks: dict[tuple[str, str], dict[str, float]],
    *,
    probe: str,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Bootstrap confidence intervals for mean rank, resampling ENDPOINTS.

    Endpoints are the unit of resampling because the claim being bounded is
    a cross-endpoint one. Resampling molecules would answer a different
    question and would badly understate the uncertainty here.
    """
    per_endpoint: dict[str, dict[str, float]] = {}
    for (endpoint, cell_probe), row in ranks.items():
        if cell_probe == probe:
            per_endpoint[endpoint] = row
    endpoints = sorted(per_endpoint)
    representations = sorted(protocol.TRACK_A_REPRESENTATIONS)

    matrix = np.array(
        [[per_endpoint[e].get(r, np.nan) for r in representations] for e in endpoints],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(endpoints), size=(resamples, len(endpoints)))
    means = np.nanmean(matrix[draws], axis=1)

    out = []
    for index, representation in enumerate(representations):
        column = means[:, index]
        out.append(
            {
                "probe": probe,
                "representation": representation,
                "mean_rank": float(np.nanmean(matrix[:, index])),
                "ci_lower_95": float(np.percentile(column, 2.5)),
                "ci_upper_95": float(np.percentile(column, 97.5)),
                "bootstrap_resamples": resamples,
                "bootstrap_seed": seed,
                "resampling_unit": "endpoint",
            }
        )
    return sorted(out, key=lambda r: r["mean_rank"])


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def analysis_identity(*, raw_identity: str, configuration: dict[str, Any]) -> str:
    """Deterministic identity for the derived analysis.

    Covers the raw matrix it was computed from and the analysis
    configuration; excludes timestamps, paths and machine names, so the same
    analysis of the same results reproduces the same identity.
    """
    payload = json.dumps(
        {
            "analysis_version": ANALYSIS_VERSION,
            "raw_scientific_identity": raw_identity,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "configuration": configuration,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_table(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """Write one derived table and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return hashlib.sha256(b"").hexdigest()
    fields = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "ANALYSIS_VERSION",
    "BOOTSTRAP_SEED",
    "SEED_AGGREGATION",
    "AnalysisError",
    "EndpointScore",
    "aggregate_seeds",
    "analysis_identity",
    "bootstrap_mean_rank",
    "friedman",
    "holm",
    "load_raw_results",
    "nonlinear_gain",
    "pairwise_tests",
    "pairwise_wins",
    "primary_metric_rows",
    "rank_biserial",
    "rank_endpoint",
    "rank_table",
    "summarise_ranks",
    "verify_raw_results",
    "write_table",
]
