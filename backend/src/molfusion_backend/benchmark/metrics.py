"""Metric computation and the direction normalization ranking depends on.

Two responsibilities that must not be conflated: computing a score, and
knowing whether a larger score is better. Cross-endpoint aggregation is
impossible without the second, and getting it wrong inverts a conclusion
rather than merely blurring it.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from molfusion_backend.benchmark import protocol


def is_lower_better(metric: str) -> bool:
    return metric in protocol.LOWER_IS_BETTER


def orient(metric: str, value: float) -> float:
    """A score rewritten so that larger is always better.

    Used only for ranking and comparison. Reported values stay in their
    natural direction, because an MAE printed as its own negation is a
    reporting bug waiting to be quoted.
    """
    return -value if is_lower_better(metric) else value


def classification_metrics(
    y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]
) -> dict[str, float]:
    """AUROC, AUPRC and the secondary classification metrics.

    AUPRC is not optional. ADMET endpoints are frequently imbalanced, and
    AUROC can look healthy for a model that ranks the minority class badly --
    which is usually the class the endpoint exists to identify.
    """
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    truth = np.asarray(y_true)
    if len(np.unique(truth)) < 2:
        raise ValueError(
            "classification metrics require both classes present in the evaluation "
            "partition; a single-class fold makes AUROC and AUPRC undefined"
        )

    return {
        "auroc": float(roc_auc_score(truth, y_score)),
        "auprc": float(average_precision_score(truth, y_score)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, y_pred)),
        "mcc": float(matthews_corrcoef(truth, y_pred)),
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """MAE, RMSE, R2 and Spearman.

    Spearman travels with MAE everywhere because it is unit-free: MAE is
    interpretable within an endpoint but meaningless to compare across
    endpoints measured in different units, and a benchmark spanning log
    solubility and percent binding needs at least one metric that is not.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)

    correlation = spearmanr(truth, predicted).statistic
    return {
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
        "r2": float(r2_score(truth, predicted)),
        # A constant prediction makes Spearman undefined; NaN is the honest
        # value and is preserved rather than replaced with 0.0, which would
        # read as "no correlation measured" rather than "not measurable".
        "spearman": float(correlation) if correlation == correlation else float("nan"),
    }


def primary_metric(task_type: str) -> str:
    if task_type == protocol.TASK_CLASSIFICATION:
        return protocol.PRIMARY_CLASSIFICATION_METRIC
    if task_type == protocol.TASK_REGRESSION:
        return protocol.PRIMARY_REGRESSION_METRIC
    raise ValueError(f"unknown task_type: {task_type!r}")


def rank_within_endpoint(scores: dict[str, float], metric: str) -> dict[str, float]:
    """Rank representations 1..n within one endpoint, best first.

    Ranking within an endpoint and aggregating only the ranks is what makes
    a heterogeneous suite comparable: AUROC and MAE cannot be averaged
    together, but "was this representation first or fifth here" can. Ties
    receive the average rank, so a tie neither rewards nor penalises.
    """
    if not scores:
        return {}
    oriented = {name: orient(metric, value) for name, value in scores.items()}
    ordered = sorted(oriented.items(), key=lambda item: (-item[1], item[0]))

    ranks: dict[str, float] = {}
    position = 0
    while position < len(ordered):
        stop = position + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[position][1]:
            stop += 1
        average = (position + stop + 1) / 2  # 1-based average of the tied block
        for name, _ in ordered[position:stop]:
            ranks[name] = average
        position = stop
    return ranks


def aggregate_ranks(per_endpoint: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Mean rank, median rank and win count across endpoints."""
    names = sorted({name for ranks in per_endpoint for name in ranks})
    summary: dict[str, Any] = {}
    for name in names:
        values = [ranks[name] for ranks in per_endpoint if name in ranks]
        wins = sum(1 for ranks in per_endpoint if ranks.get(name) == min(ranks.values()))
        summary[name] = {
            "endpoints": len(values),
            "mean_rank": float(np.mean(values)) if values else None,
            "median_rank": float(np.median(values)) if values else None,
            "wins": wins,
        }
    return summary


__all__ = [
    "aggregate_ranks",
    "classification_metrics",
    "is_lower_better",
    "orient",
    "primary_metric",
    "rank_within_endpoint",
    "regression_metrics",
]
