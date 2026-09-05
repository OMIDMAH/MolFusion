"""The long-format result schema and its deterministic serialization.

One row per (dataset, endpoint, task, split, seed, representation, model,
metric). Long rather than wide on purpose: a wide table with one column per
representation has to be rewritten whenever the representation set changes,
cannot hold per-split rows without becoming three-dimensional, and forces a
choice of aggregation at write time. Publication tables are projections of
this, generated later; the raw rows stay the source of truth.
"""

import csv
import json
import platform
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from molfusion_backend.benchmark import protocol

RESULT_SCHEMA_VERSION = 1

RESULT_FIELDS = (
    "protocol_version",
    "dataset",
    "endpoint",
    "task_type",
    "split_id",
    "split_strategy",
    "seed",
    "representation",
    "representation_version",
    "model",
    "probe",
    "metric",
    "value",
    "n_train",
    "n_valid",
    "n_test",
    "feature_dim",
    "feature_failures",
    "hyperparameters",
    "fit_seconds",
    "predict_seconds",
    "feature_seconds",
)


@dataclass(frozen=True)
class ResultRow:
    """One measured number, with everything needed to interpret it."""

    dataset: str
    endpoint: str
    task_type: str
    split_id: str
    split_strategy: str
    seed: int
    representation: str
    representation_version: str
    model: str
    probe: str
    metric: str
    value: float
    n_train: int
    n_valid: int
    n_test: int
    feature_dim: int
    feature_failures: int
    hyperparameters: str = ""
    fit_seconds: float | None = None
    predict_seconds: float | None = None
    # Kept apart from fit/predict time throughout: feature generation is a
    # property of the representation, model fitting a property of the head,
    # and a single blended "cost" number would hide which one dominates.
    feature_seconds: float | None = None
    protocol_version: str = protocol.PROTOCOL_VERSION

    def as_row(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in RESULT_FIELDS}


@dataclass
class RunManifest:
    """Reproducibility metadata for one benchmark run.

    Recorded next to the rows rather than in a notebook, because a result
    whose software versions and protocol are not recorded cannot be
    reproduced or superseded -- only re-run and hoped about.
    """

    protocol_version: str = protocol.PROTOCOL_VERSION
    schema_version: int = RESULT_SCHEMA_VERSION
    started_at: str = ""
    software: dict[str, Any] = field(default_factory=dict)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    representations: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def as_report(self) -> dict[str, Any]:
        return asdict(self)


def software_versions() -> dict[str, Any]:
    """The versions any reader needs to reproduce or date a result."""
    import numpy
    import rdkit

    from molfusion_backend.corpus.provenance import git_commit, working_tree_is_clean

    marker = Path(__file__).resolve().parent
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "molfusion_git_commit": git_commit(marker),
        "molfusion_git_working_tree_clean": working_tree_is_clean(marker),
    }
    for name in ("sklearn", "scipy"):
        try:
            module = __import__(name)
            versions[name] = module.__version__
        except ImportError:  # pragma: no cover - both are dev dependencies
            versions[name] = None
    return versions


def new_manifest(**kwargs: Any) -> RunManifest:
    manifest = RunManifest(
        started_at=datetime.now(timezone.utc).isoformat(),
        software=software_versions(),
        **kwargs,
    )
    return manifest


def write_results(path: Path, rows: Iterable[ResultRow]) -> int:
    """Write the long-format table. UTF-8, LF, fixed column order.

    Rows are written in the order given; the caller controls that order, and
    the benchmark iterates datasets, splits and representations in a fixed
    sequence, so two runs of the same protocol produce the same file order.
    """
    materialized = list(rows)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(RESULT_FIELDS), lineterminator="\n", extrasaction="raise"
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow(row.as_row())
    return len(materialized)


def write_manifest(path: Path, manifest: RunManifest) -> None:
    text = json.dumps(manifest.as_report(), indent=2, sort_keys=False, ensure_ascii=False)
    Path(path).write_bytes((text + "\n").encode("utf-8"))


def read_results(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_across_splits(
    rows: Sequence[ResultRow], metric: str
) -> dict[tuple[str, str], dict[str, float]]:
    """Mean, standard deviation and spread per (endpoint, representation).

    A convenience over the raw rows, never a replacement for them: the
    per-split values stay in the table so any later analysis recomputes from
    them rather than trusting a summary.
    """
    import numpy as np

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.metric != metric:
            continue
        grouped.setdefault((row.endpoint, row.representation), []).append(row.value)

    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "n_splits": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return summary


__all__ = [
    "RESULT_FIELDS",
    "RESULT_SCHEMA_VERSION",
    "ResultRow",
    "RunManifest",
    "new_manifest",
    "read_results",
    "software_versions",
    "summarize_across_splits",
    "write_manifest",
    "write_results",
]
