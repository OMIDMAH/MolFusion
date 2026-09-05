"""Track A2 execution: MolFusion's independent scaffold repartitioning.

Track A1 asked what the seven representations do on TDC's official
partitions. It answered with one genuinely surprising result -- the leader
reverses between probes -- resting on a single scaffold partition per
endpoint. A2 exists to test whether that survives repartitioning.

The differences from A1 are deliberate and are the whole point:

  * **five independent test partitions**, not one fixed test set scored five
    times, so the spread across seeds now includes which molecules are
    evaluated;
  * **the full MolFusion cleaning policy** -- canonicalize, collapse
    agreeing duplicates, drop conflicting groups entirely -- so A2 also
    measures what that cleaning does to the ranking;
  * 70/10/20 Bemis-Murcko splits over the cleaned universe, seeds 0-4.

Everything else is held fixed: the same seven representations at the same
versions, the same two probe families, the same frozen hyperparameter grids,
the same endpoint primary metrics and directions, the same finite-value
handling accepted in A1. A2 is a robustness test, not a second chance at
tuning, and nothing here reads an A2 test score to make a decision.

Track A1's code is not imported or modified: A1 is frozen, and a shared
helper that later needed changing for A2 would put A1's reproducibility at
risk. Nothing here imports PyTDC.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from molfusion_backend.agents import registry as agent_registry
from molfusion_backend.benchmark import (
    datasets,
    feature_store,
    metrics,
    pipelines,
    protocol,
    release,
    splits,
)

TRACK = protocol.TRACK_A2
SEEDS = tuple(protocol.TRACK_A2_SEEDS)


class TrackA2Error(RuntimeError):
    """Execution cannot proceed on the terms the protocol requires."""


# ---------------------------------------------------------------------------
# endpoint loading, with the full cleaning policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanedEndpoint:
    """One endpoint after the full MolFusion cleaning policy."""

    name: str
    task_type: str
    tdc_official_metric: str
    molfusion_primary_metric: str
    molecules: tuple[Any, ...]
    canonical_smiles: tuple[str, ...]
    labels: tuple[float, ...]
    ingestion: dict[str, Any]
    raw_rows: int

    @property
    def size(self) -> int:
        return len(self.canonical_smiles)


def load_cleaned_endpoint(
    name: str, *, frozen_dir: Path, manifest: dict[str, Any]
) -> CleanedEndpoint:
    """Load an endpoint and apply the full cleaning policy.

    Unlike A1, train_val and test are pooled first: A2 draws its own
    partitions, so TDC's partition boundary is not meaningful here and
    keeping it would silently constrain the repartitioning.
    """
    entry = manifest["endpoints"][name]
    task_type = entry["task_type"]

    records: list[tuple[str, float | None]] = []
    raw_rows = 0
    for part in ("train_val", "test"):
        path = frozen_dir / name / f"{part}.csv"
        actual = release.sha256_file(path)
        expected = entry[part]["sha256"]
        if actual != expected:
            raise TrackA2Error(
                f"{name}/{part}: frozen file checksum {actual} != manifest {expected}; "
                "the benchmark data is not the data this protocol was frozen against"
            )
        _, rows = release.read_frozen_csv(path)
        raw_rows += len(rows)
        for row in rows:
            label = row[2]
            records.append(
                (row[1], None if label == "" or label.lower() in ("nan", "none") else float(label))
            )

    molecules, audit = datasets.build_dataset(records, task_type=task_type)
    included, reasons = datasets.check_inclusion(molecules, task_type=task_type)
    if not included:
        raise TrackA2Error(
            f"{name}: fails the frozen inclusion criteria after cleaning: {reasons}"
        )

    return CleanedEndpoint(
        name=name,
        task_type=task_type,
        tdc_official_metric=entry["tdc_official_metric"],
        molfusion_primary_metric=entry["molfusion_primary_metric"],
        molecules=tuple(molecules),
        canonical_smiles=tuple(m.canonical_smiles for m in molecules),
        labels=tuple(float(m.label) for m in molecules),
        ingestion=audit.as_report(),
        raw_rows=raw_rows,
    )


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class A2Split:
    """One independent scaffold partition: its own train, validation AND test."""

    seed: int
    split_id: str
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    audit: dict[str, Any]
    test_set_sha256: str


def build_splits(endpoint: CleanedEndpoint) -> dict[int, A2Split]:
    """Five independent 70/10/20 scaffold partitions, audited on the way out.

    Each seed draws a different test set -- that is the difference from A1
    and the reason A2 can say anything about partition sensitivity.
    """
    out: dict[int, A2Split] = {}
    for seed in SEEDS:
        split = splits.scaffold_split(list(endpoint.molecules), seed=seed)
        audit = splits.audit_split(list(endpoint.molecules), split)

        overlaps = list(audit["molecule_overlap"].values()) + list(
            audit["scaffold_overlap"].values()
        )
        if any(overlaps):
            raise TrackA2Error(
                f"{endpoint.name} seed {seed}: partitions overlap "
                f"(molecules={audit['molecule_overlap']}, scaffolds={audit['scaffold_overlap']})"
            )
        if not split.train or not split.test:
            raise TrackA2Error(
                f"{endpoint.name} seed {seed}: empty train or test partition"
            )

        out[seed] = A2Split(
            seed=seed,
            split_id=protocol.split_id(TRACK, seed),
            train=split.train,
            validation=split.validation,
            test=split.test,
            audit=audit,
            test_set_sha256=release.molecule_set_identity(
                [endpoint.canonical_smiles[i] for i in split.test]
            ),
        )
    return out


def split_distinctness(splits_by_seed: dict[int, A2Split]) -> dict[str, Any]:
    """Evidence that the five partitions really are different.

    A2's whole claim rests on the test sets differing across seeds. If they
    happened to coincide, A2 would be A1 with extra steps, so it is checked
    rather than assumed.
    """
    identities = {seed: s.test_set_sha256 for seed, s in splits_by_seed.items()}
    distinct = len(set(identities.values()))
    overlaps = []
    seeds = sorted(splits_by_seed)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            first = set(splits_by_seed[a].test)
            second = set(splits_by_seed[b].test)
            union = first | second
            overlaps.append(len(first & second) / len(union) if union else 0.0)
    return {
        "test_identities": identities,
        "distinct_test_sets": distinct,
        "all_test_sets_distinct": distinct == len(splits_by_seed),
        "mean_pairwise_test_jaccard": float(np.mean(overlaps)) if overlaps else 0.0,
        "max_pairwise_test_jaccard": float(np.max(overlaps)) if overlaps else 0.0,
    }


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def artifact_identity_for(representation: str) -> str | None:
    """Verified artifact identity for artifact-backed agents, else None."""
    if representation != "smiles_tfidf_4096":
        return None
    from molfusion_backend.tfidf import contract as tfidf_contract
    from molfusion_backend.tfidf.loader import load_tfidf_artifact

    artifact = load_tfidf_artifact(
        tfidf_contract.ARTIFACT_ID,
        tfidf_contract.ARTIFACT_VERSION,
        artifact_type=tfidf_contract.ARTIFACT_TYPE,
    )
    return "\x1f".join(
        (
            tfidf_contract.ARTIFACT_TYPE,
            tfidf_contract.ARTIFACT_ID,
            tfidf_contract.ARTIFACT_VERSION,
            artifact.fit_corpus_sha256 or "",
        )
    )


@dataclass
class FeatureResult:
    matrix: np.ndarray
    succeeded: tuple[int, ...]
    failures: dict[int, str]
    dimension: int
    cache_hit: bool
    seconds: float
    cache_key: str
    nonzero_fraction: float | None


def features_for(
    endpoint: CleanedEndpoint,
    representation: str,
    *,
    store: feature_store.FeatureStore,
    release_identity: str,
) -> FeatureResult:
    """Compute or load one endpoint x representation matrix.

    The cache key includes the row identity, which is computed over the
    *cleaned* molecule list -- so A2 can never be served an A1 matrix, and
    vice versa, even though both use the same release and agents.
    """
    from molfusion_backend.chemistry import parse_smiles

    agent = agent_registry.get(representation)
    if agent.output_structure != "vector":
        raise TrackA2Error(
            f"{representation!r} is a {agent.output_structure!r} agent; Track A "
            "covers fixed-length vectors only"
        )

    rows = feature_store.row_identity(endpoint.canonical_smiles)
    key = feature_store.matrix_cache_key(
        release_identity=release_identity,
        endpoint=f"{TRACK}:{endpoint.name}",
        agent_id=agent.id,
        agent_version=agent.version,
        output_dim=int(agent.output_dim or 0),
        normalization_id=protocol.CANONICALIZATION_ID,
        row_identity_sha256=rows,
        artifact_identity=artifact_identity_for(representation),
    )
    expect = {
        "cache_schema_version": feature_store.CACHE_SCHEMA_VERSION,
        "release_identity": release_identity,
        "endpoint": f"{TRACK}:{endpoint.name}",
        "agent_id": agent.id,
        "agent_version": agent.version,
        "output_dim": int(agent.output_dim or 0),
        "row_identity_sha256": rows,
        "artifact_identity": artifact_identity_for(representation),
    }

    started = time.perf_counter()
    cached = store.load(key, expect=expect)
    if cached is not None:
        return FeatureResult(
            matrix=cached.matrix,
            succeeded=cached.succeeded,
            failures=cached.failures,
            dimension=cached.dimension,
            cache_hit=True,
            seconds=time.perf_counter() - started,
            cache_key=key,
            nonzero_fraction=cached.metadata.get("nonzero_fraction"),
        )

    started = time.perf_counter()
    vectors: list[np.ndarray] = []
    succeeded: list[int] = []
    failures: dict[int, str] = {}
    for index, smiles in enumerate(endpoint.canonical_smiles):
        mol, error = parse_smiles(smiles)
        if mol is None:
            failures[index] = error or "RDKit could not parse the molecule"
            continue
        try:
            vectors.append(np.asarray(agent.compute(mol), dtype=np.float64))
            succeeded.append(index)
        except ValueError as exc:
            failures[index] = str(exc)
    seconds = time.perf_counter() - started

    matrix = np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float64)
    nonzero = float(np.count_nonzero(matrix) / matrix.size) if matrix.size else None
    store.store(
        key,
        matrix=matrix,
        succeeded=succeeded,
        failures=failures,
        metadata={
            **expect,
            "track": TRACK,
            "representation": representation,
            "value_type": agent.value_type,
            "feature_seconds": seconds,
            "nonzero_fraction": nonzero,
        },
    )
    return FeatureResult(
        matrix=matrix,
        succeeded=tuple(succeeded),
        failures=failures,
        dimension=int(matrix.shape[1]) if matrix.size else 0,
        cache_hit=False,
        seconds=seconds,
        cache_key=key,
        nonzero_fraction=nonzero,
    )


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------


def _score(model, x, y, task_type: str) -> dict[str, float]:
    if task_type == protocol.TASK_CLASSIFICATION:
        return metrics.classification_metrics(y, model.predict_proba(x)[:, 1], model.predict(x))
    return metrics.regression_metrics(y, model.predict(x))


def run_cell(
    *,
    endpoint: CleanedEndpoint,
    representation: str,
    probe: str,
    splits_by_seed: dict[int, A2Split],
    features: FeatureResult,
    release_identity: str,
) -> dict[str, Any]:
    """Execute one A2 cell over all five independent partitions.

    Selection reads validation only; the test partition of each split is
    read exactly once, after the winner is chosen. Because each seed has its
    own test set, that discipline has to hold five times rather than once.
    """
    task_type = endpoint.task_type
    primary = endpoint.molfusion_primary_metric
    grid = pipelines.hyperparameter_grid(probe, task_type)

    row_of = {index: position for position, index in enumerate(features.succeeded)}
    labels = np.asarray(endpoint.labels, dtype=np.float64)

    def block(indices: Sequence[int]):
        rows = [row_of[i] for i in indices if i in row_of]
        kept = [i for i in indices if i in row_of]
        x = features.matrix[rows]
        y = labels[kept]
        if task_type == protocol.TASK_CLASSIFICATION:
            y = y.astype(int)
        return x, y, len(rows)

    rows_out: list[dict[str, Any]] = []
    timings: list[dict[str, float]] = []

    for seed in sorted(splits_by_seed):
        split = splits_by_seed[seed]
        x_train, y_train, n_train = block(split.train)
        x_valid, y_valid, n_valid = block(split.validation)
        x_test, y_test, n_test = block(split.test)
        if not n_train or not n_test:
            raise TrackA2Error(
                f"{endpoint.name}/{representation}/{probe} seed {seed}: "
                f"empty train ({n_train}) or test ({n_test}) partition"
            )

        best_params: dict[str, Any] = {}
        best_score = -np.inf
        selection_seconds = 0.0
        for candidate in grid:
            model = pipelines.build_pipeline(
                representation=representation, probe=probe,
                task_type=task_type, hyperparameters=candidate,
            )
            started = time.perf_counter()
            model.fit(x_train, y_train)
            selection_seconds += time.perf_counter() - started
            try:
                scored = _score(model, x_valid, y_valid, task_type)[primary]
            except ValueError:
                continue
            oriented = metrics.orient(primary, scored)
            if oriented > best_score:
                best_score, best_params = oriented, candidate

        model = pipelines.build_pipeline(
            representation=representation, probe=probe,
            task_type=task_type, hyperparameters=best_params,
        )
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        if n_valid:
            _score(model, x_valid, y_valid, task_type)
        validation_predict_seconds = time.perf_counter() - started

        started = time.perf_counter()
        scores = _score(model, x_test, y_test, task_type)
        test_predict_seconds = time.perf_counter() - started

        timings.append(
            {
                "seed": seed,
                "selection_seconds": selection_seconds,
                "fit_seconds": fit_seconds,
                "validation_predict_seconds": validation_predict_seconds,
                "test_predict_seconds": test_predict_seconds,
            }
        )

        for metric, value in sorted(scores.items()):
            rows_out.append(
                {
                    "benchmark_release": release_identity,
                    "track": TRACK,
                    "endpoint": endpoint.name,
                    "task_type": task_type,
                    "split_id": split.split_id,
                    "split_strategy": protocol.TRACK_A2_SPLIT_STRATEGY,
                    "seed": seed,
                    "representation": representation,
                    "agent_version": agent_registry.get(representation).version,
                    "model_family": pipelines.model_name(probe, task_type),
                    "probe": probe,
                    "hyperparameters": json.dumps(best_params, sort_keys=True),
                    "metric": metric,
                    "metric_value": value,
                    "tdc_official_metric": endpoint.tdc_official_metric,
                    "molfusion_primary_metric": primary,
                    "n_train": n_train,
                    "n_valid": n_valid,
                    "n_test": n_test,
                    "feature_dim": features.dimension,
                    "feature_failures": len(features.failures),
                    "feature_seconds": features.seconds,
                    "feature_cache_hit": features.cache_hit,
                    "fit_seconds": fit_seconds,
                    "selection_seconds": selection_seconds,
                    "predict_seconds": test_predict_seconds,
                    "validation_predict_seconds": validation_predict_seconds,
                }
            )

    return {
        "cell": {
            "track": TRACK,
            "endpoint": endpoint.name,
            "representation": representation,
            "probe": probe,
        },
        "rows": rows_out,
        "timings": timings,
        "feature_cache_key": features.cache_key,
        "feature_cache_hit": features.cache_hit,
        "feature_failures": len(features.failures),
        "feature_dimension": features.dimension,
        "nonzero_fraction": features.nonzero_fraction,
        "seeds": sorted(splits_by_seed),
    }


__all__ = [
    "SEEDS",
    "TRACK",
    "A2Split",
    "CleanedEndpoint",
    "FeatureResult",
    "TrackA2Error",
    "artifact_identity_for",
    "build_splits",
    "features_for",
    "load_cleaned_endpoint",
    "run_cell",
    "split_distinctness",
]
