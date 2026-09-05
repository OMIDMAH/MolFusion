"""Track A1 execution: the official, TDC-comparable benchmark.

Phase 6A.1 established what TDC's protocol actually is and froze it. This
module executes it, and nothing else:

  * the shipped ``test.csv``, unchanged and identical at every seed;
  * train and validation drawn from ``train_val.csv`` by TDC's own splitter,
    replayed from the frozen membership file rather than recomputed;
  * seeds 1-5, which move the train/validation boundary and nothing else;
  * **no cleaning** -- official rows as shipped, duplicates included.

That last point is the one most likely to look like an oversight, so to be
explicit: applying MolFusion's conflicting-label rule here would remove 58%
of ``ppbr_az``'s test set, and a score on that is not comparable with any
published number. The duplicate structure is reported as metadata instead.
Track A2 is where the cleaned analysis lives.

Nothing here imports PyTDC. Execution reads frozen files only.
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
    feature_store,
    metrics,
    pipelines,
    protocol,
    release,
    splits,
    tdc,
)
from molfusion_backend.chemistry import canonical_smiles_from_mol, parse_smiles

TRACK = protocol.TRACK_A1


class TrackA1Error(RuntimeError):
    """Execution cannot proceed on the terms the protocol requires."""


# ---------------------------------------------------------------------------
# endpoint loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfficialEndpoint:
    """One endpoint's official rows, canonicalized but not cleaned."""

    name: str
    task_type: str
    tdc_official_metric: str
    molfusion_primary_metric: str
    canonical_smiles: tuple[str, ...]
    labels: tuple[float, ...]
    raw_smiles: tuple[str, ...]
    train_val_rows: int
    test_rows: int
    invalid_rows: dict[int, str] = field(default_factory=dict)

    @property
    def test_indices(self) -> tuple[int, ...]:
        """Test rows are appended after train_val, so the tail is the test set."""
        return tuple(range(self.train_val_rows, self.train_val_rows + self.test_rows))


def load_official_endpoint(name: str, *, frozen_dir: Path, manifest: dict[str, Any]) -> OfficialEndpoint:
    """Load an endpoint's official partitions, verifying them against the manifest."""
    entry = manifest["endpoints"][name]
    task_type = entry["task_type"]

    canonical: list[str] = []
    labels: list[float] = []
    raw: list[str] = []
    invalid: dict[int, str] = {}

    counts = {}
    for part in ("train_val", "test"):
        path = frozen_dir / name / f"{part}.csv"
        actual = release.sha256_file(path)
        expected = entry[part]["sha256"]
        if actual != expected:
            raise TrackA1Error(
                f"{name}/{part}: frozen file checksum {actual} != manifest {expected}; "
                "the benchmark data is not the data this protocol was frozen against"
            )
        _, rows = release.read_frozen_csv(path)
        counts[part] = len(rows)
        for row in rows:
            index = len(canonical)
            raw.append(row[1])
            mol, error = parse_smiles(row[1])
            if mol is None:
                invalid[index] = error or "RDKit could not parse the molecule"
                canonical.append(row[1])
                labels.append(float("nan"))
                continue
            canonical.append(canonical_smiles_from_mol(mol))
            labels.append(float(row[2]))

    return OfficialEndpoint(
        name=name,
        task_type=task_type,
        tdc_official_metric=entry["tdc_official_metric"],
        molfusion_primary_metric=entry["molfusion_primary_metric"],
        canonical_smiles=tuple(canonical),
        labels=tuple(labels),
        raw_smiles=tuple(raw),
        train_val_rows=counts["train_val"],
        test_rows=counts["test"],
        invalid_rows=invalid,
    )


# ---------------------------------------------------------------------------
# official splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfficialSplit:
    """One seed's train/validation realization over a fixed official test set."""

    seed: int
    split_id: str
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def official_splits(endpoint: OfficialEndpoint, *, frozen_dir: Path) -> dict[int, OfficialSplit]:
    """Replay TDC's frozen per-seed train/validation membership as row indices.

    The frozen file records membership as SMILES strings, which is ambiguous
    when a molecule appears twice. It is resolved by consuming occurrences in
    file order: the first k copies go to train, the next m to validation.
    That is exact rather than approximate -- TDC assigns whole scaffold
    groups, and duplicate rows share a scaffold, so every copy of a molecule
    lands in the same partition anyway. The multiset is verified afterwards,
    so a violation of that reasoning would raise instead of pass quietly.
    """
    path = frozen_dir / endpoint.name / "official_seed_splits.json"
    payload = json.loads(path.read_text("utf-8"))

    positions: dict[str, list[int]] = {}
    for index in range(endpoint.train_val_rows):
        positions.setdefault(endpoint.raw_smiles[index], []).append(index)

    result: dict[int, OfficialSplit] = {}
    for seed_text in sorted(payload, key=int):
        seed = int(seed_text)
        cursor = {key: 0 for key in positions}
        assigned: dict[str, list[int]] = {"train_drug": [], "valid_drug": []}
        for part in ("train_drug", "valid_drug"):
            for smiles in payload[seed_text][part]:
                available = positions.get(smiles)
                if available is None or cursor[smiles] >= len(available):
                    raise TrackA1Error(
                        f"{endpoint.name} seed {seed}: frozen split names a molecule "
                        f"that train_val does not contain (enough of): {smiles!r}"
                    )
                assigned[part].append(available[cursor[smiles]])
                cursor[smiles] += 1

        train = tuple(sorted(assigned["train_drug"]))
        validation = tuple(sorted(assigned["valid_drug"]))
        covered = len(train) + len(validation)
        if covered != endpoint.train_val_rows:
            raise TrackA1Error(
                f"{endpoint.name} seed {seed}: official split covers {covered} of "
                f"{endpoint.train_val_rows} train_val rows"
            )
        if set(train) & set(validation):
            raise TrackA1Error(f"{endpoint.name} seed {seed}: train and validation overlap")

        result[seed] = OfficialSplit(
            seed=seed,
            split_id=protocol.split_id(TRACK, seed),
            train=train,
            validation=validation,
            test=endpoint.test_indices,
        )
    return result


def verify_leakage_guards(
    endpoint: OfficialEndpoint, *, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Re-verify the frozen no-leakage claims before any model is fitted.

    Phase 6A.1 already audited these. They are checked again here because an
    audit that only ran once, in another phase, is a claim about the past;
    the run that produces publishable numbers should establish it for itself.
    """
    entry = manifest["endpoints"][endpoint.name]

    # Rows RDKit cannot parse are excluded from every set below. They carry
    # their raw string in place of a canonical one, they cannot belong to a
    # scaffold group, and -- decisively -- the frozen Phase 6A.1 identities
    # were computed over parseable rows only, so including them here would
    # both crash the scaffold call and disagree with the manifest.
    def parseable(indices) -> set[str]:
        return {
            endpoint.canonical_smiles[i]
            for i in indices
            if i not in endpoint.invalid_rows
        }

    train_val = parseable(range(endpoint.train_val_rows))
    test = parseable(range(endpoint.train_val_rows, len(endpoint.canonical_smiles)))

    molecule_overlap = len(train_val & test)
    mf_overlap = len(
        {splits.bemis_murcko_scaffold(s) for s in train_val}
        & {splits.bemis_murcko_scaffold(s) for s in test}
    )
    tdc_overlap = len(
        {tdc.tdc_scaffold(s) for s in train_val} & {tdc.tdc_scaffold(s) for s in test}
    )

    test_identity = release.molecule_set_identity(test)
    frozen_identity = entry["split_identity"]["test_set_sha256"]
    if test_identity != frozen_identity:
        raise TrackA1Error(
            f"{endpoint.name}: official test identity {test_identity} != frozen "
            f"{frozen_identity}; the test partition is not the frozen one"
        )

    if molecule_overlap or mf_overlap or tdc_overlap:
        raise TrackA1Error(
            f"{endpoint.name}: official partitions overlap "
            f"(molecules={molecule_overlap}, scaffolds mf={mf_overlap} tdc={tdc_overlap})"
        )

    return {
        "test_set_sha256": test_identity,
        "test_identity_matches_manifest": True,
        "unparseable_rows_excluded": len(endpoint.invalid_rows),
        "canonical_molecule_overlap": molecule_overlap,
        "scaffold_overlap_molfusion_convention": mf_overlap,
        "scaffold_overlap_tdc_convention": tdc_overlap,
    }


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def artifact_identity_for(representation: str) -> str | None:
    """The verified artifact identity for artifact-backed agents, else None.

    Only ``smiles_tfidf_4096`` is artifact-backed. Its identity is taken from
    the frozen TF-IDF contract plus the artifact's own checksum-verified
    corpus identity, so replacing the artifact with a different fitted
    payload invalidates every cached matrix that depended on it.
    """
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


def features_for(
    endpoint: OfficialEndpoint,
    representation: str,
    *,
    store: feature_store.FeatureStore,
    release_identity: str,
    artifact_identity: str | None,
) -> FeatureResult:
    """Compute or load one endpoint x representation matrix."""
    agent = agent_registry.get(representation)
    if agent.output_structure != "vector":
        raise TrackA1Error(
            f"{representation!r} is a {agent.output_structure!r} agent; Track A "
            "covers fixed-length vectors only"
        )

    rows = feature_store.row_identity(endpoint.canonical_smiles)
    key = feature_store.matrix_cache_key(
        release_identity=release_identity,
        endpoint=endpoint.name,
        agent_id=agent.id,
        agent_version=agent.version,
        output_dim=int(agent.output_dim or 0),
        normalization_id=protocol.CANONICALIZATION_ID,
        row_identity_sha256=rows,
        artifact_identity=artifact_identity,
    )
    expect = {
        "cache_schema_version": feature_store.CACHE_SCHEMA_VERSION,
        "release_identity": release_identity,
        "endpoint": endpoint.name,
        "agent_id": agent.id,
        "agent_version": agent.version,
        "output_dim": int(agent.output_dim or 0),
        "row_identity_sha256": rows,
        "artifact_identity": artifact_identity,
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
        )

    started = time.perf_counter()
    vectors: list[np.ndarray] = []
    succeeded: list[int] = []
    failures: dict[int, str] = dict(endpoint.invalid_rows)
    for index, smiles in enumerate(endpoint.canonical_smiles):
        if index in failures:
            continue
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
    store.store(
        key,
        matrix=matrix,
        succeeded=succeeded,
        failures=failures,
        metadata={
            **expect,
            "representation": representation,
            "value_type": agent.value_type,
            "feature_seconds": seconds,
            "nonzero_fraction": (
                float(np.count_nonzero(matrix) / matrix.size) if matrix.size else None
            ),
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
    )


# ---------------------------------------------------------------------------
# one cell: endpoint x representation x probe, over all five seeds
# ---------------------------------------------------------------------------


def _score(model, x, y, task_type: str) -> dict[str, float]:
    if task_type == protocol.TASK_CLASSIFICATION:
        return metrics.classification_metrics(y, model.predict_proba(x)[:, 1], model.predict(x))
    return metrics.regression_metrics(y, model.predict(x))


def run_cell(
    *,
    endpoint: OfficialEndpoint,
    representation: str,
    probe: str,
    splits_by_seed: dict[int, OfficialSplit],
    features: FeatureResult,
    release_identity: str,
) -> dict[str, Any]:
    """Execute one benchmark cell and return its result shard payload.

    The tune-then-test ordering is straight-line code rather than a
    convention a caller has to honour: every candidate is scored on
    validation, the winner is refit, and only then is the test partition
    touched -- once.
    """
    task_type = endpoint.task_type
    primary = endpoint.molfusion_primary_metric
    grid = pipelines.hyperparameter_grid(probe, task_type)

    row_of = {index: position for position, index in enumerate(features.succeeded)}
    labels = np.asarray(endpoint.labels, dtype=np.float64)

    def block(indices: Sequence[int]):
        rows = [row_of[i] for i in indices if i in row_of]
        x = features.matrix[rows]
        y = labels[[i for i in indices if i in row_of]]
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
            raise TrackA1Error(
                f"{endpoint.name}/{representation}/{probe} seed {seed}: "
                f"empty train ({n_train}) or test ({n_test}) partition"
            )

        # --- model selection: validation only --------------------------
        best_params: dict[str, Any] = {}
        best_score = -np.inf
        selection_seconds = 0.0
        for candidate in grid:
            model = pipelines.build_pipeline(
                representation=representation,
                probe=probe,
                task_type=task_type,
                hyperparameters=candidate,
            )
            started = time.perf_counter()
            model.fit(x_train, y_train)
            selection_seconds += time.perf_counter() - started
            try:
                scored = _score(model, x_valid, y_valid, task_type)[primary]
            except ValueError:
                # e.g. a validation fold with a single class: the candidate
                # cannot be scored, so it is skipped rather than ranked on a
                # number that does not exist.
                continue
            oriented = metrics.orient(primary, scored)
            if oriented > best_score:
                best_score, best_params = oriented, candidate

        # --- final fit, then the single look at test --------------------
        model = pipelines.build_pipeline(
            representation=representation,
            probe=probe,
            task_type=task_type,
            hyperparameters=best_params,
        )
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        _score(model, x_valid, y_valid, task_type) if n_valid else None
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
                    "split_strategy": protocol.TRACK_A1_SPLIT_STRATEGY,
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
        "seeds": sorted(splits_by_seed),
    }


__all__ = [
    "TRACK",
    "FeatureResult",
    "OfficialEndpoint",
    "OfficialSplit",
    "TrackA1Error",
    "artifact_identity_for",
    "features_for",
    "load_official_endpoint",
    "official_splits",
    "run_cell",
    "verify_leakage_guards",
]
