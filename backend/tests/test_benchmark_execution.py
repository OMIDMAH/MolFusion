"""Phase 6A.2: the Track A1 execution contract.

The full run is roughly a day of compute, so none of these tests execute it.
What they cover is everything that determines whether that run can be
trusted and resumed: cache identity and invalidation, atomic writes, shard
validation, resume without duplication, the tune-then-test separation, and
the matrix arithmetic that says whether a run is complete.

A test suite that needed the benchmark to run would never be run.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from molfusion_backend.benchmark import (
    a1,
    feature_store,
    metrics,
    pipelines,
    protocol,
    runner,
)

RELEASE = "a" * 64
OTHER_RELEASE = "b" * 64

BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
ETHANOL = "CCO"
PYRIDINE = "c1ccncc1"


def _key(**overrides):
    base = dict(
        release_identity=RELEASE,
        endpoint="demo",
        agent_id="morgan_ecfp4_1024",
        agent_version="1.0.0",
        output_dim=1024,
        normalization_id=protocol.CANONICALIZATION_ID,
        row_identity_sha256=feature_store.row_identity([BENZENE, ETHANOL]),
        artifact_identity=None,
    )
    base.update(overrides)
    return feature_store.matrix_cache_key(**base)


# --------------------------------------------------------------------------
# cache key
# --------------------------------------------------------------------------


def test_cache_key_is_deterministic():
    assert _key() == _key()


def test_cache_key_changes_with_release_identity():
    """A new frozen release must not reuse the previous release's matrices."""
    assert _key() != _key(release_identity=OTHER_RELEASE)


def test_cache_key_changes_with_agent_version():
    """The code that computed the vectors is part of what they are."""
    assert _key() != _key(agent_version="1.0.1")


def test_cache_key_changes_with_artifact_identity():
    """A refitted TF-IDF artifact produces different vectors from same input."""
    assert _key() != _key(artifact_identity="tfidf\x1fchembl37\x1f2.0.0\x1fabc")
    assert _key(artifact_identity="x") != _key(artifact_identity="y")


def test_cache_key_changes_with_output_dimension():
    """A changed width is a changed representation, even at the same version."""
    assert _key() != _key(output_dim=2048)


def test_cache_key_changes_with_endpoint_and_agent():
    assert _key() != _key(endpoint="other")
    assert _key() != _key(agent_id="maccs_keys_167")


def test_cache_key_changes_with_row_identity():
    rows = feature_store.row_identity([BENZENE, ETHANOL, TOLUENE])
    assert _key() != _key(row_identity_sha256=rows)


def test_cache_key_is_not_keyed_on_split_seed_or_probe():
    """Features do not depend on the model that will consume them."""
    contract = feature_store.cache_contract()
    for excluded in ("split", "seed", "probe", "hyperparameters", "labels", "filename"):
        assert excluded in contract["not_keyed_on"]


# --------------------------------------------------------------------------
# row identity
# --------------------------------------------------------------------------


def test_row_identity_is_order_sensitive():
    """The claim being protected is "matrix row i is molecule i"."""
    assert feature_store.row_identity([BENZENE, ETHANOL]) != \
        feature_store.row_identity([ETHANOL, BENZENE])


def test_row_identity_preserves_duplicates():
    """Track A1 consumes official rows as shipped, duplicates included."""
    assert feature_store.row_identity([BENZENE, BENZENE]) != \
        feature_store.row_identity([BENZENE])


def test_row_identity_detects_truncation():
    assert feature_store.row_identity([BENZENE, ETHANOL, TOLUENE]) != \
        feature_store.row_identity([BENZENE, ETHANOL])


# --------------------------------------------------------------------------
# store: round trip, validation, atomicity
# --------------------------------------------------------------------------


def _store_entry(store, key, rows=3, dim=4, **meta):
    matrix = np.arange(rows * dim, dtype=np.float64).reshape(rows, dim)
    metadata = {
        "cache_schema_version": feature_store.CACHE_SCHEMA_VERSION,
        "release_identity": RELEASE,
        "endpoint": "demo",
        "agent_id": "morgan_ecfp4_1024",
        "agent_version": "1.0.0",
        "output_dim": dim,
        "row_identity_sha256": "rows",
        "artifact_identity": None,
    }
    metadata.update(meta)
    store.store(key, matrix=matrix, succeeded=list(range(rows)), failures={}, metadata=metadata)
    return metadata


def test_stored_matrix_round_trips(tmp_path):
    store = feature_store.FeatureStore(tmp_path)
    expect = _store_entry(store, "k1")
    cached = store.load("k1", expect=expect)
    assert cached is not None
    assert cached.matrix.shape == (3, 4)
    assert cached.matrix.dtype == feature_store.MATRIX_DTYPE
    assert cached.succeeded == (0, 1, 2)


def test_a_missing_entry_is_a_cold_cache_not_an_error(tmp_path):
    store = feature_store.FeatureStore(tmp_path)
    assert store.load("absent", expect={"endpoint": "demo"}) is None


def test_a_present_entry_that_disagrees_is_an_error(tmp_path):
    """A stale entry is evidence of a problem; recomputing over it hides one."""
    store = feature_store.FeatureStore(tmp_path)
    expect = _store_entry(store, "k1")
    with pytest.raises(feature_store.FeatureCacheError, match="stale"):
        store.load("k1", expect={**expect, "agent_version": "9.9.9"})


def test_row_identity_mismatch_is_detected_on_load(tmp_path):
    store = feature_store.FeatureStore(tmp_path)
    expect = _store_entry(store, "k1")
    with pytest.raises(feature_store.FeatureCacheError, match="row_identity_sha256"):
        store.load("k1", expect={**expect, "row_identity_sha256": "different"})


def test_release_identity_mismatch_is_detected_on_load(tmp_path):
    store = feature_store.FeatureStore(tmp_path)
    expect = _store_entry(store, "k1")
    with pytest.raises(feature_store.FeatureCacheError, match="release_identity"):
        store.load("k1", expect={**expect, "release_identity": OTHER_RELEASE})


def test_a_truncated_matrix_file_is_rejected(tmp_path):
    store = feature_store.FeatureStore(tmp_path)
    expect = _store_entry(store, "k1")
    (store.entry_dir("k1") / "matrix.npy").write_bytes(b"not a numpy file")
    with pytest.raises(Exception):
        store.load("k1", expect=expect)


def test_a_shape_disagreement_is_rejected(tmp_path):
    store = feature_store.FeatureStore(tmp_path)
    expect = _store_entry(store, "k1")
    meta_path = store.entry_dir("k1") / "metadata.json"
    payload = json.loads(meta_path.read_text("utf-8"))
    payload["matrix_shape"] = [99, 4]
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(feature_store.FeatureCacheError, match="shape"):
        store.load("k1", expect=expect)


def test_row_index_count_must_match_matrix_rows(tmp_path):
    store = feature_store.FeatureStore(tmp_path)
    with pytest.raises(feature_store.FeatureCacheError, match="row indices"):
        store.store(
            "k1", matrix=np.zeros((3, 4)), succeeded=[0, 1],
            failures={}, metadata={"endpoint": "demo"},
        )


def test_a_failed_write_leaves_no_entry_behind(tmp_path):
    """Atomicity: a killed writer leaves nothing, never a half-valid entry."""
    store = feature_store.FeatureStore(tmp_path)
    with pytest.raises(feature_store.FeatureCacheError):
        store.store("k1", matrix=np.zeros((3, 4)), succeeded=[0], failures={}, metadata={})
    assert not store.entry_dir("k1").exists()
    assert not list(tmp_path.glob("**/.staging-*"))


def test_store_is_finalized_by_rename_not_in_place_writes(tmp_path):
    """The entry appears complete or not at all."""
    store = feature_store.FeatureStore(tmp_path)
    _store_entry(store, "k1")
    entry = store.entry_dir("k1")
    assert (entry / "matrix.npy").exists()
    assert (entry / "metadata.json").exists()
    assert not list(tmp_path.glob("**/*.partial"))


def test_rewriting_an_existing_entry_succeeds(tmp_path):
    store = feature_store.FeatureStore(tmp_path)
    _store_entry(store, "k1")
    expect = _store_entry(store, "k1")
    assert store.load("k1", expect=expect) is not None


# --------------------------------------------------------------------------
# experiment matrix
# --------------------------------------------------------------------------


@pytest.fixture
def manifest():
    return {
        "release_identity_sha256": RELEASE,
        "release_name": "TEST",
        "endpoints": {
            "clf_endpoint": {
                "included": True, "task_type": protocol.TASK_CLASSIFICATION,
                "ingestion": {"rdkit_invalid": 0},
            },
            "reg_endpoint": {
                "included": True, "task_type": protocol.TASK_REGRESSION,
                "ingestion": {"rdkit_invalid": 0},
            },
        },
    }


def test_experiment_matrix_is_endpoints_times_representations_times_probes(manifest):
    cells = runner.experiment_matrix(manifest)
    assert len(cells) == 2 * len(protocol.TRACK_A_REPRESENTATIONS) * len(protocol.PROBES)
    assert len({c[1] for c in cells}) == 7


def test_experiment_matrix_excludes_excluded_endpoints(manifest):
    manifest["endpoints"]["reg_endpoint"]["included"] = False
    assert {c[0] for c in runner.experiment_matrix(manifest)} == {"clf_endpoint"}


def test_expected_counts_match_the_frozen_protocol(manifest):
    counts = runner.expected_counts(manifest)
    cells = 2 * 7 * 2
    assert counts["cells"] == cells
    assert counts["seeds"] == 5
    assert counts["hyperparameter_candidates"] == 4
    assert counts["selection_fits"] == cells * 5 * 4
    assert counts["final_fits"] == cells * 5
    assert counts["total_fits"] == cells * 5 * 5
    assert counts["test_evaluations"] == cells * 5
    assert counts["result_rows"] == cells * 5 * 4


def test_the_matrix_covers_track_a1_seeds_not_track_a2_seeds():
    assert protocol.TRACK_A1_SEEDS == (1, 2, 3, 4, 5)
    assert set(protocol.TRACK_A1_SEEDS) != set(protocol.TRACK_A2_SEEDS)


# --------------------------------------------------------------------------
# shards, resume, duplication
# --------------------------------------------------------------------------


def _shard(endpoint="clf_endpoint", representation="maccs_keys_167", probe="linear",
           seeds=(1, 2, 3, 4, 5), rows=20, **overrides):
    payload = {
        "shard_schema_version": runner.SHARD_SCHEMA_VERSION,
        "status": "complete",
        "benchmark_release": RELEASE,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "seeds": list(seeds),
        "cell": {"track": a1.TRACK, "endpoint": endpoint,
                 "representation": representation, "probe": probe},
        "rows": [
            {"endpoint": endpoint, "representation": representation, "probe": probe,
             "seed": seed, "metric": metric, "metric_value": 0.5}
            for seed in seeds
            for metric in ("auroc", "auprc", "balanced_accuracy", "mcc")
        ][:rows],
        "timings": [],
        "test_set_sha256": "t" * 64,
        "feature_failures": 0,
    }
    payload["cell_identity"] = runner.cell_identity(
        release_identity=RELEASE, endpoint=endpoint,
        representation=representation, probe=probe, seeds=seeds,
    )
    payload.update(overrides)
    return payload


def _write(tmp_path, payload):
    path = runner.shard_path(
        tmp_path, payload["cell"]["endpoint"],
        payload["cell"]["representation"], payload["cell"]["probe"],
    )
    runner.write_shard(path, payload)
    return path


def test_a_valid_shard_is_reused(tmp_path):
    path = _write(tmp_path, _shard())
    assert runner.read_valid_shard(
        path, release_identity=RELEASE, expected_seeds=(1, 2, 3, 4, 5), expected_rows=20
    ) is not None


def test_an_incomplete_shard_is_rejected(tmp_path):
    """status != complete means the writer never finished."""
    path = _write(tmp_path, _shard(status="running"))
    assert runner.read_valid_shard(
        path, release_identity=RELEASE, expected_seeds=(1, 2, 3, 4, 5), expected_rows=20
    ) is None


def test_a_truncated_shard_is_rejected(tmp_path):
    path = _write(tmp_path, _shard())
    path.write_text(path.read_text("utf-8")[:200], encoding="utf-8")
    assert runner.read_valid_shard(
        path, release_identity=RELEASE, expected_seeds=(1, 2, 3, 4, 5), expected_rows=20
    ) is None


def test_a_shard_from_a_different_release_is_rejected(tmp_path):
    path = _write(tmp_path, _shard(benchmark_release=OTHER_RELEASE))
    assert runner.read_valid_shard(
        path, release_identity=RELEASE, expected_seeds=(1, 2, 3, 4, 5), expected_rows=20
    ) is None


def test_a_shard_with_missing_seeds_is_rejected(tmp_path):
    path = _write(tmp_path, _shard(seeds=(1, 2, 3)))
    assert runner.read_valid_shard(
        path, release_identity=RELEASE, expected_seeds=(1, 2, 3, 4, 5), expected_rows=20
    ) is None


def test_a_shard_with_the_wrong_row_count_is_rejected(tmp_path):
    path = _write(tmp_path, _shard(rows=12))
    assert runner.read_valid_shard(
        path, release_identity=RELEASE, expected_seeds=(1, 2, 3, 4, 5), expected_rows=20
    ) is None


def test_a_tampered_cell_identity_is_rejected(tmp_path):
    path = _write(tmp_path, _shard(cell_identity="0" * 64))
    assert runner.read_valid_shard(
        path, release_identity=RELEASE, expected_seeds=(1, 2, 3, 4, 5), expected_rows=20
    ) is None


def test_shard_writes_leave_no_partial_files(tmp_path):
    _write(tmp_path, _shard())
    assert not list(tmp_path.glob("**/*.partial"))


def test_cell_identity_excludes_volatile_metadata():
    """Two runs of the same cell, on different days, share an identity."""
    a = runner.cell_identity(release_identity=RELEASE, endpoint="e",
                             representation="r", probe="linear", seeds=(1, 2))
    b = runner.cell_identity(release_identity=RELEASE, endpoint="e",
                             representation="r", probe="linear", seeds=(2, 1))
    assert a == b


def test_cell_identity_changes_with_scientific_configuration():
    base = dict(release_identity=RELEASE, endpoint="e", representation="r",
                probe="linear", seeds=(1, 2))
    assert runner.cell_identity(**base) != runner.cell_identity(**{**base, "probe": "nonlinear"})
    assert runner.cell_identity(**base) != runner.cell_identity(**{**base, "endpoint": "f"})
    assert runner.cell_identity(**base) != runner.cell_identity(
        **{**base, "release_identity": OTHER_RELEASE}
    )


def test_plan_reruns_only_the_invalid_cells(tmp_path, manifest):
    """Resume: valid cells are skipped, damaged ones come back as work."""
    _write(tmp_path, _shard(endpoint="clf_endpoint", representation="maccs_keys_167",
                            probe="linear"))
    _write(tmp_path, _shard(endpoint="clf_endpoint", representation="maccs_keys_167",
                            probe="nonlinear", status="running"))
    jobs, done = runner.plan(manifest, tmp_path)
    assert ("clf_endpoint", "maccs_keys_167", "linear") in done
    units = {(j["endpoint"], j["representation"]) for j in jobs}
    assert ("clf_endpoint", "maccs_keys_167") in units


def test_collect_produces_no_duplicate_rows_from_valid_shards(tmp_path, manifest):
    _write(tmp_path, _shard())
    rows, collected = runner.collect(tmp_path, manifest)
    assert collected["audit"]["duplicate_rows"] == 0
    assert len(rows) == 20


def test_collect_reports_missing_cells_rather_than_hiding_them(tmp_path, manifest):
    _write(tmp_path, _shard())
    _rows, collected = runner.collect(tmp_path, manifest)
    audit = collected["audit"]
    assert audit["complete"] is False
    assert len(audit["missing_cells"]) == audit["expected"]["cells"] - 1


def test_collect_flags_nan_metric_values(tmp_path, manifest):
    payload = _shard()
    payload["rows"][0]["metric_value"] = float("nan")
    _write(tmp_path, payload)
    _rows, collected = runner.collect(tmp_path, manifest)
    assert len(collected["audit"]["nan_or_inf_rows"]) == 1


def test_collect_flags_a_test_identity_that_changed_within_an_endpoint(tmp_path, manifest):
    """A1's test set is fixed; a differing hash means something moved."""
    _write(tmp_path, _shard(probe="linear"))
    _write(tmp_path, _shard(probe="nonlinear", test_set_sha256="z" * 64))
    _rows, collected = runner.collect(tmp_path, manifest)
    audit = collected["audit"]
    assert audit["test_identity_stable_within_endpoint"] is False
    assert "clf_endpoint" in audit["endpoints_with_unstable_test_identity"]


# --------------------------------------------------------------------------
# result schema and identity
# --------------------------------------------------------------------------


def test_result_columns_cover_everything_the_protocol_requires():
    required = {
        "benchmark_release", "track", "endpoint", "task_type", "split_id", "seed",
        "representation", "agent_version", "model_family", "hyperparameters",
        "metric", "metric_value", "n_train", "n_valid", "n_test",
        "feature_failures", "feature_seconds", "fit_seconds", "predict_seconds",
    }
    assert required <= set(runner.RESULT_COLUMNS)


def test_results_file_is_written_in_the_declared_column_order(tmp_path):
    rows = [{c: 1 for c in runner.RESULT_COLUMNS}]
    rows[0].update({"endpoint": "e", "representation": "r", "probe": "linear",
                    "seed": 1, "metric": "auroc"})
    path = tmp_path / "results.csv"
    runner.write_results(path, rows)
    header = path.read_text("utf-8").splitlines()[0]
    assert header == ",".join(runner.RESULT_COLUMNS)


def test_scientific_identity_ignores_timings(tmp_path):
    """Otherwise two identical benchmarks would disagree."""
    base = {c: 0 for c in runner.RESULT_COLUMNS}
    base.update({"endpoint": "e", "representation": "r", "probe": "linear",
                 "seed": 1, "metric": "auroc", "metric_value": 0.75})
    slow = {**base, "fit_seconds": 999.0, "predict_seconds": 5.0,
            "feature_cache_hit": True, "selection_seconds": 12.0}
    assert runner.scientific_identity([base]) == runner.scientific_identity([slow])


def test_scientific_identity_changes_with_a_metric_value(tmp_path):
    base = {c: 0 for c in runner.RESULT_COLUMNS}
    base.update({"endpoint": "e", "representation": "r", "probe": "linear",
                 "seed": 1, "metric": "auroc", "metric_value": 0.75})
    changed = {**base, "metric_value": 0.7500000000000001}
    assert runner.scientific_identity([base]) != runner.scientific_identity([changed])


def test_scientific_identity_ignores_row_order():
    def row(seed):
        base = {c: 0 for c in runner.RESULT_COLUMNS}
        base.update({"endpoint": "e", "representation": "r", "probe": "linear",
                     "seed": seed, "metric": "auroc", "metric_value": 0.5})
        return base

    assert runner.scientific_identity([row(1), row(2)]) == \
        runner.scientific_identity([row(2), row(1)])


# --------------------------------------------------------------------------
# leakage discipline
# --------------------------------------------------------------------------


def test_linear_pipelines_fit_their_scaler_on_training_data_only(tmp_path):
    """The transform applied to test must come from train, not from test."""
    model = pipelines.build_pipeline(
        representation="rdkit_physchem_descriptors", probe="linear",
        task_type=protocol.TASK_CLASSIFICATION, hyperparameters={"C": 1.0},
    )
    train = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    y = np.array([0, 0, 1, 1])
    model.fit(train, y)
    scaler = model.named_steps.get("scale")
    assert scaler is not None
    before = scaler.mean_.copy()
    model.predict(np.array([[1000.0, 2000.0], [3000.0, 4000.0]]))
    assert np.array_equal(scaler.mean_, before), "predict must not refit the scaler"
    assert before == pytest.approx(train.mean(axis=0))


def test_binary_representations_get_no_scaler():
    """Standardizing a bit vector gives an absent bit a nonzero value."""
    model = pipelines.build_pipeline(
        representation="morgan_ecfp4_1024", probe="linear",
        task_type=protocol.TASK_CLASSIFICATION, hyperparameters={"C": 1.0},
    )
    assert "scale" not in model.named_steps


def test_tree_probes_get_no_fitted_preprocessing():
    model = pipelines.build_pipeline(
        representation="rdkit_physchem_descriptors", probe="nonlinear",
        task_type=protocol.TASK_CLASSIFICATION, hyperparameters={},
    )
    assert "scale" not in model.named_steps


def test_hyperparameter_grid_is_identical_across_representations():
    """An unequal budget cannot be introduced by accident: no rep argument."""
    import inspect

    signature = inspect.signature(pipelines.hyperparameter_grid)
    assert "representation" not in signature.parameters


def test_official_split_never_places_a_test_row_in_train_or_validation():
    """The A1 test partition is the tail of the row list and is disjoint."""
    endpoint = a1.OfficialEndpoint(
        name="demo", task_type=protocol.TASK_CLASSIFICATION,
        tdc_official_metric="roc-auc", molfusion_primary_metric="auroc",
        canonical_smiles=(BENZENE, TOLUENE, ETHANOL, PYRIDINE),
        labels=(1.0, 0.0, 1.0, 0.0),
        raw_smiles=(BENZENE, TOLUENE, ETHANOL, PYRIDINE),
        train_val_rows=2, test_rows=2,
    )
    assert endpoint.test_indices == (2, 3)


def test_official_splits_cover_train_val_exactly(tmp_path):
    endpoint = a1.OfficialEndpoint(
        name="demo", task_type=protocol.TASK_CLASSIFICATION,
        tdc_official_metric="roc-auc", molfusion_primary_metric="auroc",
        canonical_smiles=(BENZENE, TOLUENE, ETHANOL, PYRIDINE),
        labels=(1.0, 0.0, 1.0, 0.0),
        raw_smiles=(BENZENE, TOLUENE, ETHANOL, PYRIDINE),
        train_val_rows=2, test_rows=2,
    )
    directory = tmp_path / "demo"
    directory.mkdir(parents=True)
    (directory / "official_seed_splits.json").write_text(
        json.dumps({"1": {"train_drug": [BENZENE], "valid_drug": [TOLUENE]},
                    "2": {"train_drug": [TOLUENE], "valid_drug": [BENZENE]}}),
        encoding="utf-8",
    )
    splits = a1.official_splits(endpoint, frozen_dir=tmp_path)
    assert set(splits) == {1, 2}
    for split in splits.values():
        assert set(split.train) & set(split.validation) == set()
        assert split.test == (2, 3)
        assert split.split_id.startswith(protocol.TRACK_A1)


def test_the_a1_test_partition_is_identical_across_seeds(tmp_path):
    """Five train/validation realizations, one fixed external test set."""
    endpoint = a1.OfficialEndpoint(
        name="demo", task_type=protocol.TASK_CLASSIFICATION,
        tdc_official_metric="roc-auc", molfusion_primary_metric="auroc",
        canonical_smiles=(BENZENE, TOLUENE, ETHANOL, PYRIDINE),
        labels=(1.0, 0.0, 1.0, 0.0),
        raw_smiles=(BENZENE, TOLUENE, ETHANOL, PYRIDINE),
        train_val_rows=2, test_rows=2,
    )
    directory = tmp_path / "demo"
    directory.mkdir(parents=True)
    (directory / "official_seed_splits.json").write_text(
        json.dumps({str(s): {"train_drug": [BENZENE], "valid_drug": [TOLUENE]}
                    for s in protocol.TRACK_A1_SEEDS}),
        encoding="utf-8",
    )
    splits = a1.official_splits(endpoint, frozen_dir=tmp_path)
    assert len({s.test for s in splits.values()}) == 1


def test_an_official_split_that_misses_train_val_rows_is_refused(tmp_path):
    endpoint = a1.OfficialEndpoint(
        name="demo", task_type=protocol.TASK_CLASSIFICATION,
        tdc_official_metric="roc-auc", molfusion_primary_metric="auroc",
        canonical_smiles=(BENZENE, TOLUENE, ETHANOL),
        labels=(1.0, 0.0, 1.0), raw_smiles=(BENZENE, TOLUENE, ETHANOL),
        train_val_rows=2, test_rows=1,
    )
    directory = tmp_path / "demo"
    directory.mkdir(parents=True)
    (directory / "official_seed_splits.json").write_text(
        json.dumps({"1": {"train_drug": [BENZENE], "valid_drug": []}}), encoding="utf-8"
    )
    with pytest.raises(a1.TrackA1Error, match="covers 1 of 2"):
        a1.official_splits(endpoint, frozen_dir=tmp_path)


def test_track_a1_applies_no_cleaning():
    """Cleaning A1 would score it on a set no published number used."""
    assert protocol.TRACK_A1_CLEANING.startswith("none")


def test_execution_module_does_not_import_pytdc():
    """Verification must not require the tool that produced the data."""
    for module in (a1, runner, feature_store):
        source = Path(module.__file__).read_text("utf-8")
        assert "import tdc" not in source
        assert "from tdc" not in source


def test_leakage_guards_skip_rows_rdkit_cannot_parse(tmp_path):
    """Regression: unparseable official rows must not reach the scaffold call.

    solubility_aqsoldb ships two SMILES RDKit rejects. An earlier version of
    the guard passed their raw strings to the scaffold function, which
    raises, failing all seven of that endpoint's cells. They are excluded
    now -- which is also what the frozen Phase 6A.1 identities did, so
    including them would have disagreed with the manifest as well.
    """
    from molfusion_backend.benchmark import release as release_module

    broken = "O=C(O)C1=C[NH+2]([O-])[CH-]C=C1"
    endpoint = a1.OfficialEndpoint(
        name="demo", task_type=protocol.TASK_REGRESSION,
        tdc_official_metric="mae", molfusion_primary_metric="mae",
        canonical_smiles=(BENZENE, broken, ETHANOL, PYRIDINE),
        labels=(1.0, float("nan"), 2.0, 3.0),
        raw_smiles=(BENZENE, broken, ETHANOL, PYRIDINE),
        train_val_rows=2, test_rows=2,
        invalid_rows={1: "RDKit could not parse the molecule"},
    )
    manifest = {
        "endpoints": {
            "demo": {
                "split_identity": {
                    "test_set_sha256": release_module.molecule_set_identity(
                        [ETHANOL, PYRIDINE]
                    )
                }
            }
        }
    }
    guards = a1.verify_leakage_guards(endpoint, manifest=manifest)
    assert guards["unparseable_rows_excluded"] == 1
    assert guards["canonical_molecule_overlap"] == 0
    assert guards["test_identity_matches_manifest"] is True


def test_leakage_guards_still_reject_a_genuine_test_identity_mismatch():
    endpoint = a1.OfficialEndpoint(
        name="demo", task_type=protocol.TASK_REGRESSION,
        tdc_official_metric="mae", molfusion_primary_metric="mae",
        canonical_smiles=(BENZENE, ETHANOL),
        labels=(1.0, 2.0), raw_smiles=(BENZENE, ETHANOL),
        train_val_rows=1, test_rows=1,
    )
    manifest = {"endpoints": {"demo": {"split_identity": {"test_set_sha256": "0" * 64}}}}
    with pytest.raises(a1.TrackA1Error, match="test identity"):
        a1.verify_leakage_guards(endpoint, manifest=manifest)


# --------------------------------------------------------------------------
# Phase 6A.2 amendment: non-finite descriptor values
# --------------------------------------------------------------------------


def test_non_finite_fold_is_the_identity_on_finite_data():
    """Why the amendment invalidates no already-computed result.

    Only one matrix in the whole benchmark contained inf. If the fold is the
    identity everywhere else, adding it cannot have changed any number that
    was already computed without it.
    """
    finite = np.array([[1.0, -2.5, 0.0], [1e300, 3.25, -7.0]])
    folded = pipelines._non_finite_to_nan(finite)
    assert np.array_equal(folded, finite)


def test_non_finite_fold_maps_both_infinities_to_nan():
    folded = pipelines._non_finite_to_nan(np.array([[np.inf, -np.inf, 1.0]]))
    assert np.isnan(folded[0, 0]) and np.isnan(folded[0, 1])
    assert folded[0, 2] == 1.0


def test_non_finite_fold_leaves_existing_nan_alone():
    folded = pipelines._non_finite_to_nan(np.array([[np.nan, 2.0]]))
    assert np.isnan(folded[0, 0]) and folded[0, 1] == 2.0


def test_non_finite_fold_is_stateless_and_cannot_leak():
    """It fits nothing, so it cannot carry information from test to train."""
    step = dict(pipelines._finite_step() for _ in [0])["finite"]
    before = step.get_params()
    step.fit(np.array([[np.inf, 1.0]]))
    assert step.get_params() == before


def test_both_probes_survive_an_infinite_descriptor_value():
    """The failure that stopped solubility_aqsoldb, as a test."""
    x = np.array([[1.0, np.inf], [2.0, 3.0], [4.0, np.nan], [6.0, 7.0]])
    y = np.array([0, 0, 1, 1])
    for probe, params in (("linear", {"C": 1.0}), ("nonlinear", {})):
        model = pipelines.build_pipeline(
            representation="rdkit_physchem_descriptors", probe=probe,
            task_type=protocol.TASK_CLASSIFICATION, hyperparameters=params,
        )
        model.fit(x, y)
        assert model.predict(x).shape == (4,)


def test_the_fold_is_applied_to_every_representation_not_just_descriptors():
    """Uniform, so no representation gets special treatment."""
    for representation in protocol.TRACK_A_REPRESENTATIONS:
        for probe in protocol.PROBES:
            model = pipelines.build_pipeline(
                representation=representation, probe=probe,
                task_type=protocol.TASK_CLASSIFICATION, hyperparameters={},
            )
            assert "finite" in model.named_steps
