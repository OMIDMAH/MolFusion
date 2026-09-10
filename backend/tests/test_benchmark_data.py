"""Phase 6A.1: frozen benchmark-data contract, and the official-split audit.

None of these tests download anything. The TDC acquisition is a one-off
operation whose product is a frozen directory plus a manifest; what has to
keep working is the *contract* around that product -- deterministic
serialization, honest checksums, set identity, and the audit that
distinguishes TDC's official split from MolFusion's own. All of that is
testable on small fixtures, and a test suite that needed the network would
simply stop running.
"""

import csv
import json
from pathlib import Path

import pytest

from molfusion_backend.benchmark import datasets, protocol, release, splits, tdc

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

# Real drug-like SMILES with the properties the audit has to notice: two
# spellings of one molecule, a stereocentre, an acyclic compound, and an
# unparseable string.
BENZENE = "c1ccccc1"
TOLUENE_A = "Cc1ccccc1"
TOLUENE_B = "c1ccccc1C"          # same molecule, different spelling
ALANINE = "C[C@@H](N)C(=O)O"     # stereocentre; acyclic
ETHANOL = "CCO"                  # acyclic
PYRIDINE = "c1ccncc1"
NAPHTHALENE = "c1ccc2ccccc2c1"
BROKEN = "not-a-molecule"


def _write_frozen(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(release.frozen_csv_bytes(rows))


@pytest.fixture
def frozen_endpoint(tmp_path):
    """A miniature frozen endpoint with a fixed test set and seed splits."""
    root = tmp_path / "frozen"
    endpoint = root / "demo_endpoint"

    train_val = [
        ("D1", BENZENE, 1.0),
        ("D2", TOLUENE_A, 0.0),
        ("D3", PYRIDINE, 1.0),
        ("D4", ETHANOL, 0.0),
    ]
    test = [
        ("D5", NAPHTHALENE, 1.0),
        ("D6", ALANINE, 0.0),
    ]
    _write_frozen(endpoint / "train_val.csv", train_val)
    _write_frozen(endpoint / "test.csv", test)

    # Two seeds that partition train_val differently but share the test file.
    seed_splits = {
        "1": {"train_drug": [BENZENE, TOLUENE_A, PYRIDINE], "valid_drug": [ETHANOL]},
        "2": {"train_drug": [BENZENE, ETHANOL, PYRIDINE], "valid_drug": [TOLUENE_A]},
    }
    (endpoint / "official_seed_splits.json").write_text(
        json.dumps(seed_splits, sort_keys=True), encoding="utf-8"
    )

    metadata = {
        "tdc_dataset_name": "demo_endpoint",
        "tdc_category": "ADME",
        "tdc_official_metric": "roc-auc",
        "tdc_split_method": "scaffold",
    }
    return root, metadata


# --------------------------------------------------------------------------
# serialization contract
# --------------------------------------------------------------------------


def test_frozen_serialization_uses_lf_and_has_no_bom():
    """On Windows the csv module would emit CRLF unless told otherwise."""
    data = release.frozen_csv_bytes([("D1", BENZENE, 1.0)])
    assert b"\r\n" not in data
    assert not data.startswith(b"\xef\xbb\xbf")
    assert data.endswith(b"\n")


def test_frozen_serialization_pins_the_column_order():
    data = release.frozen_csv_bytes([("D1", BENZENE, 1.0)]).decode("utf-8")
    assert data.splitlines()[0] == "Drug_ID,Drug,Y"
    assert list(release.FIELDS) == ["Drug_ID", "Drug", "Y"]


def test_frozen_serialization_is_byte_identical_across_repeats():
    rows = [("D1", BENZENE, 1.0), ("D2", TOLUENE_A, 0.5)]
    assert release.frozen_csv_bytes(rows) == release.frozen_csv_bytes(rows)


def test_serialization_preserves_source_row_order():
    """Row order is part of the upstream dataset's identity."""
    forward = release.frozen_csv_bytes([("A", BENZENE, 1.0), ("B", ETHANOL, 0.0)])
    reversed_ = release.frozen_csv_bytes([("B", ETHANOL, 0.0), ("A", BENZENE, 1.0)])
    assert forward != reversed_


def test_float_labels_round_trip_exactly():
    """repr() is the shortest form that reads back as the identical float."""
    awkward = 0.1 + 0.2                      # 0.30000000000000004
    data = release.frozen_csv_bytes([("D1", BENZENE, awkward)]).decode("utf-8")
    written = data.splitlines()[1].split(",")[2]
    assert float(written) == awkward


def test_reading_back_a_frozen_file_reproduces_the_rows(tmp_path):
    rows = [("D1", BENZENE, 1.0), ("D2", TOLUENE_A, 0.0)]
    path = tmp_path / "f.csv"
    _write_frozen(path, rows)
    header, back = release.read_frozen_csv(path)
    assert header == list(release.FIELDS)
    assert [r[1] for r in back] == [BENZENE, TOLUENE_A]


def test_reading_an_empty_file_is_an_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="header"):
        release.read_frozen_csv(path)


# --------------------------------------------------------------------------
# checksums and identity
# --------------------------------------------------------------------------


def test_sha256_of_file_matches_sha256_of_its_bytes(tmp_path):
    rows = [("D1", BENZENE, 1.0)]
    path = tmp_path / "f.csv"
    _write_frozen(path, rows)
    assert release.sha256_file(path) == release.sha256_bytes(release.frozen_csv_bytes(rows))


def test_one_changed_label_changes_the_checksum(tmp_path):
    a = release.sha256_bytes(release.frozen_csv_bytes([("D1", BENZENE, 1.0)]))
    b = release.sha256_bytes(release.frozen_csv_bytes([("D1", BENZENE, 0.0)]))
    assert a != b


def test_molecule_set_identity_ignores_order_and_repetition():
    """It is a hash of the set, so it can prove set-level claims."""
    assert release.molecule_set_identity([BENZENE, ETHANOL]) == \
        release.molecule_set_identity([ETHANOL, BENZENE, BENZENE])


def test_molecule_set_identity_distinguishes_different_sets():
    assert release.molecule_set_identity([BENZENE]) != \
        release.molecule_set_identity([BENZENE, ETHANOL])


def test_release_identity_is_stable_and_ignores_endpoint_ordering():
    def entry(tag):
        return {
            "train_val": {"sha256": f"tv{tag}"},
            "test": {"sha256": f"te{tag}"},
            "split_identity": {"test_set_sha256": f"ts{tag}", "seeds": {"1": {}}},
        }

    forward = release.release_identity(
        release_name="R", protocol_version="6A.1",
        endpoints={"a": entry("a"), "b": entry("b")},
    )
    backward = release.release_identity(
        release_name="R", protocol_version="6A.1",
        endpoints={"b": entry("b"), "a": entry("a")},
    )
    assert forward == backward


def test_release_identity_changes_when_a_checksum_changes():
    def build(sha):
        return release.release_identity(
            release_name="R", protocol_version="6A.1",
            endpoints={"a": {
                "train_val": {"sha256": sha},
                "test": {"sha256": "te"},
                "split_identity": {"test_set_sha256": "ts", "seeds": {}},
            }},
        )

    assert build("one") != build("two")


def test_release_identity_changes_with_the_protocol_version():
    entry = {
        "train_val": {"sha256": "tv"}, "test": {"sha256": "te"},
        "split_identity": {"test_set_sha256": "ts", "seeds": {}},
    }
    a = release.release_identity(release_name="R", protocol_version="6A.1", endpoints={"x": entry})
    b = release.release_identity(release_name="R", protocol_version="6A.2", endpoints={"x": entry})
    assert a != b


# --------------------------------------------------------------------------
# official TDC split semantics
# --------------------------------------------------------------------------


def test_official_constants_match_what_the_pytdc_source_does():
    """Recorded from tdc/benchmark_group/base_group.py, not from memory."""
    assert tdc.OFFICIAL_TRAIN_VAL_FRACTIONS == (0.875, 0.125, 0.0)
    assert tdc.OFFICIAL_TRAIN_VAL_FRACTIONS[2] == 0.0, "test is held out, never re-drawn"
    assert tdc.OFFICIAL_SEEDS == (1, 2, 3, 4, 5)
    assert tdc.OFFICIAL_SPLIT_METHOD == "scaffold"


def test_official_and_molfusion_scaffold_conventions_differ_on_chirality():
    assert tdc.OFFICIAL_SCAFFOLD_INCLUDES_CHIRALITY is False
    assert tdc.MOLFUSION_SCAFFOLD_INCLUDES_CHIRALITY is True


def test_the_two_scaffold_conventions_can_disagree_on_a_real_molecule():
    """Why the overlap audit reports both, rather than picking one."""
    # A fused bicyclic whose ring-fusion stereocentres survive scaffold
    # reduction, so the two conventions really do produce different keys.
    chiral = "O=C1N[C@H]2CCCC[C@H]2C(=O)N1"
    assert tdc.tdc_scaffold(chiral) != splits.bemis_murcko_scaffold(chiral)


def test_task_type_is_derived_from_the_official_metric():
    assert tdc.task_type_for("roc-auc") == protocol.TASK_CLASSIFICATION
    assert tdc.task_type_for("pr-auc") == protocol.TASK_CLASSIFICATION
    assert tdc.task_type_for("mae") == protocol.TASK_REGRESSION
    assert tdc.task_type_for("spearman") == protocol.TASK_REGRESSION


def test_an_unknown_official_metric_is_refused():
    with pytest.raises(ValueError, match="unknown TDC metric"):
        tdc.task_type_for("f1")


def test_official_metric_maps_onto_a_molfusion_metric_name():
    for tdc_metric, molfusion_metric in tdc.TDC_METRIC_TO_MOLFUSION.items():
        assert tdc.task_type_for(tdc_metric) in protocol.TASK_TYPES
        assert isinstance(molfusion_metric, str)


def test_test_set_identity_does_not_change_across_seeds(frozen_endpoint):
    """The central Phase 6A.1 claim, asserted as an observation."""
    root, _ = frozen_endpoint
    identity, _ = tdc.audit_official_split(
        name="demo_endpoint", frozen_dir=root, task_type=protocol.TASK_CLASSIFICATION
    )
    hashes = {v["test_set_sha256"] for v in identity["seeds"].values()}
    assert len(hashes) == 1
    assert identity["test_identity_invariant_across_seeds"] is True


def test_train_and_validation_identity_do_change_across_seeds(frozen_endpoint):
    """The seed moves the train/validation boundary, and only that."""
    root, _ = frozen_endpoint
    identity, _ = tdc.audit_official_split(
        name="demo_endpoint", frozen_dir=root, task_type=protocol.TASK_CLASSIFICATION
    )
    train_hashes = {v["train_set_sha256"] for v in identity["seeds"].values()}
    assert len(train_hashes) == len(identity["seeds"])


def test_official_partitions_are_audited_for_canonical_overlap(frozen_endpoint):
    root, _ = frozen_endpoint
    _, overlap = tdc.audit_official_split(
        name="demo_endpoint", frozen_dir=root, task_type=protocol.TASK_CLASSIFICATION
    )
    assert overlap["canonical_molecule_overlap"] == 0


def test_overlap_audit_detects_a_molecule_shared_across_official_partitions(tmp_path):
    """A leaked molecule must be reported, not silently repaired."""
    root = tmp_path / "frozen"
    endpoint = root / "leaky"
    _write_frozen(endpoint / "train_val.csv", [("D1", TOLUENE_A, 1.0), ("D2", ETHANOL, 0.0)])
    # The same molecule as D1, spelled differently -- invisible to a raw
    # string comparison, which is exactly why the audit canonicalizes first.
    _write_frozen(endpoint / "test.csv", [("D3", TOLUENE_B, 1.0)])
    (endpoint / "official_seed_splits.json").write_text(
        json.dumps({"1": {"train_drug": [TOLUENE_A], "valid_drug": [ETHANOL]}}),
        encoding="utf-8",
    )

    _, overlap = tdc.audit_official_split(
        name="leaky", frozen_dir=root, task_type=protocol.TASK_CLASSIFICATION
    )
    assert overlap["canonical_molecule_overlap"] == 1


def test_scaffold_overlap_is_reported_under_both_conventions(frozen_endpoint):
    root, _ = frozen_endpoint
    _, overlap = tdc.audit_official_split(
        name="demo_endpoint", frozen_dir=root, task_type=protocol.TASK_CLASSIFICATION
    )
    assert "scaffold_overlap_tdc_convention" in overlap
    assert "scaffold_overlap_molfusion_convention" in overlap


def test_split_identity_records_the_official_test_fraction(frozen_endpoint):
    root, _ = frozen_endpoint
    identity, _ = tdc.audit_official_split(
        name="demo_endpoint", frozen_dir=root, task_type=protocol.TASK_CLASSIFICATION
    )
    assert identity["train_val_rows"] == 4
    assert identity["test_rows"] == 2
    assert identity["test_fraction"] == pytest.approx(2 / 6)


# --------------------------------------------------------------------------
# endpoint audit
# --------------------------------------------------------------------------


def test_endpoint_audit_runs_without_training_anything(frozen_endpoint):
    root, metadata = frozen_endpoint
    audit = tdc.audit_endpoint(name="demo_endpoint", metadata=metadata, frozen_dir=root)
    assert audit.task_type == protocol.TASK_CLASSIFICATION
    assert audit.tdc_official_metric == "roc-auc"
    assert audit.molfusion_primary_metric == "auroc"
    assert audit.raw_rows == 6


def test_endpoint_audit_covers_train_val_and_test_together(frozen_endpoint):
    """TDC split one dataset; the ingestion audit is of that dataset."""
    root, metadata = frozen_endpoint
    audit = tdc.audit_endpoint(name="demo_endpoint", metadata=metadata, frozen_dir=root)
    assert audit.raw_rows == audit.checksums["train_val"]["rows"] + audit.checksums["test"]["rows"]


def test_endpoint_audit_accounting_balances(frozen_endpoint):
    """Every input record reaches exactly one disposition."""
    root, metadata = frozen_endpoint
    audit = tdc.audit_endpoint(name="demo_endpoint", metadata=metadata, frozen_dir=root)
    report = audit.ingestion
    accounted = (
        report["missing_label_dropped"]
        + report["rdkit_invalid"]
        + report["duplicates_collapsed"]
        + report["duplicates_conflicting_dropped"]
        + report["usable"]
    )
    assert report["input_records"] == audit.raw_rows
    assert accounted == report["input_records"]


def test_endpoint_audit_records_both_metric_concepts(frozen_endpoint):
    """Leaderboard comparability and cross-endpoint analysis are separate."""
    root, metadata = frozen_endpoint
    audit = tdc.audit_endpoint(name="demo_endpoint", metadata=metadata, frozen_dir=root)
    assert audit.tdc_official_metric != audit.molfusion_primary_metric or \
        audit.tdc_official_metric in tdc.TDC_METRIC_TO_MOLFUSION.values()
    assert audit.tdc_official_metric in tdc.TDC_METRIC_TO_MOLFUSION


def test_endpoint_audit_reports_checksums_for_both_frozen_files(frozen_endpoint):
    root, metadata = frozen_endpoint
    audit = tdc.audit_endpoint(name="demo_endpoint", metadata=metadata, frozen_dir=root)
    for part in ("train_val", "test"):
        assert len(audit.checksums[part]["sha256"]) == 64
        assert audit.checksums[part]["bytes"] > 0


def test_endpoint_audit_reports_scaffold_profile(frozen_endpoint):
    root, metadata = frozen_endpoint
    audit = tdc.audit_endpoint(name="demo_endpoint", metadata=metadata, frozen_dir=root)
    assert audit.scaffolds["unique_scaffolds"] >= 1
    # ethanol and alanine are acyclic and share the explicit group key
    assert audit.scaffolds["acyclic_group_size"] == 2


def test_a_small_endpoint_is_excluded_with_a_reason(frozen_endpoint):
    """Six molecules is far below the frozen minimum; say so explicitly."""
    root, metadata = frozen_endpoint
    audit = tdc.audit_endpoint(name="demo_endpoint", metadata=metadata, frozen_dir=root)
    assert audit.included is False
    assert audit.exclusion_reasons
    assert any(str(protocol.MINIMUM_MOLECULES) in r for r in audit.exclusion_reasons)


def test_invalid_smiles_are_counted_not_silently_skipped(tmp_path):
    root = tmp_path / "frozen"
    endpoint = root / "withbad"
    _write_frozen(endpoint / "train_val.csv", [("D1", BENZENE, 1.0), ("D2", BROKEN, 0.0)])
    _write_frozen(endpoint / "test.csv", [("D3", ETHANOL, 1.0)])
    (endpoint / "official_seed_splits.json").write_text(
        json.dumps({"1": {"train_drug": [BENZENE], "valid_drug": [BROKEN]}}), encoding="utf-8"
    )
    metadata = {"tdc_category": "ADME", "tdc_official_metric": "roc-auc"}
    audit = tdc.audit_endpoint(name="withbad", metadata=metadata, frozen_dir=root)
    assert audit.ingestion["rdkit_invalid"] == 1
    assert audit.ingestion["usable"] == 2
    assert audit.ingestion["input_records"] == 3


def test_duplicate_spellings_of_one_molecule_collapse(tmp_path):
    """Canonicalization happens before deduplication, so spelling cannot hide."""
    root = tmp_path / "frozen"
    endpoint = root / "dupes"
    _write_frozen(endpoint / "train_val.csv",
                  [("D1", TOLUENE_A, 1.0), ("D2", TOLUENE_B, 1.0), ("D3", ETHANOL, 0.0)])
    _write_frozen(endpoint / "test.csv", [("D4", BENZENE, 1.0)])
    (endpoint / "official_seed_splits.json").write_text(
        json.dumps({"1": {"train_drug": [TOLUENE_A], "valid_drug": [ETHANOL]}}), encoding="utf-8"
    )
    metadata = {"tdc_category": "ADME", "tdc_official_metric": "roc-auc"}
    audit = tdc.audit_endpoint(name="dupes", metadata=metadata, frozen_dir=root)
    assert audit.ingestion["duplicates_collapsed"] == 1
    assert audit.usable == 3


def test_conflicting_labels_drop_every_copy(tmp_path):
    root = tmp_path / "frozen"
    endpoint = root / "conflict"
    _write_frozen(endpoint / "train_val.csv",
                  [("D1", TOLUENE_A, 1.0), ("D2", TOLUENE_B, 0.0), ("D3", ETHANOL, 0.0)])
    _write_frozen(endpoint / "test.csv", [("D4", BENZENE, 1.0)])
    (endpoint / "official_seed_splits.json").write_text(
        json.dumps({"1": {"train_drug": [ETHANOL], "valid_drug": [BENZENE]}}), encoding="utf-8"
    )
    metadata = {"tdc_category": "ADME", "tdc_official_metric": "roc-auc"}
    audit = tdc.audit_endpoint(name="conflict", metadata=metadata, frozen_dir=root)
    assert audit.ingestion["duplicates_conflicting_dropped"] == 2
    # toluene is gone entirely: both copies dropped, not one kept
    assert audit.usable == 2


def test_regression_endpoint_reports_label_spread(tmp_path):
    """The spread is what the relative conflict tolerance is a fraction of."""
    root = tmp_path / "frozen"
    endpoint = root / "reg"
    _write_frozen(endpoint / "train_val.csv",
                  [("D1", BENZENE, 1.0), ("D2", ETHANOL, 11.0)])
    _write_frozen(endpoint / "test.csv", [("D3", PYRIDINE, 6.0)])
    (endpoint / "official_seed_splits.json").write_text(
        json.dumps({"1": {"train_drug": [BENZENE], "valid_drug": [ETHANOL]}}), encoding="utf-8"
    )
    metadata = {"tdc_category": "ADME", "tdc_official_metric": "mae"}
    audit = tdc.audit_endpoint(name="reg", metadata=metadata, frozen_dir=root)
    assert audit.task_type == protocol.TASK_REGRESSION
    assert audit.label_summary["min"] == 1.0
    assert audit.label_summary["max"] == 11.0
    assert audit.label_summary["spread"] == 10.0
    assert audit.label_summary["conflict_tolerance"] == pytest.approx(0.1)


def test_classification_endpoint_reports_class_balance(frozen_endpoint):
    root, metadata = frozen_endpoint
    audit = tdc.audit_endpoint(name="demo_endpoint", metadata=metadata, frozen_dir=root)
    assert "minority_fraction" in audit.label_summary
    assert audit.label_summary["minority_fraction"] <= 0.5


# --------------------------------------------------------------------------
# Phase 6A.1 reconciliation: the two evaluation tracks
# --------------------------------------------------------------------------


def test_track_a1_uses_a_fixed_test_set():
    """The whole point of the amendment."""
    assert protocol.TRACK_A1_TEST_IS_FIXED is True
    assert protocol.TRACK_A1_TRAIN_VAL_FRACTIONS == tdc.OFFICIAL_TRAIN_VAL_FRACTIONS


def test_track_a2_still_draws_its_own_test_partitions():
    assert protocol.TRACK_A2_TEST_IS_FIXED is False
    assert protocol.TRACK_A2_SEEDS == protocol.SPLIT_SEEDS


def test_phase_6a_fractions_survive_because_they_nest_correctly():
    """0.8 * 0.875 = 0.70 train and 0.8 * 0.125 = 0.10 validation."""
    test_fraction = 1.0 - protocol.TEST_FRACTION
    assert test_fraction * protocol.TRACK_A1_TRAIN_VAL_FRACTIONS[0] == pytest.approx(
        protocol.TRAIN_FRACTION
    )
    assert test_fraction * protocol.TRACK_A1_TRAIN_VAL_FRACTIONS[1] == pytest.approx(
        protocol.VALIDATION_FRACTION
    )


def test_the_two_tracks_use_different_seed_values():
    """So a bare seed can never make a result row ambiguous."""
    assert set(protocol.TRACK_A1_SEEDS) != set(protocol.TRACK_A2_SEEDS)


def test_split_id_names_its_track():
    a1 = protocol.split_id(protocol.TRACK_A1, 1)
    a2 = protocol.split_id(protocol.TRACK_A2, 1)
    assert a1 != a2
    assert protocol.TRACK_A1 in a1 and protocol.TRACK_A2 in a2


def test_split_id_refuses_an_unknown_track():
    with pytest.raises(ValueError, match="unknown evaluation track"):
        protocol.split_id("official_tdc_probably", 1)


def test_track_a2_is_never_labelled_official():
    """A2 is a robustness analysis, not a leaderboard result."""
    assert protocol.TRACK_A2_STATUS == "supplementary"
    assert "not comparable with" in protocol.TRACK_A2_DEFINITION.lower()


def test_track_a1_applies_no_cleaning_to_official_rows():
    """Cleaning would score A1 on a different set than the leaderboard used."""
    assert protocol.TRACK_A1_CLEANING.startswith("none")
    assert "shipped" in protocol.TRACK_A1_CLEANING


def test_a_cleaning_divergence_threshold_exists_and_is_reported():
    assert 0.0 < protocol.CLEANING_DIVERGENCE_ALERT < 1.0


def test_protocol_summary_exposes_both_tracks():
    summary = protocol.protocol_summary()
    tracks = summary["evaluation_tracks"]
    assert set(tracks) == set(protocol.EVALUATION_TRACKS)
    assert tracks[protocol.TRACK_A1]["status"] == "primary"
    assert tracks[protocol.TRACK_A2]["status"] == "supplementary"


def test_protocol_version_is_unchanged_by_the_amendment():
    """6A.1 amends the reading of TDC's protocol, not MolFusion's own."""
    assert protocol.PROTOCOL_VERSION == "6A.1"
