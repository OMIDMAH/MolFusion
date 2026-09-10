"""Tests for the frozen Phase 6A benchmark protocol utilities.

None of these require a network download or a large dataset: the protocol
must be testable independently of the data it will eventually run on.
"""

import json

import numpy as np
import pytest

from molfusion_backend.benchmark import (
    cache,
    datasets,
    features,
    metrics,
    pipelines,
    protocol,
    results,
    splits,
)
from molfusion_backend.chemistry import canonicalize_smiles

CORES = [
    "c1ccc(cc1){R}",
    "c1ccc2ccccc2c1{R}",
    "c1ccncc1{R}",
    "c1csc(n1){R}",
    "O=C1CCCN1{R}",
    "c1ccc(cc1)S(=O)(=O){R}",
]
# Enough cores x substituents to clear the protocol's own 100-molecule
# inclusion floor, with variation inside each scaffold so a scaffold split
# still yields both classes in every partition.
SUBSTITUENTS = [
    "C", "CC", "CCC", "CCCC", "CCCCC", "O", "OC", "OCC", "N", "NC",
    "NCC", "Cl", "F", "Br", "C(=O)O", "C(=O)N", "S", "SC", "C#N", "CO",
]


def fixture_smiles():
    from rdkit import Chem

    seen = {}
    for core in CORES:
        for substituent in SUBSTITUENTS:
            smiles = core.replace("{R}", substituent)
            if Chem.MolFromSmiles(smiles) is not None:
                seen[canonicalize_smiles(smiles)] = smiles
    return sorted(seen.values())


def fixture_rows(task_type=protocol.TASK_CLASSIFICATION):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    weights = [
        (smiles, float(Descriptors.MolWt(Chem.MolFromSmiles(smiles))))
        for smiles in fixture_smiles()
    ]
    median = sorted(weight for _, weight in weights)[len(weights) // 2]
    if task_type == protocol.TASK_REGRESSION:
        return [(smiles, weight) for smiles, weight in weights]
    return [(smiles, float(weight > median)) for smiles, weight in weights]


@pytest.fixture(scope="module")
def molecules():
    built, _ = datasets.build_dataset(fixture_rows(), task_type=protocol.TASK_CLASSIFICATION)
    return built


# ---------------------------------------------------------------------------
# the protocol is frozen and self-consistent
# ---------------------------------------------------------------------------


def test_track_a_holds_the_seven_fixed_vector_representations():
    assert protocol.TRACK_A_REPRESENTATIONS == (
        "morgan_ecfp4_1024",
        "maccs_keys_167",
        "rdkit_physchem_descriptors",
        "avalon_1024",
        "erg_reduced_graph_315",
        "rdkit_fragment_descriptors",
        "smiles_tfidf_4096",
    )


def test_selfies_is_in_track_b_not_track_a():
    """The decision must be explicit in code, not only in prose."""
    assert "selfies_sequence" in protocol.TRACK_B_REPRESENTATIONS
    assert "selfies_sequence" not in protocol.TRACK_A_REPRESENTATIONS
    assert protocol.TRACK_B_DECISION


def test_track_a_agents_are_all_registered_fixed_vector_agents():
    from molfusion_backend.agents import registry

    for name in protocol.TRACK_A_REPRESENTATIONS:
        agent = registry.get(name)
        assert agent.output_structure == "vector"
        assert isinstance(agent.output_dim, int)


def test_every_track_a_representation_has_a_scaling_policy():
    for name in protocol.TRACK_A_REPRESENTATIONS:
        assert name in protocol.LINEAR_SCALING
        assert name in protocol.NONLINEAR_SCALING


def test_split_fractions_sum_to_one():
    total = protocol.TRAIN_FRACTION + protocol.VALIDATION_FRACTION + protocol.TEST_FRACTION
    assert total == pytest.approx(1.0)


def test_the_protocol_summary_is_json_serializable():
    """It is written into every result manifest, so it must serialize."""
    payload = json.dumps(protocol.protocol_summary())
    assert protocol.PROTOCOL_VERSION in payload


def test_the_tfidf_artifact_identity_is_pinned():
    assert protocol.TFIDF_ARTIFACT_IDENTITY == "smiles_tfidf/chembl37_token_ngrams_1_3/1.0.0"
    assert "never refitted" in protocol.TFIDF_REFIT_POLICY


# ---------------------------------------------------------------------------
# dataset ingestion, duplicates and conflicts
# ---------------------------------------------------------------------------


def test_canonical_duplicates_are_detected_across_spellings():
    """"CCO" and "OCC" are one molecule; deduplicating on the raw string
    would leave the same compound in train and test."""
    built, audit = datasets.build_dataset(
        [("CCO", 1.0), ("OCC", 1.0)], task_type=protocol.TASK_CLASSIFICATION
    )
    assert len(built) == 1
    assert audit.duplicates_collapsed == 1


def test_conflicting_labels_drop_every_copy():
    built, audit = datasets.build_dataset(
        [("CCO", 1.0), ("OCC", 0.0), ("c1ccccc1", 1.0)],
        task_type=protocol.TASK_CLASSIFICATION,
    )
    assert [molecule.canonical_smiles for molecule in built] == ["c1ccccc1"]
    assert audit.duplicates_conflicting_dropped == 2
    assert audit.conflicting_molecules == [canonicalize_smiles("CCO")]


def test_a_conflicting_label_is_never_averaged():
    built, _ = datasets.build_dataset(
        [("CCO", 0.0), ("CCO", 1.0)], task_type=protocol.TASK_CLASSIFICATION
    )
    assert built == []


def test_regression_duplicates_within_tolerance_agree():
    """Two measurements of one compound rarely match to the last float."""
    rows = [("CCO", 1.000), ("OCC", 1.0005), ("c1ccccc1", 100.0)]
    built, audit = datasets.build_dataset(rows, task_type=protocol.TASK_REGRESSION)
    assert audit.duplicates_collapsed == 1
    assert audit.duplicates_conflicting_dropped == 0
    assert len(built) == 2


def test_regression_duplicates_beyond_tolerance_conflict():
    rows = [("CCO", 1.0), ("OCC", 90.0), ("c1ccccc1", 100.0)]
    _, audit = datasets.build_dataset(rows, task_type=protocol.TASK_REGRESSION)
    assert audit.duplicates_conflicting_dropped == 2


def test_missing_labels_are_dropped_never_imputed():
    built, audit = datasets.build_dataset(
        [("CCO", None), ("c1ccccc1", 1.0)], task_type=protocol.TASK_CLASSIFICATION
    )
    assert audit.missing_label_dropped == 1
    assert [molecule.label for molecule in built] == [1.0]


def test_unparseable_smiles_are_counted_not_crashed_on():
    built, audit = datasets.build_dataset(
        [("not-a-molecule", 1.0), ("CCO", 0.0)], task_type=protocol.TASK_CLASSIFICATION
    )
    assert audit.rdkit_invalid == 1
    assert len(built) == 1


def test_every_input_record_is_accounted_for():
    rows = [("CCO", 1.0), ("OCC", 0.0), ("bad", 1.0), ("c1ccccc1", None), ("CCN", 1.0)]
    _, audit = datasets.build_dataset(rows, task_type=protocol.TASK_CLASSIFICATION)
    audit.validate()  # raises if the categories do not sum to the input count
    assert audit.input_records == 5


def test_ingestion_is_independent_of_row_order():
    rows = fixture_rows()
    forward, _ = datasets.build_dataset(rows, task_type=protocol.TASK_CLASSIFICATION)
    backward, _ = datasets.build_dataset(
        list(reversed(rows)), task_type=protocol.TASK_CLASSIFICATION
    )
    assert forward == backward


def test_an_unknown_task_type_is_rejected():
    with pytest.raises(ValueError, match="task_type"):
        datasets.build_dataset([("CCO", 1.0)], task_type="ranking")


# ---------------------------------------------------------------------------
# inclusion criteria
# ---------------------------------------------------------------------------


def test_a_small_endpoint_is_excluded_with_a_reason(molecules):
    included, reasons = datasets.check_inclusion(
        molecules[:10], task_type=protocol.TASK_CLASSIFICATION
    )
    assert included is False
    assert any("minimum" in reason for reason in reasons)


def test_a_severely_imbalanced_endpoint_is_excluded():
    built, _ = datasets.build_dataset(
        [(smiles, 0.0) for smiles in fixture_smiles()[:50]]
        + [(smiles, 1.0) for smiles in fixture_smiles()[50:52]],
        task_type=protocol.TASK_CLASSIFICATION,
    )
    included, reasons = datasets.check_inclusion(
        built, task_type=protocol.TASK_CLASSIFICATION
    )
    assert included is False
    assert any("minority" in reason for reason in reasons)


def test_a_healthy_endpoint_is_included(molecules):
    included, reasons = datasets.check_inclusion(
        molecules, task_type=protocol.TASK_CLASSIFICATION
    )
    assert included is True
    assert reasons == []


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def test_scaffold_splits_are_deterministic(molecules):
    first = splits.scaffold_split(molecules, seed=0)
    second = splits.scaffold_split(molecules, seed=0)
    assert first == second


def test_different_seeds_give_different_splits(molecules):
    assert splits.scaffold_split(molecules, seed=0) != splits.scaffold_split(
        molecules, seed=1
    )


def test_a_split_is_independent_of_molecule_order(molecules):
    """Assignment depends on (scaffold, seed) alone, not input order."""
    forward = splits.scaffold_split(molecules, seed=3)
    shuffled = list(reversed(molecules))
    backward = splits.scaffold_split(shuffled, seed=3)

    forward_test = {molecules[index].canonical_smiles for index in forward.test}
    backward_test = {shuffled[index].canonical_smiles for index in backward.test}
    assert forward_test == backward_test


def test_no_molecule_appears_in_two_partitions(molecules):
    for seed in protocol.SPLIT_SEEDS:
        audit = splits.audit_split(molecules, splits.scaffold_split(molecules, seed=seed))
        assert audit["molecule_overlap"] == {
            "train_test": 0,
            "train_validation": 0,
            "validation_test": 0,
        }


def test_no_scaffold_straddles_the_train_test_boundary(molecules):
    """The property that makes a scaffold split worth the name."""
    for seed in protocol.SPLIT_SEEDS:
        audit = splits.audit_split(molecules, splits.scaffold_split(molecules, seed=seed))
        assert audit["scaffold_overlap"]["train_test"] == 0


def test_a_split_partitions_the_dataset_exactly_once(molecules):
    split = splits.scaffold_split(molecules, seed=0)
    combined = sorted([*split.train, *split.validation, *split.test])
    assert combined == list(range(len(molecules)))


def test_acyclic_molecules_form_one_explicit_group():
    assert splits.bemis_murcko_scaffold("CCO") == protocol.EMPTY_SCAFFOLD_KEY
    assert splits.bemis_murcko_scaffold("CCCC") == protocol.EMPTY_SCAFFOLD_KEY
    assert splits.bemis_murcko_scaffold("c1ccccc1CC") == "c1ccccc1"


def test_scaffold_computation_rejects_an_unparseable_molecule():
    with pytest.raises(ValueError, match="scaffold"):
        splits.bemis_murcko_scaffold("not-a-molecule")


def test_generate_splits_uses_the_frozen_seeds(molecules):
    generated = splits.generate_splits(molecules)
    assert len(generated) == protocol.N_SPLITS
    assert [split.seed for split in generated] == list(protocol.SPLIT_SEEDS)


def test_an_empty_dataset_cannot_be_split():
    with pytest.raises(ValueError, match="empty"):
        splits.scaffold_split([], seed=0)


# ---------------------------------------------------------------------------
# metrics and direction
# ---------------------------------------------------------------------------


def test_metric_directions_are_declared_correctly():
    assert metrics.is_lower_better("mae") is True
    assert metrics.is_lower_better("rmse") is True
    for metric in ("auroc", "auprc", "r2", "spearman", "balanced_accuracy", "mcc"):
        assert metrics.is_lower_better(metric) is False


def test_orientation_inverts_only_lower_is_better_metrics():
    assert metrics.orient("auroc", 0.9) == 0.9
    assert metrics.orient("mae", 0.9) == -0.9


def test_classification_metrics_reward_a_better_ranking():
    truth = [0, 0, 1, 1]
    good = metrics.classification_metrics(truth, [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    bad = metrics.classification_metrics(truth, [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert good["auroc"] == 1.0
    assert bad["auroc"] == 0.0
    assert good["auprc"] > bad["auprc"]


def test_classification_metrics_refuse_a_single_class_fold():
    """AUROC and AUPRC are undefined there; a fabricated value would be worse
    than an error."""
    with pytest.raises(ValueError, match="both classes"):
        metrics.classification_metrics([1, 1, 1], [0.2, 0.5, 0.9], [1, 1, 1])


def test_regression_metrics_reward_a_better_fit():
    truth = [1.0, 2.0, 3.0, 4.0]
    good = metrics.regression_metrics(truth, [1.1, 2.1, 2.9, 4.1])
    bad = metrics.regression_metrics(truth, [4.0, 3.0, 2.0, 1.0])
    assert good["mae"] < bad["mae"]
    assert good["r2"] > bad["r2"]
    assert good["spearman"] == pytest.approx(1.0)
    assert bad["spearman"] == pytest.approx(-1.0)


def test_a_constant_prediction_gives_nan_spearman_not_zero():
    """NaN is the honest value: 0.0 would read as "no correlation measured"
    rather than "not measurable". SciPy's warning about the constant input
    is expected here, so it is asserted rather than leaked into the run."""
    from scipy.stats import ConstantInputWarning

    with pytest.warns(ConstantInputWarning):
        scores = metrics.regression_metrics([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    assert scores["spearman"] != scores["spearman"]  # NaN


def test_primary_metrics_match_the_protocol():
    assert metrics.primary_metric(protocol.TASK_CLASSIFICATION) == "auroc"
    assert metrics.primary_metric(protocol.TASK_REGRESSION) == "mae"
    with pytest.raises(ValueError):
        metrics.primary_metric("ranking")


def test_ranking_respects_metric_direction():
    higher = metrics.rank_within_endpoint({"a": 0.9, "b": 0.7, "c": 0.8}, "auroc")
    assert higher == {"a": 1.0, "c": 2.0, "b": 3.0}
    lower = metrics.rank_within_endpoint({"a": 0.9, "b": 0.7, "c": 0.8}, "mae")
    assert lower == {"b": 1.0, "c": 2.0, "a": 3.0}


def test_ties_receive_the_average_rank():
    ranks = metrics.rank_within_endpoint({"a": 1.0, "b": 1.0, "c": 0.5}, "auroc")
    assert ranks == {"a": 1.5, "b": 1.5, "c": 3.0}


def test_rank_aggregation_reports_mean_median_and_wins():
    summary = metrics.aggregate_ranks(
        [{"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 2.0}]
    )
    assert summary["a"]["wins"] == 2
    assert summary["b"]["wins"] == 1
    assert summary["a"]["mean_rank"] == pytest.approx(4 / 3)


# ---------------------------------------------------------------------------
# pipelines: scaling policy and training-only fitting
# ---------------------------------------------------------------------------


def test_binary_fingerprints_are_not_standardized_for_the_linear_probe():
    """Centering a bit vector destroys sparsity and gives an absent bit a
    nonzero value, which is not what absence means."""
    for name in ("morgan_ecfp4_1024", "maccs_keys_167", "avalon_1024"):
        assert pipelines.scaling_for(name, protocol.PROBE_LINEAR) == protocol.SCALING_NONE


def test_continuous_descriptors_are_standardized_for_the_linear_probe():
    assert (
        pipelines.scaling_for("rdkit_physchem_descriptors", protocol.PROBE_LINEAR)
        == protocol.SCALING_STANDARD
    )


def test_tfidf_is_left_alone_because_it_is_already_l2_normalized():
    assert (
        pipelines.scaling_for("smiles_tfidf_4096", protocol.PROBE_LINEAR)
        == protocol.SCALING_NONE
    )


def test_the_tree_probe_never_scales():
    for name in protocol.TRACK_A_REPRESENTATIONS:
        assert pipelines.scaling_for(name, protocol.PROBE_NONLINEAR) == protocol.SCALING_NONE


def test_an_unknown_representation_has_no_scaling_policy():
    with pytest.raises(ValueError, match="no scaling policy"):
        pipelines.scaling_for("not_a_representation", protocol.PROBE_LINEAR)


def test_an_unknown_probe_is_rejected():
    with pytest.raises(ValueError, match="probe"):
        pipelines.scaling_for("morgan_ecfp4_1024", "quantum")


def test_scalers_are_fitted_on_training_data_only():
    """The classic silent leak. Structurally prevented by Pipeline: fitting
    on train and transforming test must not change the fitted statistics."""
    train = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    test = np.array([[1000.0, 2000.0]])
    model = pipelines.build_pipeline(
        representation="rdkit_physchem_descriptors",
        probe=protocol.PROBE_LINEAR,
        task_type=protocol.TASK_REGRESSION,
    )
    model.fit(train, np.array([1.0, 2.0, 3.0]))
    fitted_mean = model.named_steps["scale"].mean_.copy()
    model.predict(test)
    assert np.array_equal(model.named_steps["scale"].mean_, fitted_mean)
    assert np.allclose(fitted_mean, train.mean(axis=0))


def test_the_linear_probe_imputes_the_nan_rdkit_descriptors_produce():
    train = np.array([[1.0, 1.0], [3.0, np.nan], [5.0, 5.0]])
    model = pipelines.build_pipeline(
        representation="rdkit_physchem_descriptors",
        probe=protocol.PROBE_LINEAR,
        task_type=protocol.TASK_REGRESSION,
    )
    model.fit(train, np.array([1.0, 2.0, 3.0]))
    assert np.all(np.isfinite(model.predict(train)))


def test_the_hyperparameter_grid_is_identical_for_every_representation():
    """The grid function does not take a representation, so a per-
    representation budget cannot be introduced by accident."""
    grid = pipelines.hyperparameter_grid(protocol.PROBE_LINEAR, protocol.TASK_CLASSIFICATION)
    assert len(grid) == len(protocol.LINEAR_CLASSIFIER_GRID["C"])
    nonlinear = pipelines.hyperparameter_grid(
        protocol.PROBE_NONLINEAR, protocol.TASK_REGRESSION
    )
    assert len(nonlinear) == 4


def test_model_names_are_recorded_per_probe_and_task():
    assert "LogisticRegression" in pipelines.model_name(
        protocol.PROBE_LINEAR, protocol.TASK_CLASSIFICATION
    )
    assert "Ridge" in pipelines.model_name(protocol.PROBE_LINEAR, protocol.TASK_REGRESSION)
    assert "HistGradientBoosting" in pipelines.model_name(
        protocol.PROBE_NONLINEAR, protocol.TASK_CLASSIFICATION
    )


# ---------------------------------------------------------------------------
# feature extraction and the common evaluation set
# ---------------------------------------------------------------------------


def test_extraction_reports_dimension_sparsity_and_cost(molecules):
    result = features.extract(
        [molecule.canonical_smiles for molecule in molecules[:20]], "maccs_keys_167"
    )
    profile = features.representation_profile(result)
    assert profile["dimension"] == 167
    assert 0.0 <= profile["sparsity"] <= 1.0
    assert profile["feature_seconds"] > 0
    assert profile["failures"] == 0


def test_extraction_refuses_a_sequence_agent():
    """Track B is not silently encoded into a vector."""
    with pytest.raises(ValueError, match="Track A"):
        features.extract(["CCO"], "selfies_sequence")


def test_extraction_isolates_a_per_molecule_failure(monkeypatch):
    from molfusion_backend.agents import registry

    agent = registry.get("maccs_keys_167")
    original = agent.compute

    def flaky(mol):
        from molfusion_backend.chemistry import canonical_smiles_from_mol

        if canonical_smiles_from_mol(mol) == "CCO":
            raise ValueError("synthetic per-molecule failure")
        return original(mol)

    monkeypatch.setattr(agent, "compute", flaky)
    result = features.extract(["CCO", "c1ccccc1"], "maccs_keys_167")
    assert result.succeeded == (1,)
    assert 0 in result.failures


def test_the_common_evaluation_set_is_the_intersection():
    """A representation must not get an easier test set by failing on the
    hard molecules."""
    left = features.ExtractionResult(
        "a", "1", np.zeros((3, 2)), succeeded=(0, 1, 2), failures={}
    )
    right = features.ExtractionResult(
        "b", "1", np.zeros((2, 2)), succeeded=(0, 2), failures={1: "boom"}
    )
    common, accounting = features.common_evaluation_set([left, right], total=3)
    assert common == (0, 2)
    assert accounting["lost_to_intersection"] == 1
    assert accounting["per_representation_failures"] == {"a": 0, "b": 1}


def test_a_large_intersection_loss_is_flagged():
    left = features.ExtractionResult("a", "1", np.zeros((100, 2)), tuple(range(100)), {})
    right = features.ExtractionResult("b", "1", np.zeros((90, 2)), tuple(range(90)), {})
    _, accounting = features.common_evaluation_set([left, right], total=100)
    assert accounting["exceeds_alert_threshold"] is True


# ---------------------------------------------------------------------------
# result schema and reproducibility metadata
# ---------------------------------------------------------------------------


def sample_row(**overrides):
    defaults = dict(
        dataset="d",
        endpoint="e",
        task_type=protocol.TASK_CLASSIFICATION,
        split_id="s",
        split_strategy=protocol.SPLIT_SCAFFOLD,
        seed=0,
        representation="morgan_ecfp4_1024",
        representation_version="1.0.0",
        model="m",
        probe=protocol.PROBE_LINEAR,
        metric="auroc",
        value=0.8,
        n_train=10,
        n_valid=2,
        n_test=3,
        feature_dim=1024,
        feature_failures=0,
    )
    defaults.update(overrides)
    return results.ResultRow(**defaults)


def test_results_are_long_format_one_row_per_metric(tmp_path):
    rows = [sample_row(metric="auroc", value=0.8), sample_row(metric="auprc", value=0.6)]
    path = tmp_path / "results.csv"
    assert results.write_results(path, rows) == 2

    read_back = results.read_results(path)
    assert list(read_back[0]) == list(results.RESULT_FIELDS)
    assert {row["metric"] for row in read_back} == {"auroc", "auprc"}


def test_the_result_schema_keeps_feature_and_model_cost_separate():
    assert "feature_seconds" in results.RESULT_FIELDS
    assert "fit_seconds" in results.RESULT_FIELDS
    assert "predict_seconds" in results.RESULT_FIELDS


def test_the_result_schema_records_split_and_seed():
    for field_name in ("split_id", "split_strategy", "seed"):
        assert field_name in results.RESULT_FIELDS


def test_results_are_written_with_lf_and_no_bom(tmp_path):
    path = tmp_path / "results.csv"
    results.write_results(path, [sample_row()])
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert not raw.startswith(b"\xef\xbb\xbf")


def test_split_level_scores_are_retained_not_only_their_mean():
    rows = [
        sample_row(seed=0, value=0.80),
        sample_row(seed=1, value=0.90),
        sample_row(seed=2, value=0.70),
    ]
    summary = results.summarize_across_splits(rows, "auroc")
    cell = summary[("e", "morgan_ecfp4_1024")]
    assert cell["n_splits"] == 3
    assert cell["mean"] == pytest.approx(0.80)
    assert cell["std"] > 0
    assert cell["min"] == 0.70 and cell["max"] == 0.90


def test_the_manifest_records_the_software_and_protocol(tmp_path):
    manifest = results.new_manifest()
    assert manifest.protocol_version == protocol.PROTOCOL_VERSION
    assert manifest.software["python"]
    assert manifest.software["rdkit"]
    assert manifest.software["sklearn"]
    assert "molfusion_git_commit" in manifest.software

    path = tmp_path / "manifest.json"
    results.write_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["protocol_version"] == protocol.PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# feature cache contract
# ---------------------------------------------------------------------------


def base_key(**overrides):
    defaults = dict(
        canonical_smiles="CCO",
        agent_id="morgan_ecfp4_1024",
        agent_version="1.0.0",
        normalization_id=protocol.CANONICALIZATION_ID,
    )
    defaults.update(overrides)
    return cache.feature_cache_key(**defaults)


def test_the_cache_key_is_stable_for_identical_inputs():
    assert base_key() == base_key()


@pytest.mark.parametrize(
    "override",
    [
        {"canonical_smiles": "CCN"},
        {"agent_id": "maccs_keys_167"},
        {"agent_version": "2.0.0"},
        {"normalization_id": "something_else_v1"},
        {"artifact_identity": "smiles_tfidf/chembl37_token_ngrams_1_3/1.0.0"},
    ],
)
def test_every_key_component_changes_the_cache_key(override):
    assert base_key(**override) != base_key()


def test_the_cache_key_distinguishes_artifact_versions():
    """An artifact-backed agent produces different vectors from identical
    source when its artifact changes, so the artifact identity is keyed."""
    first = base_key(artifact_identity="smiles_tfidf/chembl37_token_ngrams_1_3/1.0.0")
    second = base_key(artifact_identity="smiles_tfidf/chembl37_token_ngrams_1_3/2.0.0")
    assert first != second


def test_components_cannot_impersonate_one_another():
    """A separator-free concatenation would let one field's suffix look like
    the next field's prefix."""
    assert base_key(agent_id="a", agent_version="bc") != base_key(
        agent_id="ab", agent_version="c"
    )


def test_the_cache_contract_excludes_filenames():
    contract = cache.cache_contract()
    assert "filename" in contract["not_keyed_on"]
    assert "agent_version" in contract["key_components"]
