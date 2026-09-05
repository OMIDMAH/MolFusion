"""Phase 6A.4: the Track A2 execution and analysis contract.

Nothing here runs the benchmark. What is tested is what makes A2 a valid
repartitioning experiment rather than A1 with extra steps: that the five
partitions are independent and audited, that partition variability is
measured rather than assumed, that A2 can never be served an A1 feature
matrix, that stability is computed at the split level while the statistical
unit stays the endpoint, and that the A1/A2 comparison classifies outcomes
by a rule fixed in advance.
"""

import json

import numpy as np
import pytest

from molfusion_backend.benchmark import (
    a2,
    a2_runner,
    analysis,
    analysis_a2,
    feature_store,
    protocol,
)

REPS = list(protocol.TRACK_A_REPRESENTATIONS)


def _row(endpoint, representation, probe, seed, metric, value, task_type="classification"):
    base = {c: 0 for c in a2_runner.RESULT_COLUMNS}
    base.update({
        "benchmark_release": "r" * 64, "track": protocol.TRACK_A2, "endpoint": endpoint,
        "task_type": task_type, "split_id": protocol.split_id(protocol.TRACK_A2, seed),
        "split_strategy": protocol.TRACK_A2_SPLIT_STRATEGY, "seed": seed,
        "representation": representation, "agent_version": "1.0.0", "model_family": "M",
        "probe": probe, "hyperparameters": json.dumps({"C": 1.0}), "metric": metric,
        "metric_value": value,
        "molfusion_primary_metric": "auroc" if task_type == "classification" else "mae",
        "tdc_official_metric": "roc-auc" if task_type == "classification" else "mae",
    })
    return base


# --------------------------------------------------------------------------
# track separation
# --------------------------------------------------------------------------


def test_a2_seed_set_differs_from_a1_but_the_values_overlap():
    """The tracks are NOT disambiguated by seed alone -- values 1-4 occur in
    both. What disambiguates a row is the track column and the split_id.
    An earlier protocol note claimed otherwise; it was corrected in 6A.4."""
    assert a2.SEEDS == tuple(protocol.TRACK_A2_SEEDS)
    assert set(a2.SEEDS) != set(protocol.TRACK_A1_SEEDS)
    assert set(a2.SEEDS) & set(protocol.TRACK_A1_SEEDS) == {1, 2, 3, 4}


def test_rows_are_disambiguated_by_track_and_split_id_not_by_seed():
    shared = 3
    assert shared in protocol.TRACK_A1_SEEDS and shared in protocol.TRACK_A2_SEEDS
    a1_id = protocol.split_id(protocol.TRACK_A1, shared)
    a2_id = protocol.split_id(protocol.TRACK_A2, shared)
    assert a1_id != a2_id
    assert "track" in a2_runner.SCIENTIFIC_COLUMNS
    assert "split_id" in a2_runner.SCIENTIFIC_COLUMNS


def test_a2_split_ids_name_their_track():
    assert protocol.split_id(a2.TRACK, 0).startswith(protocol.TRACK_A2)
    assert protocol.split_id(a2.TRACK, 0) != protocol.split_id(protocol.TRACK_A1, 0)


def test_a2_feature_cache_key_cannot_collide_with_a1():
    """Same release, same agent, same molecules -- different track namespace."""
    common = dict(
        release_identity="r" * 64, agent_id="morgan_ecfp4_1024", agent_version="1.0.0",
        output_dim=1024, normalization_id=protocol.CANONICALIZATION_ID,
        row_identity_sha256="rows", artifact_identity=None,
    )
    a1_key = feature_store.matrix_cache_key(endpoint="dili", **common)
    a2_key = feature_store.matrix_cache_key(endpoint=f"{a2.TRACK}:dili", **common)
    assert a1_key != a2_key


def test_a2_runner_rejects_a_shard_from_another_track():
    payload = {
        "shard_schema_version": a2_runner.SHARD_SCHEMA_VERSION, "status": "complete",
        "track": protocol.TRACK_A1, "benchmark_release": "r" * 64,
        "protocol_version": protocol.PROTOCOL_VERSION, "seeds": list(a2.SEEDS),
        "cell": {"endpoint": "e", "representation": "morgan_ecfp4_1024", "probe": "linear"},
        "rows": [], "cell_identity": "x",
    }
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.json"
        a2_runner.write_shard(path, payload)
        assert a2_runner.read_valid_shard(
            path, release_identity="r" * 64, expected_seeds=a2.SEEDS, expected_rows=0) is None


def test_a2_expected_counts_match_the_frozen_protocol():
    manifest = {"endpoints": {f"e{i}": {"included": True, "task_type": "classification"}
                              for i in range(22)}}
    counts = a2_runner.expected_counts(manifest)
    assert counts["cells"] == 22 * 7 * 2
    assert counts["seeds"] == 5
    assert counts["result_rows"] == 22 * 7 * 2 * 5 * 4
    assert counts["total_fits"] == 22 * 7 * 2 * 5 * 5


# --------------------------------------------------------------------------
# partition variability -- the amendment C machinery
# --------------------------------------------------------------------------


def _split(seed, train, validation, test, identity):
    return a2.A2Split(seed=seed, split_id=protocol.split_id(a2.TRACK, seed),
                      train=tuple(train), validation=tuple(validation), test=tuple(test),
                      audit={}, test_set_sha256=identity)


def test_distinct_test_sets_are_detected():
    splits = {i: _split(i, [0], [1], [2 + i], f"h{i}") for i in range(5)}
    result = a2.split_distinctness(splits)
    assert result["distinct_test_sets"] == 5
    assert result["all_test_sets_distinct"] is True
    assert result["mean_pairwise_test_jaccard"] == 0.0


def test_identical_test_sets_are_detected():
    """The solubility_aqsoldb case: five seeds, one test set."""
    splits = {i: _split(i, [0], [1], [2, 3], "same") for i in range(5)}
    result = a2.split_distinctness(splits)
    assert result["distinct_test_sets"] == 1
    assert result["all_test_sets_distinct"] is False
    assert result["mean_pairwise_test_jaccard"] == pytest.approx(1.0)


def test_partially_overlapping_test_sets_give_an_intermediate_jaccard():
    splits = {0: _split(0, [], [], [1, 2, 3, 4], "a"),
              1: _split(1, [], [], [3, 4, 5, 6], "b")}
    result = a2.split_distinctness(splits)
    assert result["mean_pairwise_test_jaccard"] == pytest.approx(2 / 6)


def test_the_variability_alert_threshold_is_frozen_and_sensible():
    assert 0.0 < protocol.A2_PARTITION_VARIABILITY_ALERT < 1.0
    assert "solubility_aqsoldb" in protocol.A2_LOW_VARIABILITY_ENDPOINTS


def test_low_variability_is_recorded_not_fatal():
    """Dropping the endpoint would shrink A2 relative to A1."""
    assert "flagged" in protocol.A2_PARTITION_VARIABILITY_POLICY.lower() or \
        "never" in protocol.A2_PARTITION_VARIABILITY_POLICY.lower()


# --------------------------------------------------------------------------
# split stability
# --------------------------------------------------------------------------


def _stability_rows(per_seed_scores, probe="linear"):
    rows = []
    for seed, scores in per_seed_scores.items():
        for representation, value in scores.items():
            rows.append(_row("e", representation, probe, seed, "auroc", value))
    return rows


def test_split_stability_reports_rank_movement_across_partitions():
    rows = _stability_rows({
        0: {"morgan_ecfp4_1024": 0.9, "maccs_keys_167": 0.5},
        1: {"morgan_ecfp4_1024": 0.4, "maccs_keys_167": 0.8},
        2: {"morgan_ecfp4_1024": 0.9, "maccs_keys_167": 0.5},
        3: {"morgan_ecfp4_1024": 0.4, "maccs_keys_167": 0.8},
        4: {"morgan_ecfp4_1024": 0.9, "maccs_keys_167": 0.5},
    })
    stability = {r["representation"]: r for r in analysis_a2.split_stability(rows)}
    assert stability["morgan_ecfp4_1024"]["rank_range"] == 1.0
    assert stability["morgan_ecfp4_1024"]["rank_sd_across_splits"] > 0


def test_a_representation_that_always_leads_has_zero_rank_spread():
    rows = _stability_rows({
        seed: {"morgan_ecfp4_1024": 0.9, "maccs_keys_167": 0.5} for seed in range(5)
    })
    stability = {r["representation"]: r for r in analysis_a2.split_stability(rows)}
    assert stability["morgan_ecfp4_1024"]["rank_sd_across_splits"] == 0.0
    assert stability["morgan_ecfp4_1024"]["mean_rank"] == 1.0


def test_kendall_w_is_one_when_every_split_agrees():
    rows = _stability_rows({
        seed: {r: 0.5 + i / 100 for i, r in enumerate(REPS)} for seed in range(5)
    })
    result = analysis_a2.kendall_w(rows, probe="linear")[0]
    assert result["kendall_w"] == pytest.approx(1.0)
    assert result["n_splits"] == 5


def test_kendall_w_falls_when_splits_disagree():
    forward = {r: 0.5 + i / 100 for i, r in enumerate(REPS)}
    backward = {r: 0.5 + (len(REPS) - i) / 100 for i, r in enumerate(REPS)}
    rows = _stability_rows({0: forward, 1: backward, 2: forward, 3: backward, 4: forward})
    result = analysis_a2.kendall_w(rows, probe="linear")[0]
    assert result["kendall_w"] < 0.5


def test_stability_is_computed_per_split_but_statistics_stay_per_endpoint():
    """n must never inflate from 22 endpoints to 110 endpoint-splits."""
    rows = _stability_rows({seed: {r: 0.5 + i / 100 for i, r in enumerate(REPS)}
                            for seed in range(5)})
    per_split = analysis_a2.per_seed_ranks(rows)
    assert len(per_split) == 5                      # five split-level views
    scores = analysis.aggregate_seeds(rows)
    assert len({s.endpoint for s in scores}) == 1   # one endpoint for the omnibus
    assert all(s.n_runs == 5 for s in scores)


# --------------------------------------------------------------------------
# A1 versus A2 comparison
# --------------------------------------------------------------------------


def test_classify_calls_a_small_move_reproduced():
    assert analysis_a2.classify(1.9, 2.1, tolerance=0.5) == "reproduced"


def test_classify_calls_a_large_adverse_move_weakened():
    assert analysis_a2.classify(1.9, 3.5, tolerance=0.5) == "weakened"


def test_classify_calls_a_large_favourable_move_strengthened():
    assert analysis_a2.classify(3.5, 1.9, tolerance=0.5) == "strengthened"


def test_classify_respects_metric_direction():
    """For mean rank, lower is better; the flag flips the interpretation."""
    assert analysis_a2.classify(1.0, 3.0, tolerance=0.5, lower_is_better=False) == "strengthened"


def test_leader_picks_the_lowest_mean_rank():
    summary = [{"probe": "linear", "subset": "all", "representation": "a", "mean_rank": 3.0},
               {"probe": "linear", "subset": "all", "representation": "b", "mean_rank": 1.5}]
    assert analysis_a2.leader(summary, probe="linear") == "b"


def test_compare_rankings_reports_position_change():
    a1 = [{"probe": "linear", "subset": "all", "representation": "a", "mean_rank": 1.0,
           "wins": 5, "top3": 10},
          {"probe": "linear", "subset": "all", "representation": "b", "mean_rank": 2.0,
           "wins": 1, "top3": 4}]
    a2_summary = [{"probe": "linear", "subset": "all", "representation": "a", "mean_rank": 2.0,
                   "wins": 1, "top3": 4},
                  {"probe": "linear", "subset": "all", "representation": "b", "mean_rank": 1.0,
                   "wins": 5, "top3": 10}]
    rows = {r["representation"]: r for r in
            analysis_a2.compare_rankings(a1, a2_summary, probe="linear")}
    assert rows["a"]["a1_position"] == 1 and rows["a"]["a2_position"] == 2
    assert rows["a"]["position_change"] == -1
    assert rows["b"]["position_change"] == 1


def test_reproduced_contrasts_only_considers_a1_significant_pairs():
    a1 = [{"probe": "linear", "task_type": "all", "a": "x", "b": "y", "p_holm": 0.01,
           "effect_size_rank_biserial": -0.8, "significant_after_holm": True},
          {"probe": "linear", "task_type": "all", "a": "x", "b": "z", "p_holm": 0.9,
           "effect_size_rank_biserial": -0.1, "significant_after_holm": False}]
    a2_tests = [{"probe": "linear", "task_type": "all", "a": "x", "b": "y", "p_holm": 0.02,
                 "effect_size_rank_biserial": -0.7, "significant_after_holm": True}]
    out = analysis_a2.reproduced_contrasts(a1, a2_tests)
    assert len(out) == 1
    assert out[0]["reproduced"] is True
    assert out[0]["effect_direction_preserved"] is True


def test_a_contrast_that_loses_significance_is_not_reproduced():
    a1 = [{"probe": "linear", "task_type": "all", "a": "x", "b": "y", "p_holm": 0.01,
           "effect_size_rank_biserial": -0.8, "significant_after_holm": True}]
    a2_tests = [{"probe": "linear", "task_type": "all", "a": "x", "b": "y", "p_holm": 0.4,
                 "effect_size_rank_biserial": -0.3, "significant_after_holm": False}]
    out = analysis_a2.reproduced_contrasts(a1, a2_tests)
    assert out[0]["reproduced"] is False
    assert out[0]["a2_tested"] is True


def test_an_untested_contrast_is_distinguished_from_a_failed_one():
    """A2's omnibus may not reject, which is not the same as a null result."""
    a1 = [{"probe": "linear", "task_type": "regression", "a": "x", "b": "y", "p_holm": 0.01,
           "effect_size_rank_biserial": -0.8, "significant_after_holm": True}]
    out = analysis_a2.reproduced_contrasts(a1, [])
    assert out[0]["a2_tested"] is False
    assert out[0]["reproduced"] is False


# --------------------------------------------------------------------------
# protocol invariants A2 must not have changed
# --------------------------------------------------------------------------


def test_a2_did_not_widen_the_hyperparameter_grid():
    from molfusion_backend.benchmark import pipelines

    assert len(pipelines.hyperparameter_grid("linear", "classification")) == 4
    assert len(pipelines.hyperparameter_grid("nonlinear", "regression")) == 4


def test_a2_uses_the_same_seven_representations():
    assert len(protocol.TRACK_A_REPRESENTATIONS) == 7


def test_a2_applies_the_full_cleaning_policy():
    assert protocol.TRACK_A2_CLEANING.startswith("full Phase 6A policy")


def test_a2_fractions_are_the_frozen_ones():
    assert protocol.TRAIN_FRACTION == 0.70
    assert protocol.VALIDATION_FRACTION == 0.10
    assert protocol.TEST_FRACTION == 0.20


def test_a2_modules_do_not_import_pytdc_or_track_a1():
    from pathlib import Path

    for module in (a2, a2_runner, analysis_a2):
        source = Path(module.__file__).read_text("utf-8")
        assert "import tdc" not in source and "from tdc" not in source
        assert "import a1" not in source
        assert "benchmark.a1" not in source
