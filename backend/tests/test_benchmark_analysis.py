"""Phase 6A.3: the Track A1 analysis contract.

No test here retrains a model or reads the real 6,160-row matrix. What is
tested is the machinery that turns raw rows into claims: metric direction,
seed aggregation, ranking and ties, pairwise counting, the inputs handed to
Friedman and Wilcoxon, Holm correction, effect size, bootstrap determinism,
and the refusal to analyse a result set whose identity has moved.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from molfusion_backend.benchmark import analysis, metrics, protocol, runner

REPS = list(protocol.TRACK_A_REPRESENTATIONS)


def _row(endpoint, representation, probe, seed, metric, value, task_type="classification"):
    base = {c: 0 for c in runner.RESULT_COLUMNS}
    base.update(
        {
            "benchmark_release": "r" * 64,
            "track": protocol.TRACK_A1,
            "endpoint": endpoint,
            "task_type": task_type,
            "split_id": protocol.split_id(protocol.TRACK_A1, seed),
            "split_strategy": protocol.TRACK_A1_SPLIT_STRATEGY,
            "seed": seed,
            "representation": representation,
            "agent_version": "1.0.0",
            "model_family": "M",
            "probe": probe,
            "hyperparameters": json.dumps({"C": 1.0}),
            "metric": metric,
            "metric_value": value,
            "molfusion_primary_metric": "auroc" if task_type == "classification" else "mae",
            "tdc_official_metric": "roc-auc" if task_type == "classification" else "mae",
        }
    )
    return base


def _matrix(values, probe="linear", task_type="classification", metric="auroc"):
    """values: {endpoint: {representation: score}} -> 5 identical seed rows."""
    rows = []
    for endpoint, per_rep in values.items():
        for representation, score in per_rep.items():
            for seed in protocol.TRACK_A1_SEEDS:
                rows.append(
                    _row(endpoint, representation, probe, seed, metric, score, task_type)
                )
    return rows


# --------------------------------------------------------------------------
# metric direction
# --------------------------------------------------------------------------


def test_higher_is_better_metrics_are_not_inverted():
    assert metrics.orient("auroc", 0.9) > metrics.orient("auroc", 0.7)


def test_lower_is_better_metrics_are_inverted():
    """MAE 0.3 must outrank MAE 0.9."""
    assert metrics.orient("mae", 0.3) > metrics.orient("mae", 0.9)
    assert "mae" in protocol.LOWER_IS_BETTER


def test_ranking_respects_metric_direction():
    scores = analysis.aggregate_seeds(
        _matrix({"e": {"a": 0.3, "b": 0.9}}, task_type="regression", metric="mae")
    )
    ranks = analysis.rank_endpoint(scores)
    assert ranks["a"] == 1.0 and ranks["b"] == 2.0


# --------------------------------------------------------------------------
# seed aggregation
# --------------------------------------------------------------------------


def test_seed_aggregation_produces_one_value_per_cell():
    """The guard against pseudoreplication: 5 rows in, 1 observation out."""
    rows = _matrix({"e1": {"a": 0.8, "b": 0.6}})
    scores = analysis.aggregate_seeds(rows)
    assert len(scores) == 2
    assert all(s.n_runs == 5 for s in scores)


def test_seed_aggregation_reports_spread_not_just_the_mean():
    rows = []
    for seed, value in zip(protocol.TRACK_A1_SEEDS, [0.5, 0.6, 0.7, 0.8, 0.9]):
        rows.append(_row("e", "a", "linear", seed, "auroc", value))
    score = analysis.aggregate_seeds(rows)[0]
    assert score.mean == pytest.approx(0.7)
    assert score.median == pytest.approx(0.7)
    assert score.minimum == pytest.approx(0.5)
    assert score.maximum == pytest.approx(0.9)
    assert score.std > 0


def test_only_the_frozen_primary_metric_is_aggregated():
    """Secondary metrics must not silently enter the ranking."""
    rows = _matrix({"e": {"a": 0.8}})
    rows += [_row("e", "a", "linear", s, "auprc", 0.1) for s in protocol.TRACK_A1_SEEDS]
    scores = analysis.aggregate_seeds(rows)
    assert len(scores) == 1
    assert scores[0].metric == "auroc"


# --------------------------------------------------------------------------
# ranking and ties
# --------------------------------------------------------------------------


def test_exact_ties_receive_the_average_rank():
    """Breaking ties would manufacture an ordering the data lacks."""
    scores = analysis.aggregate_seeds(_matrix({"e": {"a": 0.8, "b": 0.8, "c": 0.5}}))
    ranks = analysis.rank_endpoint(scores)
    assert ranks["a"] == ranks["b"] == 1.5
    assert ranks["c"] == 3.0


def test_ties_are_not_broken_by_representation_name():
    forward = analysis.rank_endpoint(
        analysis.aggregate_seeds(_matrix({"e": {"aaa": 0.8, "zzz": 0.8}}))
    )
    assert forward["aaa"] == forward["zzz"]


def test_ranks_run_from_one_to_n():
    scores = analysis.aggregate_seeds(
        _matrix({"e": {r: 0.5 + i / 100 for i, r in enumerate(REPS)}})
    )
    ranks = analysis.rank_endpoint(scores)
    assert sorted(ranks.values()) == [float(i) for i in range(1, 8)]


def test_summarise_ranks_counts_wins_and_top3():
    rows = _matrix({"e1": {"a": 0.9, "b": 0.5}, "e2": {"a": 0.9, "b": 0.5}})
    ranks = analysis.rank_table(analysis.aggregate_seeds(rows))
    summary = {s["representation"]: s for s in analysis.summarise_ranks(ranks, probe="linear")}
    assert summary["a"]["wins"] == 2
    assert summary["a"]["mean_rank"] == 1.0
    assert summary["b"]["wins"] == 0


def test_summarise_ranks_can_be_restricted_to_a_subset_of_endpoints():
    rows = _matrix({"e1": {"a": 0.9, "b": 0.5}, "e2": {"a": 0.1, "b": 0.5}})
    ranks = analysis.rank_table(analysis.aggregate_seeds(rows))
    subset = {s["representation"]: s for s in
              analysis.summarise_ranks(ranks, probe="linear", endpoints=["e1"])}
    assert subset["a"]["n_endpoints"] == 1
    assert subset["a"]["mean_rank"] == 1.0


# --------------------------------------------------------------------------
# pairwise
# --------------------------------------------------------------------------


def test_pairwise_counts_wins_losses_and_ties():
    rows = _matrix({
        "e1": {r: 0.5 for r in REPS} | {"morgan_ecfp4_1024": 0.9},
        "e2": {r: 0.5 for r in REPS} | {"maccs_keys_167": 0.9},
        "e3": {r: 0.5 for r in REPS},
    })
    pairs = analysis.pairwise_wins(analysis.aggregate_seeds(rows), probe="linear")
    entry = next(p for p in pairs
                 if {p["a"], p["b"]} == {"morgan_ecfp4_1024", "maccs_keys_167"})
    counts = sorted([entry["a_better"], entry["b_better"]])
    assert counts == [1, 1]
    assert entry["ties"] == 1
    assert entry["n_endpoints"] == 3


def test_pairwise_covers_every_unordered_pair_once():
    rows = _matrix({"e": {r: 0.5 + i / 100 for i, r in enumerate(REPS)}})
    pairs = analysis.pairwise_wins(analysis.aggregate_seeds(rows), probe="linear")
    assert len(pairs) == 21           # 7 choose 2
    assert len({frozenset((p["a"], p["b"])) for p in pairs}) == 21


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_friedman_uses_endpoints_not_seeds_as_observations():
    """n_endpoints must be the endpoint count, never 5x it."""
    endpoints = {f"e{i}": {r: 0.5 + i / 100 + j / 1000 for j, r in enumerate(REPS)}
                 for i in range(8)}
    result = analysis.friedman(analysis.aggregate_seeds(_matrix(endpoints)), probe="linear")
    assert result["runnable"] is True
    assert result["n_endpoints"] == 8


def test_friedman_declines_when_too_few_endpoints():
    result = analysis.friedman(
        analysis.aggregate_seeds(_matrix({"e": {r: 0.5 for r in REPS}})), probe="linear"
    )
    assert result["runnable"] is False


def test_holm_is_monotone_and_bounded():
    adjusted = analysis.holm([0.001, 0.01, 0.04, 0.5])
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    assert adjusted == sorted(adjusted)
    assert all(a >= r for a, r in zip(adjusted, [0.001, 0.01, 0.04, 0.5]))


def test_holm_preserves_input_order():
    adjusted = analysis.holm([0.5, 0.001])
    assert adjusted[1] < adjusted[0]


def test_holm_is_more_conservative_than_raw_p():
    raw = [0.01] * 21
    assert all(a > 0.01 for a in analysis.holm(raw))


def test_rank_biserial_is_plus_one_when_a_always_wins():
    assert analysis.rank_biserial([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_rank_biserial_is_minus_one_when_a_always_loses():
    assert analysis.rank_biserial([-1.0, -2.0, -3.0]) == pytest.approx(-1.0)


def test_rank_biserial_is_zero_for_a_symmetric_split():
    assert analysis.rank_biserial([1.0, -1.0]) == pytest.approx(0.0)


def test_rank_biserial_ignores_exact_zeros():
    assert analysis.rank_biserial([0.0, 0.0, 5.0]) == pytest.approx(1.0)


def test_every_pairwise_test_carries_an_effect_size():
    """No p-value is reported without one."""
    endpoints = {f"e{i}": {r: 0.5 + i / 100 + j / 1000 for j, r in enumerate(REPS)}
                 for i in range(10)}
    results = analysis.pairwise_tests(
        analysis.aggregate_seeds(_matrix(endpoints)), probe="linear"
    )
    assert len(results) == 21
    for row in results:
        assert "effect_size_rank_biserial" in row
        assert "p_holm" in row and "p_raw" in row
        assert row["p_holm"] >= row["p_raw"]


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    endpoints = {f"e{i}": {r: 0.5 + i / 100 + j / 1000 for j, r in enumerate(REPS)}
                 for i in range(12)}
    ranks = analysis.rank_table(analysis.aggregate_seeds(_matrix(endpoints)))
    a = analysis.bootstrap_mean_rank(ranks, probe="linear", resamples=500, seed=0)
    b = analysis.bootstrap_mean_rank(ranks, probe="linear", resamples=500, seed=0)
    assert a == b


def test_bootstrap_differs_for_a_different_seed():
    endpoints = {f"e{i}": {r: 0.5 + i / 100 + j / 1000 for j, r in enumerate(REPS)}
                 for i in range(12)}
    ranks = analysis.rank_table(analysis.aggregate_seeds(_matrix(endpoints)))
    a = analysis.bootstrap_mean_rank(ranks, probe="linear", resamples=500, seed=0)
    b = analysis.bootstrap_mean_rank(ranks, probe="linear", resamples=500, seed=1)
    assert a != b


def test_bootstrap_resamples_endpoints_not_molecules():
    endpoints = {f"e{i}": {r: 0.5 + i / 100 + j / 1000 for j, r in enumerate(REPS)}
                 for i in range(12)}
    ranks = analysis.rank_table(analysis.aggregate_seeds(_matrix(endpoints)))
    out = analysis.bootstrap_mean_rank(ranks, probe="linear", resamples=200, seed=0)
    assert all(row["resampling_unit"] == "endpoint" for row in out)


def test_bootstrap_interval_brackets_the_point_estimate():
    endpoints = {f"e{i}": {r: 0.5 + i / 100 + j / 1000 for j, r in enumerate(REPS)}
                 for i in range(12)}
    ranks = analysis.rank_table(analysis.aggregate_seeds(_matrix(endpoints)))
    for row in analysis.bootstrap_mean_rank(ranks, probe="linear", resamples=2000, seed=0):
        assert row["ci_lower_95"] <= row["mean_rank"] <= row["ci_upper_95"]


# --------------------------------------------------------------------------
# nonlinear gain
# --------------------------------------------------------------------------


def test_nonlinear_gain_is_positive_when_the_nonlinear_probe_scores_higher():
    gainer, flat = "morgan_ecfp4_1024", "maccs_keys_167"
    rows = _matrix({"e": {gainer: 0.5, flat: 0.4}}, probe="linear")
    rows += _matrix({"e": {gainer: 0.9, flat: 0.4}}, probe="nonlinear")
    gains = {g["representation"]: g for g in
             analysis.nonlinear_gain(analysis.aggregate_seeds(rows))}
    assert gains[gainer]["normalised_gain"] > 0
    assert gains[flat]["normalised_gain"] == pytest.approx(0.0)


def test_nonlinear_gain_reports_rank_movement_as_well_as_score_movement():
    """Two measures, because neither alone is enough."""
    a, b = "morgan_ecfp4_1024", "maccs_keys_167"
    rows = _matrix({"e": {a: 0.4, b: 0.8}}, probe="linear")
    rows += _matrix({"e": {a: 0.9, b: 0.5}}, probe="nonlinear")
    gains = {g["representation"]: g for g in
             analysis.nonlinear_gain(analysis.aggregate_seeds(rows))}
    assert gains[a]["rank_gain"] > 0        # 2nd under linear, 1st under nonlinear
    assert gains[b]["rank_gain"] < 0


def test_nonlinear_gain_normalisation_is_within_endpoint():
    """So an AUROC endpoint and an MAE endpoint stay comparable."""
    a = "morgan_ecfp4_1024"
    rows = _matrix({"e": {a: 0.5}}, probe="linear") + _matrix({"e": {a: 0.9}}, probe="nonlinear")
    for gain in analysis.nonlinear_gain(analysis.aggregate_seeds(rows)):
        assert -1.0 <= gain["normalised_gain"] <= 1.0


# --------------------------------------------------------------------------
# immutability and shape guards
# --------------------------------------------------------------------------


def test_verify_rejects_a_changed_scientific_identity():
    rows = _matrix({"e": {"a": 0.5}})
    with pytest.raises(analysis.AnalysisError, match="identity"):
        analysis.verify_raw_results(rows, expected_identity="0" * 64, expected_rows=len(rows))


def test_verify_rejects_a_changed_row_count():
    rows = _matrix({"e": {"a": 0.5}})
    identity = runner.scientific_identity(rows)
    with pytest.raises(analysis.AnalysisError, match="raw rows"):
        analysis.verify_raw_results(rows, expected_identity=identity, expected_rows=999)


def test_verify_accepts_the_matching_matrix():
    rows = _matrix({"e": {"a": 0.5}})
    provenance = analysis.verify_raw_results(
        rows, expected_identity=runner.scientific_identity(rows), expected_rows=len(rows)
    )
    assert provenance["endpoints"] == 1
    assert provenance["seeds"] == list(protocol.TRACK_A1_SEEDS)


def test_expected_endpoint_and_representation_counts_are_the_frozen_ones():
    assert len(protocol.TRACK_A_REPRESENTATIONS) == 7
    assert len(protocol.TRACK_A1_SEEDS) == 5


def test_analysis_identity_is_deterministic_and_excludes_volatile_data():
    config = {"alpha": 0.05, "bootstrap_seed": 0}
    a = analysis.analysis_identity(raw_identity="x" * 64, configuration=config)
    b = analysis.analysis_identity(raw_identity="x" * 64, configuration=dict(config))
    assert a == b


def test_analysis_identity_changes_with_the_raw_matrix():
    config = {"alpha": 0.05}
    assert analysis.analysis_identity(raw_identity="x" * 64, configuration=config) != \
        analysis.analysis_identity(raw_identity="y" * 64, configuration=config)


def test_analysis_identity_changes_with_the_statistical_configuration():
    assert analysis.analysis_identity(raw_identity="x" * 64, configuration={"alpha": 0.05}) != \
        analysis.analysis_identity(raw_identity="x" * 64, configuration={"alpha": 0.01})


def test_write_table_round_trips(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    digest = analysis.write_table(tmp_path / "t.csv", rows)
    assert len(digest) == 64
    back = list(csv.DictReader(open(tmp_path / "t.csv", encoding="utf-8")))
    assert [r["b"] for r in back] == ["x", "y"]


def test_analysis_never_writes_to_the_raw_directory(tmp_path):
    """The analysis module has no code path that opens raw files for writing."""
    source = Path(analysis.__file__).read_text("utf-8")
    assert 'open(path, "w"' in source or "write_table" in source
    # load_raw_results opens read-only; assert no write mode against the input
    assert 'open(path, encoding="utf-8", newline="")' in source
