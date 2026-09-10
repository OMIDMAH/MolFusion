"""Phase 6B: the publication package must not overstate the results.

The failure this suite guards against is not a wrong number -- every number
comes from a frozen 6A.3/6A.4 table -- but a true number stated more
strongly than it was earned. So most of these tests are about claim
discipline: that a claim without a recorded limitation cannot exist, that
an exploratory observation cannot reach the abstract, that separation is
checked rather than assumed, and that the two tracks stay distinguishable.

One of them exists because it already caught something. An earlier draft of
C11 asserted that the same two representations held the bottom two
positions under both probes in both tracks; the tables say otherwise, since
erg_reduced_graph_315 ranks fifth under the A1 nonlinear probe. The claim is
now derived from the ranking table, and
test_bottom_rank_claim_is_derived_not_asserted keeps it that way.
"""

import csv
import json
from pathlib import Path

import pytest

from molfusion_backend.benchmark import (
    protocol,
    publication,
    publication_claims,
    publication_figures,
)

PACKAGE = Path("benchmark_runs/publication")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _rows(path: Path):
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


@pytest.fixture(scope="module")
def package():
    if not (PACKAGE / "publication_report.json").exists():
        pytest.skip("publication package not generated in this checkout")
    return json.loads((PACKAGE / "publication_report.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def registry(package):
    return package["claim_registry"]


BOOTSTRAP = [
    {"probe": "nonlinear", "representation": "a", "mean_rank": 1.8,
     "ci_lower_95": 1.3, "ci_upper_95": 2.3},
    {"probe": "nonlinear", "representation": "b", "mean_rank": 3.0,
     "ci_lower_95": 2.4, "ci_upper_95": 3.8},
    {"probe": "nonlinear", "representation": "c", "mean_rank": 4.2,
     "ci_lower_95": 3.7, "ci_upper_95": 4.7},
]


# ---------------------------------------------------------------------------
# CI separation is checked, never assumed
# ---------------------------------------------------------------------------


def test_separation_is_detected_when_intervals_are_clear():
    result = publication.ci_separation(BOOTSTRAP, probe="nonlinear")
    assert result["leader"] == "a"
    assert result["separated_from_all"] is True
    assert result["overlapping_competitors"] == []


def test_separation_is_denied_when_an_interval_overlaps():
    overlapping = [dict(BOOTSTRAP[0]), dict(BOOTSTRAP[1], ci_lower_95=2.1), BOOTSTRAP[2]]
    result = publication.ci_separation(overlapping, probe="nonlinear")
    assert result["separated_from_all"] is False
    assert result["overlapping_competitors"] == ["b"]


def test_separation_reports_every_competitor():
    result = publication.ci_separation(BOOTSTRAP, probe="nonlinear")
    assert result["n_competitors"] == 2
    assert len(result["comparisons"]) == 2


def test_the_real_nonlinear_leader_is_actually_separated(package):
    """The headline claim's precondition, verified against the real data."""
    assert package["ci_separation"]["a2"]["nonlinear"]["separated_from_all"] is True
    assert package["ci_separation"]["a1"]["nonlinear"]["separated_from_all"] is True


def test_the_real_linear_leader_is_not_separated(package):
    """C3 depends on this being false in both tracks."""
    assert package["ci_separation"]["a2"]["linear"]["separated_from_all"] is False
    assert package["ci_separation"]["a1"]["linear"]["separated_from_all"] is False


# ---------------------------------------------------------------------------
# claim registry schema and discipline
# ---------------------------------------------------------------------------


def test_the_registry_validates(registry):
    assert publication_claims.validate_registry(registry) == []


def test_every_claim_has_every_required_field(registry):
    for entry in registry:
        for column in publication_claims.REGISTRY_COLUMNS:
            assert column in entry, f"{entry['claim_id']} missing {column}"


def test_claim_ids_are_unique(registry):
    ids = [e["claim_id"] for e in registry]
    assert len(ids) == len(set(ids))


def test_every_claim_type_is_known(registry):
    for entry in registry:
        assert entry["claim_type"] in publication.CLAIM_TYPES


def test_every_claim_records_a_limitation_and_a_prohibition(registry):
    for entry in registry:
        assert entry["limitations"], f"{entry['claim_id']} has no limitation"
        assert entry["prohibited_wording"], f"{entry['claim_id']} has no prohibition"


def test_exploratory_claims_cannot_reach_the_abstract_or_conclusion(registry):
    for entry in registry:
        if entry["claim_type"] == "EXPLORATORY":
            assert entry["allowed_in_abstract"] is False
            assert entry["allowed_in_conclusion"] is False


def test_a_missing_limitation_is_rejected():
    bad = [{
        "claim_id": "X", "claim_type": "PRIMARY", "claim_text": "t",
        "supported_by": "s", "statistical_basis": "b", "limitations": "",
        "allowed_in_abstract": True, "allowed_in_conclusion": True,
        "confidence": "high", "recommended_wording": "w",
        "prohibited_wording": "p",
    }]
    assert any("no limitation" in p for p in publication_claims.validate_registry(bad))


def test_an_exploratory_claim_in_the_abstract_is_rejected():
    bad = [{
        "claim_id": "X", "claim_type": "EXPLORATORY", "claim_text": "t",
        "supported_by": "s", "statistical_basis": "b", "limitations": "l",
        "allowed_in_abstract": True, "allowed_in_conclusion": False,
        "confidence": "low", "recommended_wording": "w",
        "prohibited_wording": "p",
    }]
    problems = publication_claims.validate_registry(bad)
    assert any("abstract" in p for p in problems)


def test_a_registry_without_a_primary_claim_is_rejected():
    assert any("no PRIMARY" in p for p in publication_claims.validate_registry([]))


def test_there_is_exactly_one_primary_claim(registry):
    primary = [e for e in registry if e["claim_type"] == "PRIMARY"]
    assert len(primary) == 1
    assert "physicochemical" in primary[0]["claim_text"]


# ---------------------------------------------------------------------------
# terminology
# ---------------------------------------------------------------------------


def test_no_claim_calls_the_competitor_set_structural_fingerprints(registry):
    """The Track A competitors are a mixed set; one label would be inaccurate."""
    for entry in registry:
        assert publication.PROHIBITED_COLLECTIVE_TERM not in entry["claim_text"]
        assert publication.PROHIBITED_COLLECTIVE_TERM not in entry["recommended_wording"]


def test_the_prohibited_collective_term_is_named_as_prohibited(registry):
    primary = next(e for e in registry if e["claim_type"] == "PRIMARY")
    assert publication.PROHIBITED_COLLECTIVE_TERM in primary["prohibited_wording"]


def test_every_representation_has_an_accurate_category():
    assert set(publication.REPRESENTATION_CATEGORY) == set(protocol.TRACK_A_REPRESENTATIONS)
    assert len(set(publication.REPRESENTATION_CATEGORY.values())) > 1


def test_the_exposure_claim_never_says_leakage(registry):
    exposure = next(e for e in registry if e["claim_id"] == "C9")
    assert "leakage" not in exposure["claim_text"].lower()
    assert "unsupervised" in exposure["claim_text"]
    assert "leakage" in exposure["prohibited_wording"]


# ---------------------------------------------------------------------------
# the overclaim this suite already caught
# ---------------------------------------------------------------------------


def test_bottom_rank_claim_is_derived_not_asserted(registry):
    """C11 must reflect the tables, which do not support a uniform pair."""
    claim = next(e for e in registry if e["claim_id"] == "C11")
    assert "rdkit_fragment_descriptors" in claim["claim_text"]
    assert "4/4" in claim["statistical_basis"]
    assert "3/4" in claim["statistical_basis"]
    assert "fifth" in claim["limitations"]


def test_bottom_rank_counts_match_the_ranking_table():
    table = _rows(PACKAGE / "tables" / "table3_a2_robustness.csv")
    if not table:
        pytest.skip("package not generated")
    counts: dict[str, int] = {}
    for probe in protocol.PROBES:
        subset = [r for r in table if r["probe"] == probe]
        for key in ("a1_position", "a2_position"):
            for row in sorted(subset, key=lambda r: int(r[key]))[-2:]:
                counts[row["representation"]] = counts.get(row["representation"], 0) + 1
    assert counts.get("rdkit_fragment_descriptors") == 4
    assert counts.get("erg_reduced_graph_315") == 3


# ---------------------------------------------------------------------------
# stability flagging
# ---------------------------------------------------------------------------


def test_low_stability_endpoints_are_flagged_not_recommended():
    rows = publication.stability_table([
        {"endpoint": "herg", "probe": "linear", "kendall_w": "0.157"},
        {"endpoint": "herg", "probe": "nonlinear", "kendall_w": "0.514"},
        {"endpoint": "ames", "probe": "linear", "kendall_w": "0.989"},
        {"endpoint": "ames", "probe": "nonlinear", "kendall_w": "0.846"},
    ])
    herg = next(r for r in rows if r["endpoint"] == "herg")
    ames = next(r for r in rows if r["endpoint"] == "ames")
    assert herg["endpoint_stability_flag"] == "LOW"
    assert herg["per_endpoint_interpretation"] == publication.STABILITY_NOT_RECOMMENDED
    assert ames["endpoint_stability_flag"] == "OK"
    assert ames["per_endpoint_interpretation"] == publication.STABILITY_RECOMMENDED


def test_stability_uses_the_weaker_probe():
    """One strong probe must not mask an unstable one."""
    rows = publication.stability_table([
        {"endpoint": "x", "probe": "linear", "kendall_w": "0.95"},
        {"endpoint": "x", "probe": "nonlinear", "kendall_w": "0.20"},
    ])
    assert rows[0]["kendall_w_min"] == 0.20
    assert rows[0]["rule_low_stability"] is True


def test_a_rule_pre_registration_disagreement_is_surfaced():
    rows = publication.stability_table([
        {"endpoint": "vdss_lombardo", "probe": "linear", "kendall_w": "0.903"},
        {"endpoint": "vdss_lombardo", "probe": "nonlinear", "kendall_w": "0.343"},
    ])
    assert publication.stability_disagreements(rows) == ["vdss_lombardo"]
    assert rows[0]["endpoint_stability_flag"] == "BORDERLINE"


def test_all_six_pre_registered_endpoints_are_flagged_in_the_package():
    rows = _rows(PACKAGE / "tables" / "table6_endpoint_stability.csv")
    if not rows:
        pytest.skip("package not generated")
    flagged = {r["endpoint"] for r in rows if r["endpoint_stability_flag"] == "LOW"}
    assert flagged == set(publication.PRE_REGISTERED_LOW_STABILITY)


def test_low_stability_endpoints_remain_in_the_benchmark():
    """Flagged, never dropped: all 22 endpoints stay in cross-endpoint work."""
    rows = _rows(PACKAGE / "tables" / "table6_endpoint_stability.csv")
    if not rows:
        pytest.skip("package not generated")
    assert len(rows) == 22


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("table1_representation_characteristics.csv", 7),
    ("table2_a1_primary_performance.csv", 7),
    ("table3_a2_robustness.csv", 14),
    ("table5_computational_cost.csv", 7),
    ("table6_endpoint_stability.csv", 22),
])
def test_table_row_completeness(name, expected):
    rows = _rows(PACKAGE / "tables" / name)
    if not rows:
        pytest.skip("package not generated")
    assert len(rows) == expected


def test_every_table_lists_all_seven_representations():
    for name in ("table1_representation_characteristics.csv",
                 "table2_a1_primary_performance.csv",
                 "table5_computational_cost.csv"):
        rows = _rows(PACKAGE / "tables" / name)
        if not rows:
            pytest.skip("package not generated")
        assert {r["representation"] for r in rows} == set(protocol.TRACK_A_REPRESENTATIONS)


def test_representation_order_is_the_recorded_rule(package):
    order = package["representation_order"]
    assert order[0] == "rdkit_physchem_descriptors"
    assert len(order) == 7
    assert "nonlinear mean rank" in package["representation_order_rule"]


def test_table_order_matches_the_declared_order(package):
    rows = _rows(PACKAGE / "tables" / "table5_computational_cost.csv")
    if not rows:
        pytest.skip("package not generated")
    assert [r["representation"] for r in rows] == package["representation_order"]


def test_key_contrasts_are_a_subset_not_a_dump():
    key = _rows(PACKAGE / "tables" / "table4_key_statistical_contrasts.csv")
    full = _rows(PACKAGE / "supplementary" / "all_pairwise_contrasts_a2.csv")
    if not key or not full:
        pytest.skip("package not generated")
    assert len(full) == 126
    assert 0 < len(key) < 30, "the main table must not dump all 126 contrasts"


def test_every_key_contrast_carries_a_corrected_p_and_an_effect_size():
    rows = _rows(PACKAGE / "tables" / "table4_key_statistical_contrasts.csv")
    if not rows:
        pytest.skip("package not generated")
    for row in rows:
        assert row["p_holm"], "corrected p-value required"
        assert row["rank_biserial"], "effect size required alongside significance"
        assert row["direction"]


def test_the_cost_table_keeps_cost_and_rank_on_separate_axes():
    rows = _rows(PACKAGE / "tables" / "table5_computational_cost.csv")
    if not rows:
        pytest.skip("package not generated")
    columns = set(rows[0])
    assert "nonlinear_mean_rank" in columns
    assert "nonlinear_model_seconds" in columns
    for forbidden in ("efficiency", "score", "cost_adjusted"):
        assert not any(forbidden in c for c in columns), (
            "cost and performance must not be combined into one scalar")


def test_the_22_and_19_endpoint_subsets_are_both_reported():
    rows = _rows(PACKAGE / "tables" / "table7_22_vs_19_endpoint_subset.csv")
    if not rows:
        pytest.skip("package not generated")
    subsets = {r["subset"] for r in rows}
    assert subsets == {"all", "repartitioned_only"}
    counts = {r["subset"]: int(r["n_endpoints"]) for r in rows}
    assert counts["all"] == 22
    assert counts["repartitioned_only"] == 19


# ---------------------------------------------------------------------------
# A1 and A2 must stay separable
# ---------------------------------------------------------------------------


def test_the_robustness_table_keeps_the_tracks_in_separate_columns():
    rows = _rows(PACKAGE / "tables" / "table3_a2_robustness.csv")
    if not rows:
        pytest.skip("package not generated")
    columns = set(rows[0])
    assert {"a1_mean_rank", "a2_mean_rank"} <= columns
    assert not any(c.startswith("pooled") or c.startswith("combined") for c in columns)


def test_the_two_tracks_have_distinct_identities(package):
    inputs = package["inputs"]
    assert inputs["a1_raw_identity"] != inputs["a2_raw_identity"]
    assert inputs["a1_raw_identity"] == publication.A1_RAW_IDENTITY
    assert inputs["a2_raw_identity"] == publication.A2_RAW_IDENTITY


def test_a1_is_recorded_as_primary_and_a2_as_supplementary(registry):
    text = " ".join(e["claim_text"] + e["statistical_basis"] for e in registry)
    assert "A1" in text and "A2" in text
    robustness = [e for e in registry if e["claim_type"] == "ROBUSTNESS"]
    assert robustness, "A2 must appear as robustness evidence, not as a rival benchmark"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", [1, 2, 3, 4, 5])
def test_every_figure_has_an_exported_data_table(index):
    path = PACKAGE / "figure_data" / f"figure_{index:02d}_data.csv"
    if not PACKAGE.exists():
        pytest.skip("package not generated")
    assert path.exists(), f"figure {index} has no regenerable data table"
    assert _rows(path), f"figure {index} data table is empty"


@pytest.mark.parametrize("name", [
    "figure_01_rank_heatmap.svg", "figure_02_mean_rank_ci.svg",
    "figure_03_rank_robustness.svg", "figure_04_rank_vs_cost.svg",
    "figure_05_endpoint_stability.svg",
])
def test_every_figure_renders_valid_svg(name):
    path = PACKAGE / "figures" / name
    if not PACKAGE.exists():
        pytest.skip("package not generated")
    body = path.read_text(encoding="utf-8")
    assert body.startswith("<svg")
    assert body.rstrip().endswith("</svg>")
    assert "<title>" in body


def test_figure_rendering_is_deterministic():
    """Same input, same bytes -- no timestamps, no library-version drift."""
    first = publication_figures.figure_mean_rank_ci(BOOTSTRAP, probes=["nonlinear"])
    second = publication_figures.figure_mean_rank_ci(BOOTSTRAP, probes=["nonlinear"])
    assert first == second


def test_figure_two_shares_one_scale_across_panels():
    """Small linear differences must not be visually exaggerated."""
    source = Path(publication_figures.__file__).read_text(encoding="utf-8")
    assert "lo, hi = 1.0, 7.0" in source, (
        "both panels must map rank 1-7 onto the same axis")


def test_the_heatmap_marks_low_stability_endpoints():
    body = (PACKAGE / "figures" / "figure_01_rank_heatmap.svg")
    if not body.exists():
        pytest.skip("package not generated")
    text = body.read_text(encoding="utf-8")
    assert "herg *" in text or "herg&#42;" in text or "herg *" in text.replace("&#42;", "*")


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_publication_identity_is_deterministic():
    args = dict(a1_identity="a" * 64, a2_identity="b" * 64,
                a2_analysis_identity="c" * 64,
                table_checksums={"t/one.csv": "1" * 64, "t/two.csv": "2" * 64})
    assert publication.publication_identity(**args) == publication.publication_identity(**args)


def test_publication_identity_is_order_independent():
    base = dict(a1_identity="a" * 64, a2_identity="b" * 64,
                a2_analysis_identity="c" * 64)
    first = publication.publication_identity(
        **base, table_checksums={"a.csv": "1" * 64, "b.csv": "2" * 64})
    second = publication.publication_identity(
        **base, table_checksums={"b.csv": "2" * 64, "a.csv": "1" * 64})
    assert first == second


def test_publication_identity_changes_when_a_table_changes():
    base = dict(a1_identity="a" * 64, a2_identity="b" * 64,
                a2_analysis_identity="c" * 64)
    first = publication.publication_identity(**base, table_checksums={"a.csv": "1" * 64})
    second = publication.publication_identity(**base, table_checksums={"a.csv": "9" * 64})
    assert first != second


def test_publication_identity_changes_when_an_input_changes():
    common = dict(a2_analysis_identity="c" * 64, table_checksums={"a.csv": "1" * 64})
    first = publication.publication_identity(
        a1_identity="a" * 64, a2_identity="b" * 64, **common)
    second = publication.publication_identity(
        a1_identity="a" * 64, a2_identity="d" * 64, **common)
    assert first != second


def test_table_checksum_is_column_ordered():
    rows = [{"a": 1, "b": 2}]
    assert (publication.table_checksum(rows, ("a", "b"))
            != publication.table_checksum(rows, ("b", "a")))


def test_the_package_records_its_identity(package):
    assert len(package["publication_identity_sha256"]) == 64
    assert package["publication_version"] == publication.PUBLICATION_VERSION


# ---------------------------------------------------------------------------
# provenance honesty
# ---------------------------------------------------------------------------


def test_the_package_separates_execution_from_hardening_commits(package):
    prov = package["provenance"]
    assert prov["provenance_hardening_commit"] not in prov["a1_execution_commits"]
    assert prov["provenance_hardening_commit"] not in prov["a2_execution_commits"]
    assert "did NOT produce these results" in prov["historical_shard_provenance"]


def test_the_package_discloses_the_null_shard_defect(package):
    assert "338" in package["provenance"]["historical_shard_provenance"]


# ---------------------------------------------------------------------------
# the package must not touch raw results
# ---------------------------------------------------------------------------


def test_building_the_package_writes_only_under_the_publication_directory():
    source = Path(
        __import__("molfusion_backend.benchmark.publication_cli",
                   fromlist=["x"]).__file__).read_text(encoding="utf-8")
    assert 'open(out /' in source or "out /" in source
    for forbidden in ('results_track_a1.csv", "w', 'results_track_a2.csv", "w'):
        assert forbidden not in source, "publication code must never write raw results"


def test_raw_results_are_untouched_by_the_package():
    import hashlib

    a2 = Path("benchmark_runs/track_a2/results_track_a2.csv")
    if not a2.exists():
        pytest.skip("raw results not present in this checkout")
    digest = hashlib.sha256(a2.read_bytes()).hexdigest()
    assert digest == "c334a2ed6380309fb1e708674bec3f2657b85649a79f4a6827aaa7452035a15e"
