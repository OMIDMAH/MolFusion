"""Phase 6C.3: the Results draft must stay inside the frozen claim registry.

Three guards carry most of the weight here.

The **regression guard**: the Track A1 nonlinear Friedman restricted to the
nine regression endpoints does not reject (p = 0.079), so no
regression-specific nonlinear superiority claim may appear. That is the
easiest overclaim to make by accident, because the nonlinear result is
strong everywhere else.

The **CI guard**: the nonlinear leader's bootstrap interval genuinely is
clear of every competitor's in both tracks, which makes it tempting to
present as a test. It is not one, and inference must be attributed to the
Friedman/Wilcoxon/Holm chain.

The **low-stability guard**: six endpoints cannot support endpoint-specific
claims, and `vdss_lombardo` must stay BORDERLINE rather than being quietly
promoted into the excluded set.
"""

import csv
import json
import re
from pathlib import Path

import pytest

from molfusion_backend.benchmark import publication

DOCS = Path("../docs/manuscript")
RESULTS = DOCS / "RESULTS_DRAFT.md"
EVIDENCE = DOCS / "RESULTS_EVIDENCE_MAP.md"
PACKAGE = Path("benchmark_runs/publication")


def _rows(path: Path):
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.skip("results draft not present")
    return RESULTS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plain(results):
    return re.sub(r"\s+", " ", results.replace("**", "").replace("`", ""))


@pytest.fixture(scope="module")
def sections(results):
    out = {}
    parts = re.split(r"^## (3\.\d+)[^\n]*$", results, flags=re.M)
    for index in range(1, len(parts), 2):
        out[parts[index]] = re.sub(r"\s+", " ", parts[index + 1])
    return out


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def test_all_ten_subsections_exist_in_order(results):
    found = re.findall(r"^## (3\.\d+)", results, re.M)
    assert found == [f"3.{i}" for i in range(1, 11)]


def test_tfidf_robustness_and_exposure_are_separate_sections(sections):
    assert "3.8" in sections and "3.9" in sections
    assert "1.33" in sections["3.8"], "TF-IDF regression evidence belongs in 3.8"
    assert "corpus exposure" in sections["3.9"]
    assert "1.33" not in sections["3.9"]


# ---------------------------------------------------------------------------
# the regression guard
# ---------------------------------------------------------------------------


def test_a1_nonlinear_regression_friedman_does_not_reject():
    """The fact the guard exists to protect."""
    rows = _rows(Path("benchmark_runs/track_a1/analysis/friedman.csv"))
    row = next(r for r in rows
               if r["probe"] == "nonlinear" and r["task_type"] == "regression")
    assert row["reject_at_alpha"] == "False"
    assert 0.07 < float(row["p_value"]) < 0.08


def test_results_discloses_the_regression_non_rejection(plain):
    assert "0.079" in plain
    assert "did not reject" in plain


def test_results_makes_no_regression_specific_nonlinear_claim(plain):
    assert "no claim of nonlinear superiority on regression endpoints is made" in plain.lower()


@pytest.mark.parametrize("phrase", [
    "significantly led regression",
    "significantly outperformed competitors on regression",
    "significant on regression endpoints",
    "nonlinear regression superiority",
    "led the regression endpoints",
])
def test_no_forbidden_regression_superiority_phrasing(plain, phrase):
    assert phrase.lower() not in plain.lower()


# ---------------------------------------------------------------------------
# the confidence-interval guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "significant because the confidence",
    "not significant because",
    "ci non-overlap demonstrated",
    "confidence intervals did not overlap, indicating significance",
])
def test_ci_is_never_presented_as_a_significance_test(plain, phrase):
    assert phrase.lower() not in plain.lower()


def test_inference_is_attributed_to_the_frozen_statistical_chain(plain):
    assert "not on interval separation" in plain.lower()
    for term in ("Friedman", "Holm", "rank-biserial"):
        assert term in plain


def test_the_separation_claim_matches_the_frozen_check(plain):
    report = json.loads((PACKAGE / "publication_report.json").read_text("utf-8"))
    for track in ("a1", "a2"):
        assert report["ci_separation"][track]["nonlinear"]["separated_from_all"] is True
        assert report["ci_separation"][track]["linear"]["separated_from_all"] is False
    assert "separated from those of all competing" in plain
    assert "overlapped substantially" in plain


# ---------------------------------------------------------------------------
# the low-stability guard
# ---------------------------------------------------------------------------


def test_the_six_low_endpoints_are_named(plain):
    for endpoint in publication.PRE_REGISTERED_LOW_STABILITY:
        assert endpoint in plain


@pytest.mark.parametrize("endpoint", publication.PRE_REGISTERED_LOW_STABILITY)
def test_no_endpoint_specific_superlative_for_low_stability_endpoints(plain, endpoint):
    superlatives = ("best", "worst", "outperform", "superior", "leader", "strongest")
    for sentence in re.split(r"(?<=[.])\s+", plain):
        if endpoint in sentence:
            for word in superlatives:
                assert word not in sentence.lower(), (
                    f"endpoint-specific '{word}' claim for low-stability {endpoint}")


def test_low_stability_endpoints_are_not_described_as_removed(plain):
    assert "remain in the 22-endpoint" in plain
    for wrong in ("were excluded from the benchmark", "were dropped", "unreliable datasets"):
        assert wrong not in plain


def test_vdss_lombardo_is_borderline_not_low(plain):
    assert "vdss_lombardo" in plain
    assert "BORDERLINE" in plain
    assert "vdss_lombardo" not in publication.PRE_REGISTERED_LOW_STABILITY
    for sentence in re.split(r"(?<=[.])\s+", plain):
        if "vdss_lombardo" in sentence:
            assert "LOW" not in sentence or "BORDERLINE" in sentence


def test_vdss_lombardo_retention_reason_is_stated(plain):
    assert "post-hoc" in plain.lower()


# ---------------------------------------------------------------------------
# numerical provenance
# ---------------------------------------------------------------------------


def test_headline_ranks_match_the_frozen_tables(plain):
    table2 = {r["representation"]: r for r in
              _rows(PACKAGE / "tables" / "table2_a1_primary_performance.csv")}
    a1 = table2["rdkit_physchem_descriptors"]["nonlinear_mean_rank"]
    assert a1 == "1.91" and "1.91" in plain

    table3 = [r for r in _rows(PACKAGE / "tables" / "table3_a2_robustness.csv")
              if r["probe"] == "nonlinear"
              and r["representation"] == "rdkit_physchem_descriptors"]
    assert table3[0]["a2_mean_rank"] == "1.77" and "1.77" in plain


def test_tfidf_regression_numbers_match_the_frozen_tables(plain):
    def regression_row(path):
        return next(r for r in _rows(path)
                    if r["probe"] == "linear" and r.get("subset") == "regression"
                    and r["representation"] == "smiles_tfidf_4096")

    a1 = regression_row(Path("benchmark_runs/track_a1/analysis/representation_ranks.csv"))
    a2 = regression_row(Path("benchmark_runs/track_a2/analysis/representation_ranks.csv"))
    assert round(float(a1["mean_rank"]), 2) == 1.33
    assert round(float(a2["mean_rank"]), 2) == 2.33
    assert a1["wins"] == "7" and a2["wins"] == "4"
    assert a1["top3"] == "9" and a2["top3"] == "7"
    for token in ("1.33", "2.33", "7 of 9", "4 of 9", "9 of 9"):
        assert token in plain


def test_contrast_reproduction_numbers_match(plain):
    rows = _rows(PACKAGE / "supplementary" / "a1_vs_a2_contrasts.csv")
    assert len(rows) == 11
    assert sum(r["reproduced"] == "True" for r in rows) == 9
    assert sum(r["effect_direction_preserved"] == "True" for r in rows) == 11
    assert "9 remained significant" in plain
    assert "preserved in all 11" in plain


def test_effect_size_range_matches_the_frozen_contrasts(plain):
    rows = [r for r in _rows(PACKAGE / "supplementary" / "all_pairwise_contrasts_a2.csv")
            if r["significant_after_holm"] == "True"]
    magnitudes = [abs(float(r["effect_size_rank_biserial"])) for r in rows]
    assert round(min(magnitudes), 2) == 0.71
    assert round(max(magnitudes), 2) == 0.98
    assert "0.71 to 0.98" in plain


def test_effect_sizes_are_never_vague(plain):
    """'large effect' is only acceptable with a stated magnitude."""
    for sentence in re.split(r"(?<=[.])\s+", plain):
        if "large" in sentence.lower() and "effect" in sentence.lower():
            assert re.search(r"0\.\d\d", sentence), (
                f"vague effect-size claim: {sentence[:120]}")


def test_cost_shares_use_one_denominator(plain):
    """A1 35.4% and A2 30.7% are both nonlinear-only shares."""
    table5 = {r["representation"]: r for r in
              _rows(PACKAGE / "tables" / "table5_computational_cost.csv")}
    share = float(table5["smiles_tfidf_4096"]["share_of_nonlinear_model_seconds"])
    assert round(share * 100, 1) == 35.4
    assert "35.4%" in plain
    assert "30.7%" in plain, "the like-for-like A2 share, not the all-model 29.8%"
    assert "29.8%" not in plain, (
        "29.8% uses the all-model denominator and is not comparable with 35.4%")


def test_endpoint_counts_match_the_manifest(plain):
    manifest = json.loads(
        Path("benchmark_manifests/tdc_admet_group.json").read_text("utf-8"))
    endpoints = manifest["endpoints"]
    classification = sum(1 for v in endpoints.values()
                         if v["task_type"] == "classification")
    assert len(endpoints) == 22 and classification == 13
    assert "22 ADMET endpoints" in plain
    assert "13 classification" in plain and "9 regression" in plain


def test_the_19_endpoint_subset_is_reported_alongside_22(plain):
    assert "19 of the 22" in plain
    assert "22 endpoints remain in the headline" in plain
    assert "1.84" in plain, "19-endpoint nonlinear value must be reported"


# ---------------------------------------------------------------------------
# claim discipline
# ---------------------------------------------------------------------------


def test_no_claim_beyond_the_registry_is_introduced():
    body = EVIDENCE.read_text(encoding="utf-8")
    referenced = set(re.findall(r"\b(C\d+)\b", body))
    registry = {r["claim_id"] for r in
                _rows(PACKAGE / "evidence" / "claim_registry.csv")}
    assert referenced <= registry, f"unregistered claims: {referenced - registry}"
    assert not {"C12", "C13"} & referenced


def test_every_registered_claim_appears_in_the_evidence_map():
    body = EVIDENCE.read_text(encoding="utf-8")
    registry = {r["claim_id"] for r in
                _rows(PACKAGE / "evidence" / "claim_registry.csv")}
    for claim in registry:
        assert re.search(rf"\b{claim}\b", body), f"{claim} unmapped"


def test_every_paragraph_has_a_claim_or_is_marked_descriptive():
    body = EVIDENCE.read_text(encoding="utf-8")
    rows = re.findall(r"^\| (R3\.\d+-P\d+) \| ([^|]+) \|", body, re.M)
    assert rows, "no paragraph rows found"
    for paragraph, claim in rows:
        claim = claim.strip()
        assert claim == "DESCRIPTIVE" or re.search(r"C\d+", claim), (
            f"{paragraph} maps to neither a claim nor DESCRIPTIVE")


def test_c4_and_c9_have_one_primary_home_each():
    body = EVIDENCE.read_text(encoding="utf-8")
    assert "| C4 | **§3.8 only**" in body
    assert "| C9 | **§3.9 only**" in body


# ---------------------------------------------------------------------------
# prohibited wording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "best molecular representation", "universally", "state-of-the-art",
    "contains more information", "external validation", "confirmed by A2",
    "TF-IDF failed", "data leakage", "contamination",
    "all representations were equivalent", "efficient representation",
    "validation cohort", "replication cohort", "trend toward significance",
    "percentage improvement", "relative performance gain",
])
def test_prohibited_wording_absent(plain, phrase):
    assert phrase.lower() not in plain.lower()


def test_no_standalone_proof_language(plain):
    for word in (r"\bproved\b", r"\bproves\b", r"\bconfirmed\b"):
        assert not re.search(word, plain, re.I), f"{word} overstates the evidence"


def test_exposure_section_makes_no_causal_claim(sections):
    body = sections["3.9"].lower()
    for phrase in ("higher overlap", "explains", "inflated", "led to",
                   "improved", "due to exposure", "leakage", "contamination"):
        assert phrase not in body
    assert "untested" in body
    assert "confounded with task family" in body


def test_exposure_uses_the_frozen_term(plain):
    assert "external unsupervised corpus exposure" in plain


def test_no_discussion_style_speculation(plain):
    for phrase in ("this may be because", "one possible explanation",
                   "we speculate", "suggests a molecular mechanism"):
        assert phrase not in plain.lower()


def test_probe_conditioning_is_explicit(plain):
    assert "linear-probe mean rank" in plain or "lowest linear-probe mean rank" in plain
    assert "nonlinear probe" in plain
    assert "depended on the probe" in plain.lower()


# ---------------------------------------------------------------------------
# cross-document consistency
# ---------------------------------------------------------------------------


def test_results_does_not_contradict_the_abstract(plain):
    abstract = (DOCS / "TITLE_AND_ABSTRACT.md").read_text(encoding="utf-8")
    for value in ("1.91", "1.77", "217"):
        assert value in abstract and value in plain
    assert "no representation separated clearly" in abstract.lower()
    assert "no single representation separated clearly" in plain.lower()


def test_results_uses_reproduced_not_confirmed(plain):
    assert "was reproduced under Track A2" in plain


def test_track_roles_match_methods(plain):
    assert "primary, TDC-comparable evaluation" in plain
    assert "supplementary robustness evaluation" in plain


def test_evidence_identity_is_cited():
    identity = "5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18"
    for path in (RESULTS, EVIDENCE):
        assert identity in path.read_text(encoding="utf-8")


def test_raw_results_are_untouched():
    import hashlib

    a2 = Path("benchmark_runs/track_a2/results_track_a2.csv")
    if not a2.exists():
        pytest.skip("raw results not present")
    assert hashlib.sha256(a2.read_bytes()).hexdigest() == (
        "c334a2ed6380309fb1e708674bec3f2657b85649a79f4a6827aaa7452035a15e")
