"""Phase 6C.1: the manuscript plan must agree with the frozen claim registry.

Deliberately small. This is not a manuscript-testing framework -- it checks
the handful of invariants that would silently corrupt the paper if they
drifted: that every claim the plan references actually exists, that the
abstract-permitted set matches the registry rather than an author's memory,
that C9 stays out of the abstract and conclusion, that each main figure is
placed exactly once, and that the two frozen editorial decisions on
endpoint stability survive.
"""

import csv
import re
from pathlib import Path

import pytest

from molfusion_backend.benchmark import publication

DOCS = Path("../docs/manuscript")
REGISTRY = Path("benchmark_runs/publication/evidence/claim_registry.csv")
MAP = DOCS / "claim_to_section.csv"
ABSTRACT = DOCS / "TITLE_AND_ABSTRACT.md"
ARCHITECTURE = DOCS / "MANUSCRIPT_ARCHITECTURE.md"

EVIDENCE_IDENTITY = "5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18"


def _rows(path: Path):
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


@pytest.fixture(scope="module")
def registry():
    if not REGISTRY.exists():
        pytest.skip("publication package not generated in this checkout")
    return {r["claim_id"]: r for r in _rows(REGISTRY)}


@pytest.fixture(scope="module")
def plan():
    if not MAP.exists():
        pytest.skip("manuscript plan not present")
    return _rows(MAP)


@pytest.fixture(scope="module")
def abstract_text():
    if not ABSTRACT.exists():
        pytest.skip("manuscript plan not present")
    body = ABSTRACT.read_text(encoding="utf-8")
    match = re.search(r"## 4\. Abstract.*?\n\n(.*?)\n\n### Numerical", body, re.S)
    assert match, "abstract block not found"
    return re.sub(r"^> ?", "", match.group(1), flags=re.M).replace("\n", " ")


# ---------------------------------------------------------------------------
# the plan may only reference claims that exist
# ---------------------------------------------------------------------------


def test_every_mapped_claim_exists_in_the_registry(plan, registry):
    for row in plan:
        assert row["claim_id"] in registry, f"{row['claim_id']} is not a registered claim"


def test_every_registered_claim_is_placed(plan, registry):
    mapped = {r["claim_id"] for r in plan}
    assert mapped == set(registry), "every registered claim needs a manuscript home"


def test_claim_types_match_the_registry(plan, registry):
    for row in plan:
        assert row["claim_type"] == registry[row["claim_id"]]["claim_type"]


def test_abstract_permission_matches_the_registry(plan, registry):
    """The plan must not widen what the registry allows in the abstract."""
    for row in plan:
        allowed = registry[row["claim_id"]]["allowed_in_abstract"] == "True"
        planned = row["in_abstract"] == "yes"
        assert planned == allowed, (
            f"{row['claim_id']}: plan says abstract={planned}, registry says {allowed}")


def test_conclusion_permission_matches_the_registry(plan, registry):
    for row in plan:
        allowed = registry[row["claim_id"]]["allowed_in_conclusion"] == "True"
        planned = row["in_conclusion"] == "yes"
        assert planned == allowed, (
            f"{row['claim_id']}: plan says conclusion={planned}, registry says {allowed}")


def test_exactly_three_claims_are_abstract_permitted(plan):
    abstract = sorted(r["claim_id"] for r in plan if r["in_abstract"] == "yes")
    assert abstract == ["C1", "C2", "C3"]


# ---------------------------------------------------------------------------
# C9 containment
# ---------------------------------------------------------------------------


def test_c9_is_excluded_from_abstract_and_conclusion(plan):
    c9 = next(r for r in plan if r["claim_id"] == "C9")
    assert c9["in_abstract"] == "no"
    assert c9["in_conclusion"] == "no"
    assert c9["in_discussion"] == "no"


def test_c9_is_still_reported_in_results(plan):
    """Contained, not suppressed -- transparency requires it appear."""
    c9 = next(r for r in plan if r["claim_id"] == "C9")
    assert c9["in_results"] == "yes"
    assert c9["manuscript_section"] == "3.9"


def test_tfidf_robustness_and_exposure_are_separate_sections(plan):
    """Frozen Decision 2: C4 and C9 must not share a subsection."""
    c4 = next(r for r in plan if r["claim_id"] == "C4")
    c9 = next(r for r in plan if r["claim_id"] == "C9")
    assert c4["manuscript_section"] == "3.8"
    assert c9["manuscript_section"] == "3.9"
    assert c4["manuscript_section"] != c9["manuscript_section"]


# ---------------------------------------------------------------------------
# the abstract
# ---------------------------------------------------------------------------


def test_abstract_is_within_the_target_length(abstract_text):
    words = len(abstract_text.split())
    assert 200 <= words <= 250, f"abstract is {words} words"


@pytest.mark.parametrize("phrase", [
    "best representation",
    "best molecular representation",
    "state-of-the-art",
    "universally",
    "contain more molecular information",
    "structural fingerprints",
    "externally validated",
    "leakage",
    "contamination",
    "efficiency score",
    "TF-IDF fails",
])
def test_abstract_contains_no_prohibited_phrase(abstract_text, phrase):
    assert phrase.lower() not in abstract_text.lower()


def test_abstract_does_not_claim_significance_from_intervals(abstract_text):
    lowered = abstract_text.lower()
    assert "confidence interval" not in lowered, (
        "intervals are uncertainty visualization, not the abstract's inferential basis")


def test_abstract_attributes_significance_to_the_frozen_plan(abstract_text):
    lowered = abstract_text.lower()
    assert "friedman" in lowered
    assert "holm" in lowered
    assert "rank-biserial" in lowered


def test_abstract_states_the_linear_negative_result(abstract_text):
    assert "no representation separated clearly" in abstract_text.lower()


def test_abstract_scopes_the_primary_finding_to_the_nonlinear_probe(abstract_text):
    lowered = abstract_text.lower()
    position = lowered.index("217-dimensional")
    assert "nonlinear probe" in lowered[:position], (
        "the primary finding must be scoped to the probe before it is stated")


def test_abstract_omits_claims_that_belong_later(abstract_text):
    lowered = abstract_text.lower()
    for phrase in ("chembl", "kendall", "provenance", "wins fell", "compute"):
        assert phrase not in lowered, f"'{phrase}' belongs later in the paper"


def test_abstract_numbers_match_the_frozen_tables(abstract_text):
    """Every number in the abstract must be traceable, not remembered."""
    assert "217" in abstract_text
    assert "1.91" in abstract_text
    assert "1.77" in abstract_text
    assert "22" in abstract_text

    ranks = Path("benchmark_runs/track_a1/analysis/representation_ranks.csv")
    if not ranks.exists():
        pytest.skip("analysis outputs not present")
    a1 = [r for r in _rows(ranks)
          if r["probe"] == "nonlinear" and r.get("subset", "all") == "all"
          and r["representation"] == "rdkit_physchem_descriptors"]
    assert round(float(a1[0]["mean_rank"]), 2) == 1.91


# ---------------------------------------------------------------------------
# figures and tables
# ---------------------------------------------------------------------------


def test_each_main_figure_is_placed_exactly_once():
    body = ARCHITECTURE.read_text(encoding="utf-8")
    match = re.search(r"### Main figures\n\n(.*?)\n\nFigures 1 and 2", body, re.S)
    assert match, "main figure table not found"
    rows = [line for line in match.group(1).splitlines() if line.startswith("| ")]
    numbers = [line.split("|")[1].strip() for line in rows]
    numbers = [n for n in numbers if n.isdigit()]
    assert numbers == ["1", "2", "3", "4"]
    assert len(numbers) == len(set(numbers)), "a figure is placed more than once"


def test_endpoint_stability_is_supplementary():
    """Frozen Decision 3."""
    body = ARCHITECTURE.read_text(encoding="utf-8")
    assert "Supplementary Figure S1" in body
    main = re.search(r"### Main figures\n\n(.*?)\n\nFigures 1 and 2", body, re.S).group(1)
    assert "stability" not in main.lower(), "stability figure must not be main-text"


def test_every_figure_has_a_source_data_file():
    body = ARCHITECTURE.read_text(encoding="utf-8")
    for index in range(1, 6):
        assert f"figure_{index:02d}_data.csv" in body


# ---------------------------------------------------------------------------
# the two frozen stability decisions
# ---------------------------------------------------------------------------


def test_the_six_pre_registered_endpoints_are_unchanged():
    assert publication.PRE_REGISTERED_LOW_STABILITY == (
        "herg",
        "cyp2c9_substrate_carbonmangels",
        "clearance_hepatocyte_az",
        "cyp2d6_substrate_carbonmangels",
        "cyp3a4_substrate_carbonmangels",
        "bioavailability_ma",
    )


def test_vdss_lombardo_was_not_added_to_the_exclusions():
    """Frozen Decision 1: no post-hoc widening of a pre-registered set."""
    assert "vdss_lombardo" not in publication.PRE_REGISTERED_LOW_STABILITY


def test_vdss_lombardo_remains_flagged_borderline():
    table = Path("benchmark_runs/publication/tables/table6_endpoint_stability.csv")
    if not table.exists():
        pytest.skip("publication package not generated")
    row = next(r for r in _rows(table) if r["endpoint"] == "vdss_lombardo")
    assert row["endpoint_stability_flag"] == "BORDERLINE"
    assert row["pre_registered_low_stability"] == "False"


def test_low_stability_endpoints_remain_in_the_benchmark():
    table = Path("benchmark_runs/publication/tables/table6_endpoint_stability.csv")
    if not table.exists():
        pytest.skip("publication package not generated")
    assert len(_rows(table)) == 22


def test_the_architecture_records_the_borderline_decision():
    body = ARCHITECTURE.read_text(encoding="utf-8")
    assert "vdss_lombardo" in body
    assert "BORDERLINE" in body
    assert "post-hoc filter" in body


# ---------------------------------------------------------------------------
# evidence provenance
# ---------------------------------------------------------------------------


def test_the_plan_cites_the_frozen_evidence_identity():
    for path in (ABSTRACT, ARCHITECTURE, DOCS / "CLAIM_TO_SECTION_MAP.md"):
        assert EVIDENCE_IDENTITY in path.read_text(encoding="utf-8"), (
            f"{path.name} must name the evidence package it was drafted from")


def test_the_evidence_identity_still_matches():
    import json

    report = Path("benchmark_runs/publication/publication_report.json")
    if not report.exists():
        pytest.skip("publication package not generated")
    payload = json.loads(report.read_text("utf-8"))
    assert payload["publication_identity_sha256"] == EVIDENCE_IDENTITY


def test_the_architecture_discloses_the_regression_omnibus_nonresult():
    """A1 nonlinear regression-only Friedman does not reject; the plan says so."""
    body = ARCHITECTURE.read_text(encoding="utf-8")
    assert "0.079" in body
    assert "does not reject" in body


def test_selfies_is_not_claimed_as_benchmarked():
    body = ABSTRACT.read_text(encoding="utf-8")
    assert "SELFIES was implemented" in body
    assert "not** part of Track A" in body or "not* part of Track A" in body
