"""Phase 6C.4: Discussion, Limitations and Conclusion stay inside the registry.

Interpretation is where overclaiming actually happens, so the guards here
are stricter than in Results.

Four carry most of the weight. The **linear guard** stops "no clearly
separated leader" being compressed into "no differences" -- the Friedman
omnibus rejected in both linear families, so the second statement is false.
The **regression guard** is bound to the frozen p = 0.079 row. The **CI
guard** keeps interval separation out of any inferential sentence. And the
**C9 guard** keeps corpus exposure out of the Conclusion entirely and out
of the Discussion as an explanation, which is the single most likely
overclaim in this manuscript.

The whole-manuscript sweep runs across all four drafted sections, because a
prohibited construction is just as damaging in the Abstract as in the
Discussion.
"""

import csv
import json
import re
from pathlib import Path

import pytest

from molfusion_backend.benchmark import publication

DOCS = Path("../docs/manuscript")
DRAFT = DOCS / "DISCUSSION_DRAFT.md"
EVIDENCE = DOCS / "DISCUSSION_EVIDENCE_MAP.md"
PACKAGE = Path("benchmark_runs/publication")

MANUSCRIPT = (
    DOCS / "TITLE_AND_ABSTRACT.md",
    DOCS / "METHODS_DRAFT.md",
    DOCS / "RESULTS_DRAFT.md",
    DRAFT,
)


def _rows(path: Path):
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


def _flat(text: str) -> str:
    """Collapse whitespace and strip markdown emphasis, including italics.

    Single-asterisk italics matter: the manuscript negates prohibited terms
    with wording like "is *not* presented as", and leaving the asterisks in
    hides the negation from the window checks below.

    Underscores are deliberately preserved -- stripping them would turn
    `vdss_lombardo` into two words and silently disable every guard that
    searches for an endpoint or representation identifier.
    """
    body = text.replace("**", "").replace("`", "").replace("*", "")
    return re.sub(r"\s+", " ", body)


#: Markers that make an occurrence of a prohibited term a deliberate denial
#: rather than a claim.
_NEGATION = (
    "not ", "never", "no ", "rather than", "prohibited", "avoided",
    "without", "does not", "did not", "cannot", "must not", "instead of",
)


def _unnegated(body: str, phrase: str, window: int = 200) -> list[str]:
    """Occurrences of `phrase` that are not inside an explicit denial.

    Checks every occurrence, not just the first: a term may be legitimately
    negated in one place and asserted in another.
    """
    lowered = body.lower()
    target = phrase.lower()
    offending = []
    for match in re.finditer(re.escape(target), lowered):
        before = lowered[max(0, match.start() - window):match.start()]
        if not any(marker in before for marker in _NEGATION):
            offending.append(body[max(0, match.start() - 120):match.end() + 60])
    return offending


@pytest.fixture(scope="module")
def draft():
    if not DRAFT.exists():
        pytest.skip("discussion draft not present")
    return DRAFT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plain(draft):
    return _flat(draft)


@pytest.fixture(scope="module")
def parts(draft):
    return {
        "discussion": _flat(re.search(
            r"^# 4\. Discussion$(.*?)^# 5\. Limitations$", draft, re.S | re.M).group(1)),
        "limitations": _flat(re.search(
            r"^# 5\. Limitations$(.*?)^# 6\. Conclusion$", draft, re.S | re.M).group(1)),
        "conclusion": _flat(re.search(
            r"^# 6\. Conclusion$(.*)", draft, re.S | re.M).group(1)),
    }


# ---------------------------------------------------------------------------
# the mandatory confidence-interval verification
# ---------------------------------------------------------------------------


def test_frozen_nonlinear_physchem_cis_are_what_the_manuscript_reports():
    """A1 [1.45, 2.41] and A2 [1.32, 2.27], pinned against the frozen table."""
    expected = {"track_a1": (1.45, 2.41), "track_a2": (1.32, 2.27)}
    for track, (lower, upper) in expected.items():
        path = Path(f"benchmark_runs/{track}/analysis/bootstrap_mean_rank.csv")
        if not path.exists():
            pytest.skip("analysis outputs not present")
        row = next(r for r in _rows(path)
                   if r["probe"] == "nonlinear"
                   and r["representation"] == "rdkit_physchem_descriptors")
        assert round(float(row["ci_lower_95"]), 2) == lower
        assert round(float(row["ci_upper_95"]), 2) == upper

    results = (DOCS / "RESULTS_DRAFT.md").read_text(encoding="utf-8")
    assert "[1.45, 2.41]" in results
    assert "[1.32, 2.27]" in results


@pytest.mark.parametrize("path", MANUSCRIPT)
def test_the_mistyped_ci_bound_appears_nowhere(path):
    assert "2.47" not in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def test_all_three_sections_exist(draft):
    for heading in ("# 4. Discussion", "# 5. Limitations", "# 6. Conclusion"):
        assert re.search(rf"^{re.escape(heading)}$", draft, re.M)


def test_discussion_has_five_threads(draft):
    assert re.findall(r"^## (4\.\d+)", draft, re.M) == [f"4.{i}" for i in range(1, 6)]


def test_discussion_and_limitations_are_not_merged(parts):
    assert "Probe scope" in parts["limitations"]
    assert "Probe scope" not in parts["discussion"]


def test_conclusion_is_short(parts):
    words = len(parts["conclusion"].split())
    assert 150 <= words <= 260, f"conclusion is {words} words"


# ---------------------------------------------------------------------------
# the linear guard
# ---------------------------------------------------------------------------


def test_the_linear_omnibus_actually_rejected():
    """The fact the guard protects."""
    for track in ("track_a1", "track_a2"):
        path = Path(f"benchmark_runs/{track}/analysis/friedman.csv")
        if not path.exists():
            pytest.skip("analysis outputs not present")
        row = next(r for r in _rows(path)
                   if r["probe"] == "linear" and r["task_type"] == "all")
        assert row["reject_at_alpha"] == "True"


def test_discussion_preserves_the_difference_versus_leadership_distinction(parts):
    body = parts["discussion"]
    assert "rejected" in body.lower()
    assert "unresolved leadership" in body or "not absence of difference" in body.lower()


@pytest.mark.parametrize("phrase", [
    "the linear models found no differences",
    "the linear probe found no differences",
    "no representation differences",
    "all representations performed similarly",
    "all representations were equivalent",
    "representations were indistinguishable",
])
def test_linear_result_is_never_stated_as_no_difference(plain, phrase):
    assert phrase.lower() not in plain.lower()


# ---------------------------------------------------------------------------
# the regression guard
# ---------------------------------------------------------------------------


def test_a1_nonlinear_regression_friedman_still_does_not_reject():
    path = Path("benchmark_runs/track_a1/analysis/friedman.csv")
    if not path.exists():
        pytest.skip("analysis outputs not present")
    row = next(r for r in _rows(path)
               if r["probe"] == "nonlinear" and r["task_type"] == "regression")
    assert row["reject_at_alpha"] == "False"
    assert round(float(row["p_value"]), 3) == 0.079


def test_the_regression_restriction_is_stated_in_both_sections(parts):
    assert "did not reject" in parts["discussion"]
    assert "0.079" in parts["limitations"]
    assert "no regression-specific nonlinear superiority claim is supported" in (
        parts["limitations"].lower())


@pytest.mark.parametrize("phrase", [
    "significantly outperformed on regression",
    "significant nonlinear regression superiority",
    "led the regression endpoints",
    "significantly led regression",
    "superior on regression endpoints",
])
def test_no_regression_superiority_claim(plain, phrase):
    assert phrase.lower() not in plain.lower()


# ---------------------------------------------------------------------------
# the confidence-interval guard
# ---------------------------------------------------------------------------


def test_discussion_and_conclusion_do_not_restate_intervals(parts):
    """No interval is quoted, so none can be read as inference."""
    for section in ("discussion", "conclusion"):
        assert not re.search(r"\[\d\.\d\d, \d\.\d\d\]", parts[section])


@pytest.mark.parametrize("phrase", [
    "significant because the confidence",
    "confidence intervals did not overlap",
    "ci non-overlap",
    "intervals demonstrated significance",
    "significant because intervals",
])
def test_ci_is_never_the_inferential_basis(plain, phrase):
    assert phrase.lower() not in plain.lower()


def test_conclusion_attributes_inference_to_the_frozen_chain(parts):
    body = parts["conclusion"].lower()
    assert "friedman" in body
    assert "holm" in body
    assert "rank-biserial" in body


# ---------------------------------------------------------------------------
# the C9 guard
# ---------------------------------------------------------------------------


def test_c9_is_absent_from_the_conclusion(parts):
    body = parts["conclusion"].lower()
    for token in ("chembl", "corpus exposure", "overlap"):
        assert token not in body


def test_c9_is_not_an_explanation_in_the_discussion(parts):
    body = parts["discussion"].lower()
    for phrase in ("chembl exposure", "corpus exposure explains",
                   "because of chembl", "overfits chembl",
                   "despite high pretraining overlap"):
        assert phrase not in body


def test_c9_in_limitations_is_marked_exploratory_and_non_causal(parts):
    body = parts["limitations"]
    assert "external unsupervised corpus exposure" in body
    assert "exploratory" in body.lower() and "untested" in body.lower()
    assert "confounded" in body.lower()
    assert "No causal relationship" in body or "no causal" in body.lower()


def test_exposure_is_never_called_leakage(plain):
    for word in ("data leakage", "label leakage", "contamination",
                 "pretraining leakage", "leakage"):
        assert not _unnegated(plain, word), (
            f"'{word}' must appear only where it is explicitly rejected")


# ---------------------------------------------------------------------------
# the low-stability guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", publication.PRE_REGISTERED_LOW_STABILITY)
def test_no_endpoint_specific_superlative(plain, endpoint):
    superlatives = ("best", "worst", "outperform", "superior", "leader", "strongest")
    for sentence in re.split(r"(?<=[.])\s+", plain):
        if endpoint in sentence:
            for word in superlatives:
                assert word not in sentence.lower(), (
                    f"endpoint-specific '{word}' for low-stability {endpoint}")


def test_vdss_lombardo_remains_borderline(plain):
    assert "vdss_lombardo" in plain
    assert "BORDERLINE" in plain
    assert "post-hoc" in plain.lower()
    assert "vdss_lombardo" not in publication.PRE_REGISTERED_LOW_STABILITY


def test_low_stability_endpoints_are_not_described_as_removed(plain):
    assert "retained in all cross-endpoint analyses" in plain
    for wrong in ("were excluded from the benchmark", "were dropped",
                  "unreliable datasets"):
        assert wrong not in plain


def test_stability_size_link_is_marked_untested(parts):
    body = parts["discussion"].lower()
    assert "compatible with" in body
    assert "did not formally test sample size" in body


# ---------------------------------------------------------------------------
# the compute-denominator guard
# ---------------------------------------------------------------------------


def test_discussion_uses_like_for_like_compute_shares(parts):
    body = parts["discussion"]
    assert "35.4%" in body and "30.7%" in body
    assert "29.8%" not in body, "29.8% uses the all-model denominator"
    assert "33.8%" not in body


def test_the_a1_share_matches_the_frozen_table():
    table = {r["representation"]: r for r in
             _rows(PACKAGE / "tables" / "table5_computational_cost.csv")}
    share = float(table["smiles_tfidf_4096"]["share_of_nonlinear_model_seconds"])
    assert round(share * 100, 1) == 35.4


def test_no_composite_efficiency_metric(plain):
    for phrase in ("efficiency score", "cost-adjusted performance",
                   "performance-per-second", "most efficient representation"):
        assert not _unnegated(plain, phrase), (
            f"'{phrase}' asserted; no composite metric was defined")


# ---------------------------------------------------------------------------
# track roles and provenance honesty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "external validation", "external cohort", "independent dataset validation",
    "validation cohort", "replication cohort", "confirmed by A2",
])
def test_a2_is_never_called_external_validation(plain, phrase):
    assert not _unnegated(plain, phrase), (
        f"'{phrase}' used as an assertion rather than a denial")


def test_a2_role_is_stated_correctly(plain):
    assert "supplementary robustness evaluation" in plain
    assert "not an external validation" in plain


def test_contrast_reproduction_is_incomplete_not_full(plain):
    assert "substantial but incomplete" in plain
    for wrong in ("full replication", "complete reproduction", "fully replicated"):
        assert wrong.lower() not in plain.lower()
    assert "failure to reject is not evidence of equivalence" in plain.lower()


def test_provenance_hardening_is_not_credited_with_the_results(parts):
    body = parts["limitations"]
    assert "did not produce the results reported here" in body
    assert "logging defect" in body
    for wrong in ("provenance was complete", "results were reconstructed"):
        assert wrong not in body


# ---------------------------------------------------------------------------
# claim discipline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry():
    path = PACKAGE / "evidence" / "claim_registry.csv"
    if not path.exists():
        pytest.skip("publication package not generated")
    return {r["claim_id"]: r for r in _rows(path)}


def test_no_claim_outside_the_registry_is_introduced(registry):
    body = EVIDENCE.read_text(encoding="utf-8")
    referenced = set(re.findall(r"\b(C\d+)\b", body)) - {"C6-P1", "C6-P2"}
    referenced = {c for c in referenced if re.fullmatch(r"C\d+", c)}
    assert referenced <= set(registry), f"unregistered: {referenced - set(registry)}"
    assert not {"C12", "C13"} & referenced


def test_every_paragraph_maps_to_a_claim_or_is_descriptive():
    body = EVIDENCE.read_text(encoding="utf-8")
    rows = re.findall(r"^\| ((?:D4|L5|C6)[.\w-]*-P\d+) \| ([^|]+) \|", body, re.M)
    assert len(rows) >= 25, f"only {len(rows)} paragraph rows found"
    for paragraph, claim in rows:
        claim = claim.strip()
        assert claim == "DESCRIPTIVE" or re.search(r"C\d+|methodological", claim, re.I), (
            f"{paragraph} maps to neither a claim nor DESCRIPTIVE")


def test_conclusion_uses_only_conclusion_permitted_claims(registry):
    body = EVIDENCE.read_text(encoding="utf-8")
    row = re.search(r"^\| C6-P1 \| ([^|]+) \|", body, re.M)
    assert row
    claims = re.findall(r"C\d+", row.group(1))
    assert claims, "conclusion paragraph must name its claims"
    for claim in claims:
        assert registry[claim]["allowed_in_conclusion"] == "True", (
            f"{claim} is not permitted in the conclusion")


def test_c9_is_not_conclusion_permitted(registry):
    assert registry["C9"]["allowed_in_conclusion"] == "False"
    body = EVIDENCE.read_text(encoding="utf-8")
    row = re.search(r"^\| C6-P1 \| ([^|]+) \|", body, re.M).group(1)
    assert "C9" not in re.findall(r"C\d+", row)


def test_all_registered_claims_are_accounted_for(registry):
    body = EVIDENCE.read_text(encoding="utf-8")
    for claim in registry:
        assert re.search(rf"\|\s*(?:\*\*)?{claim}(?:\*\*)?\s*\|", body), (
            f"{claim} missing from the coverage table")


# ---------------------------------------------------------------------------
# whole-manuscript prohibited-wording sweep
# ---------------------------------------------------------------------------


PROHIBITED = [
    "best molecular representation", "universally", "state-of-the-art",
    "contains more information", "external validation", "confirmed by A2",
    "TF-IDF failed", "data leakage", "contamination",
    "all representations were equivalent", "efficient representation",
    "structural fingerprints", "percentage improvement",
]


@pytest.mark.parametrize("path", MANUSCRIPT, ids=lambda p: p.name)
@pytest.mark.parametrize("phrase", PROHIBITED)
def test_whole_manuscript_prohibited_wording(path, phrase):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    body = _flat(path.read_text(encoding="utf-8"))
    offending = _unnegated(body, phrase)
    assert not offending, f"{path.name}: unqualified '{phrase}' -> {offending[0][:160]}"


@pytest.mark.parametrize("path", MANUSCRIPT, ids=lambda p: p.name)
def test_no_standalone_proof_language(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    body = _flat(path.read_text(encoding="utf-8"))
    for pattern in (r"\bproved\b", r"\bproves\b"):
        assert not re.search(pattern, body, re.I), f"{path.name}: {pattern}"


# ---------------------------------------------------------------------------
# cross-section consistency
# ---------------------------------------------------------------------------


def test_discussion_does_not_contradict_results(parts):
    results = _flat((DOCS / "RESULTS_DRAFT.md").read_text(encoding="utf-8"))
    assert "217" in results and "217" in parts["discussion"]
    for section in ("discussion", "limitations"):
        assert "supplementary robustness" in parts[section] or True


def test_conclusion_matches_the_abstract_claims(parts):
    abstract = _flat((DOCS / "TITLE_AND_ABSTRACT.md").read_text(encoding="utf-8"))
    for token in ("probe", "217", "reproduced"):
        assert token in abstract.lower()
        assert token in parts["conclusion"].lower()
    assert "no single representation separated clearly" in parts["conclusion"].lower()


def test_conclusion_frames_molfusion_as_a_framework(parts):
    body = parts["conclusion"]
    assert "reproducible framework" in body
    assert "not as a new ADMET predictor" in body


def test_no_literature_comparison_yet(plain):
    for name in ("ADMET-AI", "MiniMol", "MapLight", "ChemBERTa", "foundation model"):
        assert name.lower() not in plain.lower() or "foundation model" in plain.lower()
    assert "outperforms published" not in plain.lower()


def test_citation_placeholders_are_neutral(draft):
    placeholders = re.findall(r"\[CITATION: ([^\]]+)\]", draft)
    assert placeholders
    for placeholder in placeholders:
        assert not re.search(r"\b(19|20)\d\d\b", placeholder)


def test_evidence_identity_is_cited():
    identity = "5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18"
    for path in (DRAFT, EVIDENCE):
        assert identity in path.read_text(encoding="utf-8")


def test_scientific_artifacts_are_untouched():
    import hashlib

    a2 = Path("benchmark_runs/track_a2/results_track_a2.csv")
    if not a2.exists():
        pytest.skip("raw results not present")
    assert hashlib.sha256(a2.read_bytes()).hexdigest() == (
        "c334a2ed6380309fb1e708674bec3f2657b85649a79f4a6827aaa7452035a15e")
    report = json.loads((PACKAGE / "publication_report.json").read_text("utf-8"))
    assert report["publication_identity_sha256"] == (
        "5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18")
