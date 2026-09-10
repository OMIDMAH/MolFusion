"""Phase 6C.5: whole-manuscript audit.

The section-level suites each guard their own text. This one guards the
assembled document and the things that only exist once the sections are
joined: that every citation resolves and every reference is used, that the
Introduction asks the question the Conclusion answers, and that no guard
established in an earlier phase has been undone by a later edit.

The guards are re-run here across *all* sections rather than trusted from
the phase that introduced them, because the failure mode this phase is
exposed to is a late edit in one file quietly contradicting another.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

from molfusion_backend.benchmark import publication

DOCS = Path("../docs/manuscript")
PACKAGE = Path("benchmark_runs/publication")

ASSEMBLED = DOCS / "MANUSCRIPT_DRAFT.md"
REFERENCES_JSON = DOCS / "references.json"
REFERENCES_MD = DOCS / "REFERENCES.md"

SECTION_DRAFTS = (
    DOCS / "TITLE_AND_ABSTRACT.md",
    DOCS / "INTRODUCTION_DRAFT.md",
    DOCS / "METHODS_DRAFT.md",
    DOCS / "RESULTS_DRAFT.md",
    DOCS / "DISCUSSION_DRAFT.md",
    DOCS / "FIGURE_CAPTIONS.md",
    DOCS / "TABLE_CAPTIONS.md",
)

EVIDENCE_IDENTITY = "5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18"


def _flat(text: str) -> str:
    body = text.replace("**", "").replace("`", "").replace("*", "")
    return re.sub(r"\s+", " ", body)


_NEGATION = (
    "not ", "never", "no ", "rather than", "prohibited", "avoided",
    "without", "does not", "did not", "cannot", "must not", "instead of",
    "nowhere",
)


def _unnegated(body: str, phrase: str, window: int = 220) -> list[str]:
    lowered, target = body.lower(), phrase.lower()
    offending = []
    for match in re.finditer(re.escape(target), lowered):
        before = lowered[max(0, match.start() - window):match.start()]
        if not any(marker in before for marker in _NEGATION):
            offending.append(body[max(0, match.start() - 120):match.end() + 60])
    return offending


@pytest.fixture(scope="module")
def manuscript():
    if not ASSEMBLED.exists():
        pytest.skip("assembled manuscript not present")
    return ASSEMBLED.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plain(manuscript):
    return _flat(manuscript)


@pytest.fixture(scope="module")
def bibliography():
    if not REFERENCES_JSON.exists():
        pytest.skip("bibliography not present")
    return json.loads(REFERENCES_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sections(manuscript):
    out = {}
    for name, pattern in (
        ("abstract", r"^## Abstract$(.*?)^## Keywords$"),
        ("introduction", r"^# 1\. Introduction$(.*?)^## 2\.1 "),
        ("results", r"^## 3\.1 (.*?)^# 4\. Discussion$"),
        ("discussion", r"^# 4\. Discussion$(.*?)^# 5\. Limitations$"),
        ("limitations", r"^# 5\. Limitations$(.*?)^# 6\. Conclusion$"),
        ("conclusion", r"^# 6\. Conclusion$(.*?)^---$"),
    ):
        match = re.search(pattern, manuscript, re.S | re.M)
        out[name] = _flat(match.group(1)) if match else ""
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def test_all_sections_present(manuscript):
    for heading in ("## Abstract", "## Keywords", "# 1. Introduction",
                    "## 2.1 MolFusion framework", "## 3.1 Benchmark coverage",
                    "# 4. Discussion", "# 5. Limitations", "# 6. Conclusion",
                    "## References"):
        assert heading in manuscript, f"missing {heading}"


def test_figure_and_table_captions_are_included(manuscript):
    assert "## Figure captions" in manuscript
    assert "## Main table captions" in manuscript
    assert "## Supplementary table captions" in manuscript


def test_no_duplicated_section_headings(manuscript):
    for heading in ("# 1. Introduction", "# 4. Discussion", "# 5. Limitations",
                    "# 6. Conclusion"):
        assert len(re.findall(rf"^{re.escape(heading)}$", manuscript, re.M)) == 1


def test_assembled_manuscript_is_a_build_product(manuscript):
    assert "Edit the drafts, not this file" in manuscript


# ---------------------------------------------------------------------------
# citations and bibliography
# ---------------------------------------------------------------------------


def test_no_unresolved_citation_placeholders(manuscript):
    leftovers = re.findall(r"\[CITATION: [^\]]+\]", manuscript)
    assert leftovers == [], f"unresolved placeholders: {set(leftovers)}"


def test_every_placeholder_key_used_in_drafts_resolves(bibliography):
    resolution = bibliography["placeholder_resolution"]
    used = set()
    for path in SECTION_DRAFTS:
        if path.exists():
            for key in re.findall(r"\[CITATION: ([^\]]+)\]",
                                  path.read_text(encoding="utf-8")):
                used.add(re.sub(r"\s+", " ", key).replace("–", "-"))
    unresolved = sorted(used - set(resolution))
    assert not unresolved, f"placeholder keys with no reference: {unresolved}"


def test_every_in_text_citation_exists_in_the_bibliography(manuscript, bibliography):
    defined = {e["id"] for e in bibliography["references"]}
    body = manuscript.split("## References")[0]
    cited = set()
    for group in re.findall(r"\[(\d+(?:, \d+)*)\]", body):
        cited.update(int(n) for n in group.split(", "))
    assert cited, "no numeric citations found"
    assert cited <= defined, f"cited but undefined: {sorted(cited - defined)}"


def test_no_orphan_references(manuscript, bibliography):
    defined = {e["id"] for e in bibliography["references"]}
    body = manuscript.split("## References")[0]
    cited = set()
    for group in re.findall(r"\[(\d+(?:, \d+)*)\]", body):
        cited.update(int(n) for n in group.split(", "))
    assert defined <= cited, f"defined but never cited: {sorted(defined - cited)}"


def test_no_duplicate_references(bibliography):
    refs = bibliography["references"]
    keys = [r["key"] for r in refs]
    assert len(keys) == len(set(keys))
    dois = [r["doi"] for r in refs if r.get("doi")]
    assert len(dois) == len(set(dois))
    titles = [r["title"].lower() for r in refs]
    assert len(titles) == len(set(titles))


def test_every_reference_records_its_verification(bibliography):
    for entry in bibliography["references"]:
        assert entry.get("verified_via"), f"{entry['key']} has no verification source"
        assert entry.get("supports"), f"{entry['key']} records no supported statement"
        assert entry.get("authors") and entry.get("title")
        assert entry.get("type") in {"primary", "review", "book", "software"}


def test_no_fabricated_identifier_shape(bibliography):
    """A DOI must look like a DOI; an empty field stays empty rather than guessed."""
    for entry in bibliography["references"]:
        doi = entry.get("doi", "")
        if doi:
            assert re.fullmatch(r"10\.\d{4,9}/\S+", doi), f"{entry['key']}: {doi!r}"
        pmid = entry.get("pmid", "")
        if pmid:
            assert pmid.isdigit()


def test_holm_reference_has_no_invented_doi(bibliography):
    holm = next(e for e in bibliography["references"] if e["key"] == "holm1979")
    assert holm["doi"] == "", "no DOI is assigned to Holm 1979; it must stay empty"
    assert holm.get("jstor")


def test_rdkit_is_cited_as_software_not_an_article(bibliography):
    rdkit = next(e for e in bibliography["references"] if e["key"] == "rdkit")
    assert rdkit["type"] == "software"
    assert rdkit["container"] == "Software"
    assert "zenodo" in rdkit["doi"]


def test_morgan_and_ecfp_are_separate_references(bibliography):
    keys = {e["key"] for e in bibliography["references"]}
    assert {"morgan1965", "rogers2010"} <= keys
    resolution = bibliography["placeholder_resolution"]
    assert resolution["Morgan fingerprint; ECFP"] == [16, 18]


def test_standalone_bibliography_matches_the_json(bibliography):
    if not REFERENCES_MD.exists():
        pytest.skip("REFERENCES.md not generated")
    body = REFERENCES_MD.read_text(encoding="utf-8")
    assert len(re.findall(r"^\d+\. ", body, re.M)) == len(bibliography["references"])
    for entry in bibliography["references"]:
        if entry.get("doi"):
            assert entry["doi"] in body


# ---------------------------------------------------------------------------
# guards carried forward across the whole manuscript
# ---------------------------------------------------------------------------


def test_ci_values_are_the_frozen_ones(plain):
    assert "[1.45, 2.41]" in plain
    assert "[1.32, 2.27]" in plain


@pytest.mark.parametrize("path", SECTION_DRAFTS + (ASSEMBLED,), ids=lambda p: p.name)
def test_the_mistyped_ci_bound_appears_nowhere(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    assert "2.47" not in path.read_text(encoding="utf-8")


def test_ci_values_match_the_frozen_analysis_tables():
    for track, bounds in (("track_a1", (1.45, 2.41)), ("track_a2", (1.32, 2.27))):
        path = Path(f"benchmark_runs/{track}/analysis/bootstrap_mean_rank.csv")
        if not path.exists():
            pytest.skip("analysis outputs not present")
        with open(path, encoding="utf-8", newline="") as handle:
            row = next(r for r in csv.DictReader(handle)
                       if r["probe"] == "nonlinear"
                       and r["representation"] == "rdkit_physchem_descriptors")
        assert (round(float(row["ci_lower_95"]), 2),
                round(float(row["ci_upper_95"]), 2)) == bounds


@pytest.mark.parametrize("path", SECTION_DRAFTS + (ASSEMBLED,), ids=lambda p: p.name)
def test_compute_denominator_is_never_mixed(path):
    """35.4 % and 29.8 % use different denominators and must not be paired."""
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    body = _flat(path.read_text(encoding="utf-8"))
    if "35.4%" in body and "29.8%" in body:
        # Permitted only in the evidence maps, which exist to reconcile them.
        assert "denominator" in body.lower(), (
            f"{path.name} pairs 35.4% with 29.8% without reconciling denominators")


def test_the_manuscript_uses_like_for_like_shares(plain):
    assert "35.4%" in plain and "30.7%" in plain
    assert "29.8%" not in plain


def test_linear_result_is_difference_without_leadership(sections):
    body = sections["discussion"]
    assert "rejected" in body.lower()
    assert "unresolved leadership" in body


@pytest.mark.parametrize("phrase", [
    "the linear probe found no differences",
    "no representation differences",
    "all representations were equivalent",
    "representations were indistinguishable",
])
def test_no_no_difference_claim(plain, phrase):
    assert phrase.lower() not in plain.lower()


def test_regression_guard(plain, sections):
    assert "0.079" in plain
    assert "did not reject" in plain
    for phrase in ("significantly outperformed on regression",
                   "significant nonlinear regression superiority",
                   "superior on regression endpoints",
                   "led the regression endpoints"):
        assert phrase.lower() not in plain.lower()


def test_a1_nonlinear_regression_friedman_still_does_not_reject():
    path = Path("benchmark_runs/track_a1/analysis/friedman.csv")
    if not path.exists():
        pytest.skip("analysis outputs not present")
    with open(path, encoding="utf-8", newline="") as handle:
        row = next(r for r in csv.DictReader(handle)
                   if r["probe"] == "nonlinear" and r["task_type"] == "regression")
    assert row["reject_at_alpha"] == "False"
    assert round(float(row["p_value"]), 3) == 0.079


def test_ci_is_never_the_significance_test(plain):
    for phrase in ("significant because the confidence",
                   "confidence intervals did not overlap",
                   "ci non-overlap", "intervals demonstrated significance"):
        assert phrase.lower() not in plain.lower()
    assert "not a test of difference between" in plain or "not on interval separation" in plain


def test_c9_absent_from_abstract_and_conclusion(sections):
    for name in ("abstract", "conclusion"):
        body = sections[name].lower()
        for token in ("chembl", "corpus exposure", "overlap"):
            assert token not in body, f"C9 material in {name}"


def test_c9_not_causal_in_discussion(sections):
    body = sections["discussion"].lower()
    for phrase in ("chembl exposure", "corpus exposure explains",
                   "because of chembl", "overfits chembl"):
        assert phrase not in body


def test_c9_in_limitations_is_exploratory(sections):
    body = sections["limitations"]
    assert "external unsupervised corpus exposure" in body
    assert "untested" in body.lower() and "confounded" in body.lower()


@pytest.mark.parametrize("path", SECTION_DRAFTS + (ASSEMBLED,), ids=lambda p: p.name)
def test_exposure_never_called_leakage(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    body = _flat(path.read_text(encoding="utf-8"))
    for word in ("data leakage", "label leakage", "contamination"):
        assert not _unnegated(body, word), f"{path.name}: unqualified '{word}'"


@pytest.mark.parametrize("phrase", [
    "external validation", "external cohort", "independent dataset validation",
    "validation cohort", "replication cohort", "confirmed by A2",
])
def test_a2_never_called_external_validation(plain, phrase):
    assert not _unnegated(plain, phrase)


def test_information_content_guard(plain):
    for phrase in ("contain more information", "contains more molecular information",
                   "encode more molecular information", "intrinsically superior"):
        assert not _unnegated(plain, phrase)
    assert "predictive accessibility" in plain


@pytest.mark.parametrize("endpoint", publication.PRE_REGISTERED_LOW_STABILITY)
def test_low_stability_endpoints_carry_no_superlative(plain, endpoint):
    for sentence in re.split(r"(?<=[.])\s+", plain):
        if endpoint in sentence:
            for word in ("best", "worst", "outperform", "superior", "leader", "strongest"):
                assert word not in sentence.lower(), (
                    f"endpoint-specific '{word}' for {endpoint}")


def test_vdss_lombardo_is_borderline_not_low(plain):
    assert "vdss_lombardo" in plain and "BORDERLINE" in plain
    assert "vdss_lombardo" not in publication.PRE_REGISTERED_LOW_STABILITY
    assert "post-hoc" in plain.lower()


def test_bottom_rank_asymmetry_is_preserved(plain):
    assert "all four probe" in plain or "every probe and track" in plain
    for wrong in ("the same two representations were always last",
                  "the bottom two were the same throughout"):
        assert wrong.lower() not in plain.lower()


def test_provenance_guard(sections):
    body = sections["limitations"]
    assert "did not produce the results reported here" in body
    assert "logging defect" in body
    for wrong in ("provenance was complete", "results were reconstructed"):
        assert wrong not in body


# ---------------------------------------------------------------------------
# Introduction and Conclusion
# ---------------------------------------------------------------------------


def test_introduction_states_the_research_question(sections):
    body = sections["introduction"]
    assert "How does the apparent utility of heterogeneous molecular" in body
    assert "linear and nonlinear" in body


def test_introduction_contains_no_result(sections):
    body = sections["introduction"].lower()
    for phrase in ("1.91", "1.77", "achieved the best", "ranked highest",
                   "outperform", "strongest cross-endpoint"):
        assert phrase not in body, f"Introduction leaks a result: {phrase}"


def test_introduction_promises_only_what_is_delivered(sections):
    body = sections["introduction"].lower()
    for phrase in ("improved admet prediction", "state-of-the-art",
                   "new model architecture", "outperforms"):
        assert phrase not in body
    assert "is not a predictive model" in body
    assert "not offered as an improved admet predictor" in body


def test_introduction_marks_learned_representations_as_out_of_scope(sections):
    body = sections["introduction"]
    assert "were not part of the benchmark and are not evaluated" in body


def test_introduction_contribution_paragraph_matches_the_frozen_four(sections):
    body = sections["introduction"].lower()
    assert "contributions of this work are fourfold" in body
    for element in ("frozen-protocol framework", "seven heterogeneous",
                    "dual-track", "probe-dependent analysis"):
        assert element in body


def test_conclusion_answers_the_introduction_question(sections):
    intro, conclusion = sections["introduction"].lower(), sections["conclusion"].lower()
    assert "probe" in intro and "probe-dependent" in conclusion
    assert "reproducible framework" in conclusion
    assert "not as a new ADMET predictor" in sections["conclusion"]


def test_abstract_still_governed_by_c1_c2_c3(sections):
    body = sections["abstract"].lower()
    assert "217-dimensional physicochemical" in body
    assert "reproduced under independently generated" in body
    assert "no representation separated clearly" in body
    for later in ("chembl", "kendall", "compute", "provenance", "wins fell"):
        assert later not in body


# ---------------------------------------------------------------------------
# terminology and style
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term", [
    "representation", "probe", "endpoint", "Track A1", "Track A2",
    "scaffold repartitioning", "physicochemical descriptors",
    "external unsupervised corpus exposure",
])
def test_standard_terminology_is_used(plain, term):
    assert term in plain


def test_prohibited_collective_term_absent(plain):
    assert not _unnegated(plain, publication.PROHIBITED_COLLECTIVE_TERM)


@pytest.mark.parametrize("word", [
    "remarkably", "surprisingly", "powerful", "breakthrough",
    "best-in-class", "unprecedented",
])
def test_no_promotional_language(plain, word):
    assert word.lower() not in plain.lower()


@pytest.mark.parametrize("phrase", [
    "best molecular representation", "universally", "state-of-the-art",
    "TF-IDF failed", "percentage improvement", "efficiency score",
])
def test_whole_manuscript_prohibited_wording(plain, phrase):
    assert not _unnegated(plain, phrase)


def test_no_standalone_proof_language(plain):
    for pattern in (r"\bproved\b", r"\bproves\b"):
        assert not re.search(pattern, plain, re.I)


def test_no_external_numerical_comparison(plain):
    for name in ("ADMET-AI", "MiniMol", "MapLight", "ChemBERTa"):
        assert name.lower() not in plain.lower()
    for phrase in ("outperforms published", "exceeds prior models",
                   "our auroc exceeds", "compared with published models"):
        assert phrase.lower() not in plain.lower()


# ---------------------------------------------------------------------------
# captions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("figure", ["Figure 1.", "Figure 2.", "Figure 3.",
                                    "Figure 4.", "Supplementary Figure S1."])
def test_every_figure_has_a_caption(manuscript, figure):
    assert f"**{figure}" in manuscript


@pytest.mark.parametrize("table", ["Table 1.", "Table 2.", "Table 3.", "Table 4.",
                                   "Table 5.", "Table 6.", "Table 7."])
def test_every_main_table_has_a_caption(manuscript, table):
    assert f"**{table}" in manuscript


def test_supplementary_tables_have_captions(manuscript):
    for index in range(1, 10):
        assert f"**Supplementary Table S{index}." in manuscript


def test_captions_state_rank_direction(manuscript):
    captions = manuscript.split("## Figure captions")[1]
    assert captions.count("rank") > 10
    assert "lower mean rank indicates stronger" in captions or "Rank 1 denotes" in captions


def test_statistical_captions_name_the_inference_unit(manuscript):
    # Captions wrap across lines, so flatten before matching phrases.
    captions = _flat(manuscript.split("## Figure captions")[1])
    assert "endpoint is the resampling unit" in captions
    assert "endpoint is the unit of statistical inference" in captions


def test_figure_two_caption_does_not_claim_significance(manuscript):
    caption = re.search(r"\*\*Figure 2\..*?(?=\*\*Figure 3\.)", manuscript, re.S)
    assert caption
    body = caption.group(0)
    assert "not a test of difference" in body
    assert "[1.32, 2.27]" in body and "[1.45, 2.41]" in body
    assert "2.47" not in body


# ---------------------------------------------------------------------------
# frozen scientific artifacts
# ---------------------------------------------------------------------------


def test_publication_identity_unchanged():
    report = PACKAGE / "publication_report.json"
    if not report.exists():
        pytest.skip("publication package not generated")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["publication_identity_sha256"] == EVIDENCE_IDENTITY
    assert payload["inputs"]["a1_raw_identity"] == (
        "d40ef09b398f47914aa51f99fd6a4f5893f7778b50c0cca04404b575632de868")
    assert payload["inputs"]["a2_raw_identity"] == (
        "9dd5dfa6067c8a760b0bb8fb39648f71f662f2fa1bbf4cc5d7cb0cd495a69f14")
    assert len(payload["claim_registry"]) == 11


def test_raw_results_unchanged():
    a2 = Path("benchmark_runs/track_a2/results_track_a2.csv")
    if not a2.exists():
        pytest.skip("raw results not present")
    assert hashlib.sha256(a2.read_bytes()).hexdigest() == (
        "c334a2ed6380309fb1e708674bec3f2657b85649a79f4a6827aaa7452035a15e")


def test_claim_registry_unmodified():
    path = PACKAGE / "evidence" / "claim_registry.csv"
    if not path.exists():
        pytest.skip("publication package not generated")
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 11
    assert {r["claim_id"] for r in rows} == {f"C{i}" for i in range(1, 12)}
