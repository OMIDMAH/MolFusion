"""Phase 6C.7: the pandoc-generated FINAL submission artifacts.

`test_submission_package.py` covers the zero-dependency review builds from
Phase 6C.6. This file covers what pandoc and the journal CSL actually
produced, which is the thing that gets submitted.

That distinction earned itself immediately. Two defects were invisible in
the Markdown source and only appeared once the bibliography was rendered:
an unescaped ``%`` in the SELFIES title started a BibTeX comment and
truncated that reference, and the journal style emits no DOI for any
non-article type, so the RDKit identifier vanished from the bibliography
entirely.
"""

import csv
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

from molfusion_backend.benchmark import publication, submission_cli

DOCS = Path("../docs/manuscript")
SUB = DOCS / "submission" / "jcheminformatics"

FINAL_DOCX = SUB / "MolFusion_JCheminformatics_Manuscript_FINAL.docx"
FINAL_PDF = SUB / "MolFusion_JCheminformatics_Manuscript_FINAL.pdf"
SI_FINAL_DOCX = SUB / "supplementary" / "MolFusion_Supplementary_Information_FINAL.docx"
RENDERED = SUB / "validation" / "article_rendered.txt"
SI_TABLES = SUB / "supplementary" / "tables"
MANIFEST = SI_TABLES / "SUPPLEMENTARY_TABLE_MANIFEST.json"


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "").replace("`", "").replace("*", ""))


_NEGATION = ("not ", "never", "no ", "rather than", "without", "does not",
             "did not", "cannot", "must not", "nowhere")


def _unnegated(body: str, phrase: str, window: int = 220) -> list[str]:
    lowered, target = body.lower(), phrase.lower()
    return [body[max(0, m.start() - 100):m.end() + 40]
            for m in re.finditer(re.escape(target), lowered)
            if not any(n in lowered[max(0, m.start() - window):m.start()]
                       for n in _NEGATION)]


@pytest.fixture(scope="module")
def docx_text():
    if not FINAL_DOCX.exists():
        pytest.skip("pandoc FINAL DOCX not built")
    return submission_cli.read_docx_text(FINAL_DOCX)


@pytest.fixture(scope="module")
def flat_docx(docx_text):
    return _flat(docx_text)


@pytest.fixture(scope="module")
def rendered():
    """CSL-processed plain text: the content the FINAL PDF typesets."""
    if not RENDERED.exists():
        pytest.skip("rendered text not generated")
    return RENDERED.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def bibliography_text(rendered):
    return _flat(rendered.split("\nReferences\n")[-1])


# ---------------------------------------------------------------------------
# artifact validity
# ---------------------------------------------------------------------------


def test_final_docx_is_a_valid_package():
    if not FINAL_DOCX.exists():
        pytest.skip("pandoc FINAL DOCX not built")
    with zipfile.ZipFile(FINAL_DOCX) as archive:
        assert archive.testzip() is None
        assert "word/document.xml" in archive.namelist()


def test_final_pdf_is_a_valid_document():
    if not FINAL_PDF.exists():
        pytest.skip("pandoc FINAL PDF not built")
    raw = FINAL_PDF.read_bytes()
    assert raw.startswith(b"%PDF-")
    assert raw.rstrip().endswith(b"%%EOF")
    assert raw.count(b"/ToUnicode") > 0, "text must be extractable by readers"
    assert int(re.search(rb"/Count\s+(\d+)", raw).group(1)) > 20


def test_supplementary_final_artifacts_exist():
    for path in (SI_FINAL_DOCX,
                 SUB / "supplementary" / "MolFusion_Supplementary_Information_FINAL.pdf"):
        assert path.exists(), f"missing {path.name}"


def test_review_builds_were_not_overwritten():
    """The 6C.6 zero-dependency builds remain as separate review artifacts."""
    for path in (SUB / "MolFusion_JCheminformatics_Manuscript.docx",
                 SUB / "MolFusion_JCheminformatics_Manuscript.pdf"):
        assert path.exists(), f"{path.name} was overwritten by the FINAL build"


# ---------------------------------------------------------------------------
# scientific values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [
    "22 ADMET endpoints", "217-dimensional", "1.91", "1.77",
    "[1.45, 2.41]", "[1.32, 2.27]", "0.079", "35.4%", "30.7%",
    "9 remained significant", "preserved in all 11",
])
def test_final_docx_carries_required_values(flat_docx, value):
    assert value in flat_docx


def test_final_ci_guard(flat_docx, rendered):
    assert "2.47" not in flat_docx
    assert "2.47" not in rendered
    assert "[1.45, 2.41]" in flat_docx and "[1.32, 2.27]" in flat_docx


def test_final_compute_denominator_guard(flat_docx, rendered):
    for body in (flat_docx, _flat(rendered)):
        assert "35.4%" in body and "30.7%" in body
        assert "29.8%" not in body


def test_final_regression_guard(flat_docx):
    assert "0.079" in flat_docx and "did not reject" in flat_docx
    for phrase in ("significantly outperformed on regression",
                   "significant nonlinear regression superiority",
                   "superior on regression endpoints"):
        assert phrase.lower() not in flat_docx.lower()


def test_final_linear_probe_guard(flat_docx):
    for phrase in ("the linear probe found no differences",
                   "no representation differences",
                   "all representations were equivalent"):
        assert phrase.lower() not in flat_docx.lower()


@pytest.mark.parametrize("phrase", [
    "external validation", "external cohort", "validation cohort",
    "state-of-the-art", "best molecular representation", "universally superior",
    "data leakage", "contamination", "contains more information",
])
def test_final_prohibited_wording(flat_docx, phrase):
    assert not _unnegated(flat_docx, phrase)


def test_final_c9_guard(rendered):
    abstract = _flat(re.search(r"^Abstract$(.*?)^Introduction$", rendered,
                               re.S | re.M).group(1)).lower()
    conclusion = _flat(re.search(r"^Conclusions$(.*?)^Figure captions$", rendered,
                                 re.S | re.M).group(1)).lower()
    for token in ("chembl", "corpus exposure", "overlap"):
        assert token not in abstract
        assert token not in conclusion


@pytest.mark.parametrize("endpoint", publication.PRE_REGISTERED_LOW_STABILITY)
def test_final_low_stability_guard(flat_docx, endpoint):
    for sentence in re.split(r"(?<=[.])\s+", flat_docx):
        if endpoint in sentence:
            for word in ("best", "worst", "outperform", "superior", "leader",
                         "strongest"):
                assert word not in sentence.lower()


def test_final_vdss_lombardo_guard(flat_docx):
    assert "vdss_lombardo" in flat_docx and "BORDERLINE" in flat_docx
    assert "vdss_lombardo" not in publication.PRE_REGISTERED_LOW_STABILITY


def test_final_provenance_guard(flat_docx):
    assert "did not produce the results reported" in flat_docx
    assert "predictive accessibility" in flat_docx


# ---------------------------------------------------------------------------
# Unicode: the journal build must NOT transliterate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codepoint,name", [
    (0x03C7, "chi"), (0x03B1, "alpha"), (0x2264, "less-than-or-equal"),
    (0x00B1, "plus-minus"), (0x00D7, "multiplication sign"),
    (0x00B2, "superscript two"), (0x2192, "rightwards arrow"),
    (0x2013, "en dash"), (0x00A7, "section sign"),
])
def test_final_docx_preserves_unicode(docx_text, codepoint, name):
    assert chr(codepoint) in docx_text, f"{name} lost in the journal build"


def test_final_docx_is_not_transliterated(flat_docx):
    """The review-PDF ASCII fallbacks must not leak into the journal build."""
    assert "chi^2" not in flat_docx
    assert "alpha = 0.05" not in flat_docx


def test_rendered_text_preserves_unicode(rendered):
    for codepoint in (0x03C7, 0x03B1, 0x2264, 0x00D7, 0x00B2):
        assert chr(codepoint) in rendered


# ---------------------------------------------------------------------------
# references, as the CSL rendered them
# ---------------------------------------------------------------------------


def test_csl_produced_numeric_citations(rendered):
    body = _flat(rendered).split("References")[0]
    assert re.search(r"\[\d+\]", body)
    assert "[@" not in body, "unresolved BibTeX keys survived into the output"


def test_rendered_bibliography_is_complete_and_sequential(rendered):
    refs = rendered.split("\nReferences\n")[-1]
    numbers = [int(n) for n in re.findall(r"^\s*(\d+)\.\s", refs, re.M)]
    assert numbers == list(range(1, 26))


def test_selfies_reference_is_not_truncated(bibliography_text):
    """An unescaped % started a BibTeX comment and swallowed the title."""
    assert "Self-referencing embedded strings" in bibliography_text
    assert "100% robust molecular string representation" in bibliography_text


def test_bibtex_escapes_comment_characters():
    bib = (SUB / "bibliography.bib").read_text(encoding="utf-8")
    for match in re.finditer(r"title = \{\{(.*?)\}\}", bib, re.S):
        title = match.group(1)
        assert "%" not in title.replace(r"\%", ""), f"unescaped % in {title[:60]}"


def test_rdkit_reference_renders_its_identifier(bibliography_text):
    """The journal style emits no DOI for non-article CSL types."""
    assert "RDKit: Open-source cheminformatics" in bibliography_text
    assert "10.5281/zenodo.591637" in bibliography_text
    assert "Version 2026.03.5" in bibliography_text


def test_no_incorrect_rdkit_release_doi(rendered):
    """2026.03.6 has a Zenodo record; the software actually run was 2026.03.5."""
    assert "22140358" not in rendered


def test_morgan_and_ecfp_render_separately(bibliography_text):
    assert "Unique Machine Description" in bibliography_text
    assert "Extended-Connectivity Fingerprints" in bibliography_text


def test_no_duplicate_rendered_references(rendered):
    refs = rendered.split("\nReferences\n")[-1]
    entries = re.findall(r"^\s*\d+\.\s+(.*?)(?=\n\s*\d+\.\s|\Z)", refs, re.S | re.M)
    fingerprints = [_flat(e)[:70] for e in entries]
    assert len(fingerprints) == len(set(fingerprints))


# ---------------------------------------------------------------------------
# structure and metadata
# ---------------------------------------------------------------------------


def test_rendered_section_order(rendered):
    order = re.findall(
        r"^(Abstract|Introduction|Methods|Results|Discussion|Limitations|"
        r"Conclusions|Figure captions|Table captions|Declarations|References)$",
        rendered, re.M)
    assert order == ["Abstract", "Introduction", "Methods", "Results",
                     "Discussion", "Limitations", "Conclusions",
                     "Figure captions", "Table captions", "Declarations",
                     "References"]


def test_structured_abstract_survived_conversion(rendered):
    block = re.search(r"^Abstract$(.*?)^Introduction$", rendered, re.S | re.M).group(1)
    for label in ("Background:", "Methods:", "Results:"):
        assert label in block
    assert "Keywords:" in block


def test_abstract_word_count_after_conversion(rendered):
    block = re.search(r"^Abstract$(.*?)Keywords:", rendered, re.S | re.M).group(1)
    words = len(_flat(re.sub(r"(Background|Methods|Results):", "", block)).split())
    assert 150 <= words <= submission_cli.ABSTRACT_WORD_LIMIT


def test_final_retains_author_placeholders(flat_docx):
    assert "[AUTHOR INPUT REQUIRED]" in flat_docx


def test_availability_statement_still_names_the_branch(flat_docx):
    """Must not pre-emptively claim a tag or archive that does not exist."""
    assert "develop" in flat_docx
    assert "does not contain this code" in flat_docx
    assert "molfusion-jcheminform-2026-09" not in flat_docx


# ---------------------------------------------------------------------------
# supplementary data exports
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip("supplementary tables not exported")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.mark.parametrize("table", [f"S{i}" for i in range(1, 12)])
def test_supplementary_table_exported(manifest, table):
    entry = next((t for t in manifest["tables"] if t["table"] == table), None)
    assert entry is not None, f"{table} missing from the manifest"
    assert (SI_TABLES / entry["filename"]).exists()
    assert entry["rows"] > 0 and entry["columns"] > 0
    assert len(entry["sha256"]) == 64


def test_manifest_hashes_match_the_files(manifest):
    for entry in manifest["tables"]:
        digest = hashlib.sha256((SI_TABLES / entry["filename"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"{entry['filename']} digest drifted"


def test_supplementary_export_is_deterministic(manifest):
    from molfusion_backend.benchmark import supplementary_cli

    before = [t["sha256"] for t in manifest["tables"]]
    # --root keeps the exporter independent of the test working directory.
    supplementary_cli.main(["--out", str(SI_TABLES), "--root", ".."])
    after = [t["sha256"] for t in
             json.loads(MANIFEST.read_text(encoding="utf-8"))["tables"]]
    assert before == after


def test_supplementary_row_counts_match_the_frozen_analysis(manifest):
    rows = {t["table"]: t["rows"] for t in manifest["tables"]}
    assert rows["S1"] == 126     # all Track A2 pairwise contrasts
    assert rows["S2"] == 105     # all Track A1 pairwise contrasts
    assert rows["S3"] == 11      # A1 Holm-significant contrasts
    assert rows["S4"] == 616     # endpoint ranks, both tracks
    assert rows["S7"] == 22      # endpoints
    assert rows["S8"] == 12      # six families x two tracks
    assert rows["S10"] == 22
    assert rows["S11"] == 28


def test_manifest_records_its_evidence_identity(manifest):
    assert manifest["publication_evidence_identity"] == (
        "5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18")
    assert "recomputed" in manifest["note"]


def test_provenance_export_separates_recorded_from_reconstructed():
    path = SI_TABLES / "supplementary_table_S9_provenance_audit.csv"
    if not path.exists():
        pytest.skip("supplementary tables not exported")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert sum(int(r["shards_without_recorded_commit"]) for r in rows) == 338
    for row in rows:
        assert row["backfilled_into_shards"] in ("False", "false")
        assert row["reconstructed_execution_commits"]


# ---------------------------------------------------------------------------
# release plan: prepared, not executed
# ---------------------------------------------------------------------------


def test_release_plan_exists():
    plan = SUB / "metadata" / "RELEASE_PLAN.md"
    assert plan.exists()
    body = plan.read_text(encoding="utf-8")
    assert "molfusion-jcheminform-2026-09" in body
    assert "No Git tag, GitHub release or Zenodo archive was created" in body


def test_no_tag_was_created():
    result = subprocess.run(["git", "tag", "--list"], capture_output=True,
                            text=True, cwd="..")
    assert "molfusion" not in result.stdout.lower(), (
        "a tag was created without authorisation")


def test_author_metadata_still_unresolved():
    body = (SUB / "metadata" / "AUTHOR_METADATA.md").read_text(encoding="utf-8")
    assert body.count("[AUTHOR INPUT REQUIRED]") >= 10
    declarations = (SUB / "metadata" / "declarations.md").read_text(encoding="utf-8")
    for field in ("Competing interests", "Funding", "Authors' contributions"):
        section = declarations.split(field)[1][:200]
        assert "[AUTHOR INPUT REQUIRED]" in section
