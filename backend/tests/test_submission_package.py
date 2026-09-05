"""Phase 6C.6: the Journal of Cheminformatics submission package.

The manuscript guards have to survive format conversion, because that is
where they are easiest to lose: a value can be correct in Markdown and
mangled in the .docx, and nobody reads 37 pages of PDF closely enough to
notice. So every guard is re-run against the *extracted text* of the
generated artifacts rather than against the source that produced them.

Also checked here: that the conversion is deterministic, that the package
does not silently invent author metadata, and that the journal's own
structural requirements (structured abstract, 350-word limit, declarations
subsections) are met.
"""

import json
import re
import zipfile
from pathlib import Path

import pytest

from molfusion_backend.benchmark import publication, submission_cli

DOCS = Path("../docs/manuscript")
SUB = DOCS / "submission" / "jcheminformatics"

ARTICLE = SUB / "article.md"
DOCX = SUB / "MolFusion_JCheminformatics_Manuscript.docx"
PDF = SUB / "MolFusion_JCheminformatics_Manuscript.pdf"
SI_MD = SUB / "supplementary" / "supplementary_information.md"
SI_DOCX = SUB / "supplementary" / "MolFusion_Supplementary_Information.docx"
COVER = SUB / "COVER_LETTER.md"
BIB = SUB / "bibliography.bib"


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "").replace("`", "").replace("*", ""))


_NEGATION = ("not ", "never", "no ", "rather than", "prohibited", "avoided",
             "without", "does not", "did not", "cannot", "must not", "nowhere")


def _unnegated(body: str, phrase: str, window: int = 220) -> list[str]:
    lowered, target = body.lower(), phrase.lower()
    return [body[max(0, m.start() - 110):m.end() + 50]
            for m in re.finditer(re.escape(target), lowered)
            if not any(n in lowered[max(0, m.start() - window):m.start()]
                       for n in _NEGATION)]


@pytest.fixture(scope="module")
def article():
    if not ARTICLE.exists():
        pytest.skip("submission package not generated")
    return ARTICLE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def docx_text():
    if not DOCX.exists():
        pytest.skip("DOCX not generated")
    return _flat(submission_cli.read_docx_text(DOCX))


@pytest.fixture(scope="module")
def pdf_text():
    if not PDF.exists():
        pytest.skip("PDF not generated")
    return _flat(submission_cli.read_pdf_text(PDF))


@pytest.fixture(scope="module")
def si_text():
    if not SI_MD.exists():
        pytest.skip("SI not generated")
    return _flat(SI_MD.read_text(encoding="utf-8"))


def _artifacts():
    return [p for p in (ARTICLE, SI_MD, COVER,
                        SUB / "metadata" / "declarations.md",
                        SUB / "metadata" / "GRAPHICAL_ABSTRACT_SPEC.md")
            if p.exists()]


# ---------------------------------------------------------------------------
# journal structure
# ---------------------------------------------------------------------------


def test_article_has_the_journal_yaml_header(article):
    assert article.startswith("---\n")
    assert 'csl: journal-of-cheminformatics.csl' in article
    assert "bibliography: bibliography.bib" in article


@pytest.mark.parametrize("heading", [
    "# Abstract", "# Introduction", "# Methods", "# Results",
    "# Discussion", "# Limitations", "# Conclusions",
    "# Figure captions", "# Table captions", "# Declarations", "# References",
])
def test_required_sections_present(article, heading):
    assert re.search(rf"^{re.escape(heading)}$", article, re.M)


def test_abstract_is_structured_as_the_template_requires(article):
    block = re.search(r"^# Abstract$(.*?)^# Introduction$", article, re.S | re.M).group(1)
    for label in ("**Background:**", "**Methods:**", "**Results:**"):
        assert label in block, f"structured abstract missing {label}"
    assert "**Keywords:**" in block


def test_abstract_is_within_the_journal_word_limit(article):
    block = re.search(r"^# Abstract$(.*?)^\*\*Keywords", article, re.S | re.M).group(1)
    words = len(_flat(re.sub(r"\*\*\w+:\*\*", "", block)).split())
    assert words <= submission_cli.ABSTRACT_WORD_LIMIT, f"{words} words"
    assert words >= 150


def test_abstract_content_is_unchanged_from_the_frozen_version(article):
    """Restructured under headings, not rewritten."""
    frozen = _flat(re.search(
        r"^## 4\. Abstract[^\n]*$\n\n(.*?)\n\n### Numerical",
        (DOCS / "TITLE_AND_ABSTRACT.md").read_text(encoding="utf-8"),
        re.S | re.M).group(1).replace(">", ""))
    block = _flat(re.search(r"^# Abstract$(.*?)^\*\*Keywords", article,
                            re.S | re.M).group(1))
    stripped = re.sub(r"(Background|Methods|Results):\s*", "", block).strip()
    assert stripped == frozen.strip(), "abstract prose changed during restructuring"


def test_abstract_carries_only_c1_c2_c3(article):
    block = _flat(re.search(r"^# Abstract$(.*?)^# Introduction$", article,
                            re.S | re.M).group(1)).lower()
    for later in ("chembl", "kendall", "provenance", "compute", "wins fell", "cost"):
        assert later not in block, f"abstract gained '{later}'"


def test_keyword_count(article):
    line = re.search(r"^\*\*Keywords:\*\* (.*)$", article, re.M).group(1)
    keywords = [k.strip() for k in line.split(";") if k.strip()]
    assert 5 <= len(keywords) <= 10, f"{len(keywords)} keywords"
    assert "molecular representation" in keywords


@pytest.mark.parametrize("declaration", [
    "Availability of data and materials", "Availability of code",
    "Competing interests", "Funding", "Authors' contributions",
    "Acknowledgements", "Ethics approval and consent to participate",
    "Consent for publication",
])
def test_required_declarations_present(article, declaration):
    assert declaration in article


# ---------------------------------------------------------------------------
# citations and bibliography
# ---------------------------------------------------------------------------


def test_citations_are_bibtex_keys_not_hard_numbers(article):
    body = article.split("# References")[0]
    assert re.search(r"\[@[a-z0-9]+", body), "no @key citations found"
    leftover = re.findall(r"\[\d+(?:, \d+)*\]", body)
    assert leftover == [], f"hard-coded reference numbers remain: {leftover[:5]}"


def test_no_unresolved_citation_placeholders(article):
    assert re.findall(r"\[CITATION: [^\]]+\]", article) == []


def test_bibliography_covers_every_cited_key(article):
    if not BIB.exists():
        pytest.skip("bibliography not generated")
    bib = BIB.read_text(encoding="utf-8")
    defined = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    cited = set()
    for group in re.findall(r"\[([^\]]*@[^\]]*)\]", article.split("# References")[0]):
        cited.update(re.findall(r"@([a-z0-9]+)", group))
    assert cited, "no citations found"
    assert cited <= defined, f"cited but not in .bib: {sorted(cited - defined)}"


def test_bibliography_entry_count_matches_references_json():
    if not BIB.exists():
        pytest.skip("bibliography not generated")
    data = json.loads((DOCS / "references.json").read_text(encoding="utf-8"))
    entries = re.findall(r"^@\w+\{", BIB.read_text(encoding="utf-8"), re.M)
    assert len(entries) == len(data["references"]) == 25


def test_rdkit_keeps_the_concept_doi_with_a_documented_reason():
    data = json.loads((DOCS / "references.json").read_text(encoding="utf-8"))
    rdkit = next(e for e in data["references"] if e["key"] == "rdkit")
    assert rdkit["doi"] == "10.5281/zenodo.591637"
    assert "no record for Release_2026_03_5" in rdkit["verified_via"]


def test_morgan_and_ecfp_remain_separate():
    if not BIB.exists():
        pytest.skip("bibliography not generated")
    bib = BIB.read_text(encoding="utf-8")
    assert "morgan1965" in bib and "rogers2010" in bib
    assert "10.1021/c160017a018" in bib and "10.1021/ci100050t" in bib


# ---------------------------------------------------------------------------
# generated artifacts
# ---------------------------------------------------------------------------


def test_docx_is_a_valid_package():
    if not DOCX.exists():
        pytest.skip("DOCX not generated")
    with zipfile.ZipFile(DOCX) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
    assert {"[Content_Types].xml", "_rels/.rels", "word/document.xml",
            "word/styles.xml"} <= names


def test_pdf_is_a_valid_document():
    if not PDF.exists():
        pytest.skip("PDF not generated")
    raw = PDF.read_bytes()
    assert raw.startswith(b"%PDF-")
    assert raw.rstrip().endswith(b"%%EOF")
    assert raw.count(b"/Type /Page ") > 10


def test_supplementary_artifacts_exist():
    for path in (SI_MD, SI_DOCX):
        assert path.exists(), f"missing {path.name}"


def test_conversion_is_deterministic(tmp_path):
    """Same source twice, same scientific content."""
    first, second = tmp_path / "a", tmp_path / "b"
    for target in (first, second):
        target.mkdir()
        (target / "metadata").mkdir()
        (target / "supplementary").mkdir()
        for name in ("declarations.md",):
            source = SUB / "metadata" / name
            if source.exists():
                (target / "metadata" / name).write_text(
                    source.read_text(encoding="utf-8"), encoding="utf-8")
        if SI_MD.exists():
            (target / "supplementary" / SI_MD.name).write_text(
                SI_MD.read_text(encoding="utf-8"), encoding="utf-8")
        submission_cli.main(["--docs", str(DOCS), "--out", str(target)])

    assert (first / "article.md").read_bytes() == (second / "article.md").read_bytes()
    assert (first / "bibliography.bib").read_bytes() == (
        second / "bibliography.bib").read_bytes()
    a = submission_cli.read_docx_text(
        first / "MolFusion_JCheminformatics_Manuscript.docx")
    b = submission_cli.read_docx_text(
        second / "MolFusion_JCheminformatics_Manuscript.docx")
    assert a == b, "DOCX scientific content differs between runs"
    p = submission_cli.read_pdf_text(first / "MolFusion_JCheminformatics_Manuscript.pdf")
    q = submission_cli.read_pdf_text(second / "MolFusion_JCheminformatics_Manuscript.pdf")
    assert p == q, "PDF scientific content differs between runs"


def test_docx_and_pdf_agree_on_scientific_content(docx_text, pdf_text):
    for value in ("22 ADMET endpoints", "217-dimensional", "1.91", "1.77",
                  "[1.45, 2.41]", "[1.32, 2.27]", "0.079", "35.4%", "30.7%"):
        assert value in docx_text, f"DOCX missing {value}"
        assert value in pdf_text, f"PDF missing {value}"


# ---------------------------------------------------------------------------
# guards, re-run against the extracted artifact text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [
    "22 ADMET endpoints", "217-dimensional", "1.91", "1.77",
    "[1.45, 2.41]", "[1.32, 2.27]", "0.079", "35.4%", "30.7%",
    "9 remained significant", "preserved in all 11",
])
def test_docx_carries_the_required_values(docx_text, value):
    assert value in docx_text


def test_ci_guard_across_every_artifact():
    for path in _artifacts() + [DOCX, PDF, SI_DOCX]:
        if not path.exists():
            continue
        if path.suffix == ".docx":
            body = submission_cli.read_docx_text(path)
        elif path.suffix == ".pdf":
            body = submission_cli.read_pdf_text(path)
        else:
            body = path.read_text(encoding="utf-8")
        assert "2.47" not in body, f"{path.name} contains the erroneous 2.47"


def test_compute_denominator_guard_across_every_artifact(docx_text, pdf_text):
    for name, body in (("docx", docx_text), ("pdf", pdf_text)):
        assert "35.4%" in body
        assert "29.8%" not in body, f"{name} pairs 35.4% with the all-model 29.8%"
    for path in _artifacts():
        body = _flat(path.read_text(encoding="utf-8"))
        if "35.4%" in body and "29.8%" in body:
            assert "denominator" in body.lower(), f"{path.name} mixes denominators"


def test_supplementary_may_state_the_all_model_share_with_context(si_text):
    """SI is allowed to be precise, but not to pair the two loosely."""
    if "29.8%" in si_text:
        assert "denominator" in si_text.lower()


@pytest.mark.parametrize("phrase", [
    "the linear probe found no differences", "no representation differences",
    "all representations were equivalent",
])
def test_linear_guard(docx_text, pdf_text, phrase):
    assert phrase.lower() not in docx_text.lower()
    assert phrase.lower() not in pdf_text.lower()


def test_regression_guard(docx_text):
    assert "0.079" in docx_text
    assert "did not reject" in docx_text
    for phrase in ("significantly outperformed on regression",
                   "significant nonlinear regression superiority",
                   "superior on regression endpoints"):
        assert phrase.lower() not in docx_text.lower()


@pytest.mark.parametrize("phrase", [
    "external validation", "external cohort", "independent validation dataset",
    "validation cohort",
])
def test_a2_guard(docx_text, phrase):
    assert not _unnegated(docx_text, phrase)


def test_universal_superiority_guard(docx_text):
    for phrase in ("best molecular representation", "universally superior",
                   "optimal representation", "best representation overall",
                   "state-of-the-art"):
        assert not _unnegated(docx_text, phrase)


def test_information_content_guard(docx_text):
    for phrase in ("contain more information", "contains more molecular information",
                   "intrinsically superior"):
        assert not _unnegated(docx_text, phrase)
    assert "predictive accessibility" in docx_text


def test_c9_guard(article, docx_text):
    abstract = _flat(re.search(r"^# Abstract$(.*?)^# Introduction$", article,
                               re.S | re.M).group(1)).lower()
    conclusion = _flat(re.search(r"^# Conclusions$(.*?)^# Figure captions$", article,
                                 re.S | re.M).group(1)).lower()
    for token in ("chembl", "corpus exposure", "overlap"):
        assert token not in abstract, f"C9 material in abstract: {token}"
        assert token not in conclusion, f"C9 material in conclusion: {token}"
    assert "external unsupervised corpus exposure" in docx_text
    for word in ("data leakage", "label leakage", "contamination"):
        assert not _unnegated(docx_text, word)


@pytest.mark.parametrize("endpoint", publication.PRE_REGISTERED_LOW_STABILITY)
def test_low_stability_guard(docx_text, endpoint):
    for sentence in re.split(r"(?<=[.])\s+", docx_text):
        if endpoint in sentence:
            for word in ("best", "worst", "outperform", "superior", "leader", "strongest"):
                assert word not in sentence.lower(), (
                    f"endpoint-specific '{word}' for {endpoint}")


def test_vdss_lombardo_guard(docx_text):
    assert "vdss_lombardo" in docx_text and "BORDERLINE" in docx_text
    assert "vdss_lombardo" not in publication.PRE_REGISTERED_LOW_STABILITY


def test_provenance_guard(docx_text, si_text):
    assert "did not produce the results reported" in docx_text
    assert "338 of 616" in si_text
    assert "127" in si_text and "211" in si_text
    assert "backfilled or modified" in si_text
    for wrong in ("provenance was complete", "results were reconstructed"):
        assert wrong not in si_text


def test_no_promotional_language(docx_text):
    for word in ("remarkably", "surprisingly", "breakthrough", "best-in-class",
                 "unprecedented"):
        assert word.lower() not in docx_text.lower()


def test_no_external_numerical_comparison(docx_text):
    for name in ("ADMET-AI", "MiniMol", "MapLight", "ChemBERTa"):
        assert name.lower() not in docx_text.lower()


# ---------------------------------------------------------------------------
# author metadata is never invented
# ---------------------------------------------------------------------------


def test_author_fields_are_explicit_placeholders(article):
    assert "[AUTHOR INPUT REQUIRED]" in article
    assert "orcid:AUTHOR-INPUT-REQUIRED" in article


def test_declarations_do_not_invent_funding_or_conflicts(article):
    block = article.split("# Declarations")[1]
    for field in ("Competing interests", "Funding", "Authors' contributions"):
        section = block.split(field)[1][:200]
        assert "[AUTHOR INPUT REQUIRED]" in section, f"{field} was filled in"


def test_code_availability_does_not_claim_the_default_branch(article):
    block = article.split("Availability of code")[1][:1400]
    assert "develop" in block
    assert "does not contain this code" in block
    assert "all data are available on GitHub" not in article


def test_cover_letter_makes_no_prohibited_claim():
    if not COVER.exists():
        pytest.skip("cover letter not written")
    body = _flat(COVER.read_text(encoding="utf-8"))
    for phrase in ("state-of-the-art", "external validation",
                   "universally superior", "best molecular representation",
                   "outperforms"):
        assert not _unnegated(body, phrase), f"cover letter claims '{phrase}'"
    assert "[AUTHOR INPUT REQUIRED]" in body


def test_graphical_abstract_is_specified_not_fabricated():
    spec = SUB / "metadata" / "GRAPHICAL_ABSTRACT_SPEC.md"
    assert spec.exists()
    body = spec.read_text(encoding="utf-8")
    assert "920 × 300" in body or "920 x 300" in body
    assert "not created" in body.lower()
    assert not list(SUB.glob("figures/graphical_abstract.*"))


# ---------------------------------------------------------------------------
# frozen science is untouched
# ---------------------------------------------------------------------------


def test_scientific_identities_unchanged():
    report = Path("benchmark_runs/publication/publication_report.json")
    if not report.exists():
        pytest.skip("publication package not generated")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["publication_identity_sha256"] == (
        "5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18")
    assert payload["inputs"]["a1_raw_identity"] == (
        "d40ef09b398f47914aa51f99fd6a4f5893f7778b50c0cca04404b575632de868")
    assert payload["inputs"]["a2_raw_identity"] == (
        "9dd5dfa6067c8a760b0bb8fb39648f71f662f2fa1bbf4cc5d7cb0cd495a69f14")
    assert len(payload["claim_registry"]) == 11


def test_supplementary_records_the_reproducibility_chain(si_text):
    for item in ("TDC-ADMET-2026-09", "chembl37_token_ngrams_1_3",
                 "rdkit_canonical_isomeric_smiles_v1", "rdkit_smiles_lexer_v1",
                 "459653b", "e6ae297", "15b78a2", "89335dc",
                 "d40ef09b", "9dd5dfa6", "5790359b",
                 "3.11.15", "2026.03.5", "1.9.0"):
        assert item in si_text, f"SI missing reproducibility item {item}"
