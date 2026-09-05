"""Build the Journal of Cheminformatics submission package.

Deterministic, and generated from the frozen manuscript sources rather than
hand-written, for the same reason the assembled manuscript is: a
journal-specific copy that is edited by hand becomes a second source of
truth, and the two drift.

What this produces:

``article.md``
    The manuscript in the journal's own Markdown layout -- YAML header,
    structured Background/Methods/Results abstract, keywords inside the
    abstract block, and the Declarations subsections the template requires.
    Numeric citation markers from the assembled manuscript are converted
    back to BibTeX keys so pandoc and the journal CSL can renumber them;
    the manuscript prose therefore never carries a hard-coded reference
    number.

``bibliography.bib``
    Generated from ``references.json``, the same source the Markdown
    bibliography comes from.

``supplementary/supplementary_information.md``
    Supplementary Methods, the material moved out of the main text, and the
    reproducibility checklist.

``article.docx``
    Written directly as Office Open XML. Pandoc is the journal's own route
    and the Makefile emitted here drives it, but pandoc is not installed in
    this environment and the project adds dependencies only after
    individual approval -- so a zero-dependency writer produces a valid
    .docx today, and the pandoc path remains available unchanged.
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

DOCS = Path("docs/manuscript")
OUT = DOCS / "submission" / "jcheminformatics"

#: Journal limits, read from the official template and guidelines.
ABSTRACT_WORD_LIMIT = 350
GRAPHICAL_ABSTRACT = "920x300 px, max 150 KB, jpeg/png/svg, optional"

#: Main text keeps Tables 1-5; 6 and 7 move to Supplementary, which is the
#: brief's recommended default and keeps the main text readable.
MAIN_TABLES = (1, 2, 3, 4, 5)
SUPPLEMENTARY_TABLES = (6, 7)


# ---------------------------------------------------------------------------
# BibTeX
# ---------------------------------------------------------------------------


def _bib_authors(authors: list[str]) -> str:
    return " and ".join(authors)


def bibtex_entry(entry: dict[str, Any]) -> str:
    kind = {"book": "book", "software": "misc"}.get(entry["type"], "article")
    fields: list[tuple[str, str]] = [
        ("author", _bib_authors(entry["authors"])),
        ("title", "{" + entry["title"] + "}"),
    ]
    container = entry.get("container")
    if container and container != "Software":
        fields.append(("publisher" if kind == "book" else "journal", container))
    if entry.get("year"):
        fields.append(("year", str(entry["year"])))
    for name in ("volume", "issue", "pages"):
        if entry.get(name):
            fields.append(("number" if name == "issue" else name, entry[name]))
    if entry.get("doi"):
        fields.append(("doi", entry["doi"]))
    if entry.get("isbn"):
        fields.append(("isbn", entry["isbn"]))
    if entry.get("url"):
        fields.append(("url", entry["url"]))
    if entry.get("arxiv"):
        fields.append(("note", f"arXiv:{entry['arxiv']}"))
    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields)
    return f"@{kind}{{{entry['key']},\n  {body}\n}}\n"


# ---------------------------------------------------------------------------
# minimal Office Open XML writer
# ---------------------------------------------------------------------------


_XML_ESCAPES = ((("&"), "&amp;"), ("<", "&lt;"), (">", "&gt;"))


def _xe(text: str) -> str:
    for old, new in _XML_ESCAPES:
        text = text.replace(old, new)
    return text


def _runs(text: str) -> str:
    """Inline markdown to Word runs: bold, italic and code become runs."""
    parts: list[str] = []
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    for token in pattern.split(text):
        if not token:
            continue
        props = ""
        body = token
        if token.startswith("**") and token.endswith("**"):
            props, body = "<w:b/>", token[2:-2]
        elif token.startswith("*") and token.endswith("*"):
            props, body = "<w:i/>", token[1:-1]
        elif token.startswith("`") and token.endswith("`"):
            props, body = '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>', token[1:-1]
        rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
        parts.append(
            f'<w:r>{rpr}<w:t xml:space="preserve">{_xe(body)}</w:t></w:r>')
    return "".join(parts) or '<w:r><w:t xml:space="preserve"></w:t></w:r>'


def _paragraph(text: str, style: str | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{_runs(text)}</w:p>"


def _table(rows: list[list[str]]) -> str:
    grid = "".join('<w:gridCol w:w="1200"/>' for _ in rows[0])
    body = []
    for index, row in enumerate(rows):
        cells = []
        for cell in row:
            content = f"**{cell}**" if index == 0 else cell
            cells.append(
                '<w:tc><w:tcPr><w:tcW w:w="1200" w:type="dxa"/></w:tcPr>'
                f"{_paragraph(content)}</w:tc>")
        body.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return (
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders>'
        + "".join(f'<w:{s} w:val="single" w:sz="4" w:color="999999"/>'
                  for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        + "</w:tblBorders></w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>{''.join(body)}</w:tbl>"
        + _paragraph(""))


def markdown_to_docx_body(markdown: str) -> str:
    """Headings, paragraphs, lists, tables and blockquotes to WordprocessingML."""
    out: list[str] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()

        if not line.strip() or set(line.strip()) == {"-"} and len(line.strip()) > 2:
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = min(len(heading.group(1)), 4)
            out.append(_paragraph(heading.group(2).strip(), f"Heading{level}"))
            index += 1
            continue

        if line.lstrip().startswith("|"):
            block: list[str] = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                block.append(lines[index].strip())
                index += 1
            rows = []
            for row in block:
                cells = [c.strip() for c in row.strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in cells):
                    continue
                rows.append(cells)
            if rows:
                out.append(_table(rows))
            continue

        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            out.append(_paragraph(bullet.group(1), "ListParagraph"))
            index += 1
            continue

        numbered = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if numbered:
            out.append(_paragraph(numbered.group(1), "ListParagraph"))
            index += 1
            continue

        if line.startswith(">"):
            quote: list[str] = []
            while index < len(lines) and lines[index].startswith(">"):
                quote.append(lines[index].lstrip("> ").rstrip())
                index += 1
            out.append(_paragraph(" ".join(quote), "Quote"))
            continue

        # ordinary paragraph: join wrapped lines
        para: list[str] = []
        while index < len(lines) and lines[index].strip() and not re.match(
                r"^(#{1,6}\s|\s*[-*]\s|\s*\d+\.\s|\||>)", lines[index]) and not (
                set(lines[index].strip()) == {"-"} and len(lines[index].strip()) > 2):
            para.append(lines[index].strip())
            index += 1
        if para:
            out.append(_paragraph(" ".join(para)))
    return "".join(out)


_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
<w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
<w:pPr><w:outlineLvl w:val="0"/><w:spacing w:before="360" w:after="120"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
<w:pPr><w:outlineLvl w:val="1"/><w:spacing w:before="280" w:after="100"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>
<w:pPr><w:outlineLvl w:val="2"/><w:spacing w:before="240" w:after="80"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading4"><w:name w:val="heading 4"/>
<w:pPr><w:outlineLvl w:val="3"/><w:spacing w:before="200" w:after="80"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/>
<w:pPr><w:ind w:left="480"/><w:spacing w:after="60"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Quote"><w:name w:val="Quote"/>
<w:pPr><w:ind w:left="480" w:right="480"/><w:spacing w:after="120"/></w:pPr>
<w:rPr><w:i/></w:rPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/></w:style>
</w:styles>
"""

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""


def write_docx(path: Path, markdown: str) -> None:
    """Write a valid .docx. Deterministic: fixed timestamps, fixed order."""
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{markdown_to_docx_body(markdown)}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
        "</w:sectPr></w:body></w:document>"
    )
    parts = (
        ("[Content_Types].xml", _CONTENT_TYPES),
        ("_rels/.rels", _ROOT_RELS),
        ("word/document.xml", document),
        ("word/styles.xml", _STYLES),
        ("word/_rels/document.xml.rels", _DOC_RELS),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)


def read_docx_text(path: Path) -> str:
    """Extract visible text, for auditing what the .docx actually says."""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"</w:tc>", " | ", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    return (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))


# ---------------------------------------------------------------------------
# journal Markdown source
# ---------------------------------------------------------------------------


def numeric_to_keys(text: str, entries: dict[int, dict[str, Any]]) -> str:
    """Turn ``[3, 4]`` back into ``[@key1; @key2]`` for pandoc/CSL."""
    def replace(match: re.Match[str]) -> str:
        numbers = [int(n) for n in match.group(1).split(", ")]
        if not all(n in entries for n in numbers):
            return match.group(0)
        return "[" + "; ".join(f"@{entries[n]['key']}" for n in numbers) + "]"

    return re.sub(r"\[(\d+(?:, \d+)*)\]", replace, text)


def structured_abstract(plain_abstract: str) -> str:
    """Split the frozen abstract into the journal's Background/Methods/Results.

    The same approved sentences, redistributed under the required headings.
    No sentence is added, removed or reworded, so the abstract remains
    governed by C1, C2 and C3 alone.
    """
    sentences = re.split(r"(?<=\.)\s+", plain_abstract.strip())
    assert len(sentences) >= 6, "unexpected abstract shape"
    background = " ".join(sentences[0:2])
    methods = " ".join(sentences[2:3])
    results = " ".join(sentences[3:])
    return (
        f"**Background:** {background}\n\n"
        f"**Methods:** {methods}\n\n"
        f"**Results:** {results}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    docs, out = args.docs, args.out
    out.mkdir(parents=True, exist_ok=True)

    bibliography = json.loads((docs / "references.json").read_text(encoding="utf-8"))
    entries = {e["id"]: e for e in bibliography["references"]}

    # --- bibliography.bib ----------------------------------------------
    bib = "".join(bibtex_entry(entries[n]) + "\n" for n in sorted(entries))
    (out / "bibliography.bib").write_text(bib, encoding="utf-8", newline="\n")

    # --- source sections -------------------------------------------------
    assembled = (docs / "MANUSCRIPT_DRAFT.md").read_text(encoding="utf-8")
    header = (docs / "TITLE_AND_ABSTRACT.md").read_text(encoding="utf-8")

    title = "MolFusion: Probe-Dependent Performance of Molecular Representations Across 22 ADMET Endpoints"
    abstract_raw = re.search(
        r"^## 4\. Abstract[^\n]*$\n\n(.*?)\n\n### Numerical", header, re.S | re.M).group(1)
    abstract_plain = re.sub(r"\s+", " ", re.sub(r"^> ?", "", abstract_raw, flags=re.M)).strip()
    keywords = re.search(r"^## 5\. Keywords.*?```\n(.*?)```", header, re.S | re.M)
    keyword_line = "; ".join(k.strip() for k in keywords.group(1).strip().splitlines())

    def section(pattern: str) -> str:
        match = re.search(pattern, assembled, re.S | re.M)
        return match.group(1).strip() if match else ""

    introduction = section(r"^# 1\. Introduction$(.*?)^---\s*$\n\n## 2\.1 ")
    methods = section(r"^(## 2\.1 .*?)^---\s*$\n\n## 3\.1 ")
    results = section(r"^(## 3\.1 .*?)^# 4\. Discussion$")
    discussion = section(r"^# 4\. Discussion$(.*?)^# 5\. Limitations$")
    limitations = section(r"^# 5\. Limitations$(.*?)^# 6\. Conclusion$")
    conclusion = section(r"^# 6\. Conclusion$(.*?)^---\s*$")
    figure_captions = section(r"^## Figure captions$(.*?)^---\s*$\n\n## Main table captions")
    table_captions_all = section(r"^## Main table captions$(.*?)\Z")

    main_caps, supp_caps = [], []
    for block in re.split(r"\n(?=\*\*(?:Table|Supplementary Table) )", table_captions_all):
        block = block.strip()
        if not block:
            continue
        number = re.match(r"\*\*Table (\d+)\.", block)
        if number and int(number.group(1)) in MAIN_TABLES:
            main_caps.append(block)
        else:
            supp_caps.append(block)

    declarations = (docs / "submission" / "jcheminformatics" / "metadata"
                    / "declarations.md")
    declarations_text = (declarations.read_text(encoding="utf-8")
                         if declarations.exists() else "[AUTHOR INPUT REQUIRED]")

    article = "\n".join([
        "---",
        f'title: "{title}"',
        "author:",
        "- '[AUTHOR INPUT REQUIRED] [1,*,orcid:AUTHOR-INPUT-REQUIRED]'",
        "link-citations: yes",
        "bibliography: bibliography.bib",
        "csl: journal-of-cheminformatics.csl",
        "...",
        "",
        "\\* Correspondence: [AUTHOR INPUT REQUIRED]",
        "",
        "# Abstract",
        "",
        structured_abstract(abstract_plain),
        f"**Keywords:** {keyword_line}",
        "",
        "<!-- Graphical abstract optional: 920 x 300 px, max 150 KB, jpeg/png/svg. "
        "Specification in metadata/GRAPHICAL_ABSTRACT_SPEC.md; artwork not generated. -->",
        "",
        "# Introduction",
        "", introduction, "",
        "# Methods",
        "", methods, "",
        "# Results",
        "", results, "",
        "# Discussion",
        "", discussion, "",
        "# Limitations",
        "", limitations, "",
        "# Conclusions",
        "", conclusion, "",
        "# Figure captions",
        "", figure_captions, "",
        "# Table captions",
        "", "\n\n".join(main_caps), "",
        "# Declarations",
        "", declarations_text, "",
        "# References",
        "",
    ])
    article = numeric_to_keys(article, entries)
    article = re.sub(r"^## (\d)\.(\d+) ", r"## ", article, flags=re.M)
    (out / "article.md").write_text(article, encoding="utf-8", newline="\n")

    # --- DOCX -------------------------------------------------------------
    bibliography_md = ["", "# References", ""]
    from molfusion_backend.benchmark.manuscript_cli import format_reference
    for number in sorted(entries):
        bibliography_md.append(f"{number}. {format_reference(entries[number])}")
    docx_source = article.split("# References")[0] + "\n".join(bibliography_md)
    docx_source = re.sub(r"^---\n.*?^\.\.\.\n", "", docx_source, flags=re.S | re.M)
    docx_source = f"# {title}\n\n" + docx_source
    write_docx(out / "MolFusion_JCheminformatics_Manuscript.docx", docx_source)
    write_pdf(out / "MolFusion_JCheminformatics_Manuscript.pdf", docx_source)

    supplementary = out / "supplementary"
    si_path = supplementary / "supplementary_information.md"
    if si_path.exists():
        si_text = si_path.read_text(encoding="utf-8")
        si_text += "\n\n# Supplementary table captions\n\n" + "\n\n".join(supp_caps) + "\n"
        write_docx(supplementary / "MolFusion_Supplementary_Information.docx", si_text)
        write_pdf(supplementary / "MolFusion_Supplementary_Information.pdf", si_text)

    words = len(re.sub(r"[#*`>|\[\]]", " ",
                       re.sub(r"^\|.*$", "", article, flags=re.M)).split())
    abstract_words = len(re.sub(r"[*:]", " ", structured_abstract(abstract_plain)).split()) - 3
    print(f"wrote {out / 'article.md'}")
    print(f"  main-text words     {words}")
    print(f"  abstract words      {abstract_words} (limit {ABSTRACT_WORD_LIMIT})")
    print(f"  keywords            {len(keyword_line.split(';'))}")
    print(f"  bibtex entries      {len(entries)}")
    print(f"  main table captions {len(main_caps)}  supplementary {len(supp_caps)}")
    remaining = re.findall(r"\[CITATION: [^\]]+\]", article)
    print(f"  unresolved citation placeholders {len(remaining)}")
    return 1 if remaining or abstract_words > ABSTRACT_WORD_LIMIT else 0




# ---------------------------------------------------------------------------
# minimal PDF writer
# ---------------------------------------------------------------------------
#
# A *review* PDF, not a typeset one. Pandoc plus the journal template is the
# route to a submission-quality PDF and the emitted Makefile drives it; this
# writer exists so that a reviewable, auditable PDF with the same scientific
# content is available in an environment where pandoc is not installed.
# Tables are rendered as pipe-delimited text rather than ruled grids.

_PAGE_W, _PAGE_H = 595, 842          # A4 in points
_MARGIN, _LEADING = 56, 13.2
_BODY_SIZE = 9.5


#: Characters the manuscript actually uses that WinAnsi cannot encode.
#: Dropping them to "?" would render chi-squared as "?2" and alpha as "?",
#: which is worse than a plain-text spelling, so each is transliterated.
_PDF_TRANSLITERATE = {
    "α": "alpha", "χ": "chi",
    "–": "-", "—": "--", "−": "-",
    "→": "->", "∈": "in", "∞": "inf",
    "≤": "<=", "≥": ">=",
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3",
    "⁴": "^4", "⁵": "^5", "⁶": "^6", "⁷": "^7",
    "⁸": "^8", "⁹": "^9", "⁻": "^-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ",
}

#: Encodable directly by WinAnsiEncoding, so they survive as themselves.
_PDF_WINANSI_OK = set("§±×éèüöä")


def _pdf_escape(text: str) -> str:
    for source, target in _PDF_TRANSLITERATE.items():
        text = text.replace(source, target)
    out = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return "".join(
        c if ord(c) < 128 or c in _PDF_WINANSI_OK else "?" for c in out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _pdf_lines(markdown: str) -> list[tuple[str, float, bool]]:
    """(text, size, bold) per rendered line."""
    out: list[tuple[str, float, bool]] = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if not line.strip():
            out.append(("", _BODY_SIZE, False))
            continue
        if set(line.strip()) == {"-"} and len(line.strip()) > 2:
            out.append(("", _BODY_SIZE, False))
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            size = {1: 15.0, 2: 12.5, 3: 11.0}.get(level, 10.0)
            out.append(("", _BODY_SIZE, False))
            for part in _wrap(heading.group(2).strip(), int(92 * _BODY_SIZE / size)):
                out.append((part, size, True))
            continue
        clean = re.sub(r"\*\*|\*|`", "", line).replace(">", "").strip()
        for source, target in _PDF_TRANSLITERATE.items():
            clean = clean.replace(source, target)
        for part in _wrap(clean, 96):
            out.append((part, _BODY_SIZE, False))
    return out


def write_pdf(path: Path, markdown: str) -> None:
    """Paginated text PDF. Deterministic: no timestamps, fixed layout."""
    lines = _pdf_lines(markdown)
    usable = _PAGE_H - 2 * _MARGIN
    per_page = int(usable / _LEADING)
    pages: list[list[tuple[str, float, bool]]] = [
        lines[i:i + per_page] for i in range(0, len(lines), per_page)] or [[]]

    streams: list[bytes] = []
    for page in pages:
        parts = ["BT"]
        y = _PAGE_H - _MARGIN
        for text, size, bold in page:
            font = "/F2" if bold else "/F1"
            parts.append(f"1 0 0 1 {_MARGIN} {y:.1f} Tm")
            parts.append(f"{font} {size} Tf")
            parts.append(f"({_pdf_escape(text)}) Tj")
            y -= _LEADING
        parts.append("ET")
        streams.append("\n".join(parts).encode("latin-1", "replace"))

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    resources = (f"<< /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> >>"
                 ).encode()

    pages_id = len(objects) + 1 + 2 * len(streams) + 1
    page_ids: list[int] = []
    for stream in streams:
        content_id = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream")
        page_ids.append(add(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {_PAGE_W} {_PAGE_H}]"
            f" /Resources ".encode() + resources
            + f" /Contents {content_id} 0 R >>".encode()))
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    actual_pages_id = add(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    catalog = add(f"<< /Type /Catalog /Pages {actual_pages_id} 0 R >>".encode())

    buffer = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(buffer)
    buffer += f"xref\n0 {len(objects) + 1}\n".encode()
    buffer += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        buffer += f"{offset:010d} 00000 n \n".encode()
    buffer += (f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
               f"startxref\n{xref}\n%%EOF\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buffer))


def read_pdf_text(path: Path) -> str:
    """Extract the text this writer emitted, for auditing."""
    import zlib

    data = path.read_bytes()
    out: list[str] = []
    for match in re.finditer(rb"stream\n(.*?)\nendstream", data, re.S):
        chunk = match.group(1)
        try:
            chunk = zlib.decompress(chunk)
        except Exception:
            pass
        for line in re.findall(rb"\((.*?)\) Tj", chunk, re.S):
            text = line.decode("latin-1")
            text = text.replace(r"\(", "(").replace(r"\)", ")").replace("\\\\", "\\")
            out.append(text)
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
