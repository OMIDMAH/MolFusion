"""Assemble the manuscript deterministically from its section drafts.

The section drafts are the source of truth; the consolidated manuscript is
generated from them. Keeping two hand-maintained copies of the same prose
is how a manuscript acquires contradictions that nobody notices until
review, so the assembled file is a build product and is regenerated rather
than edited.

Two jobs beyond concatenation. Citation placeholders of the form
``[CITATION: key]`` are resolved to numeric references from
``references.json``, so the bibliography and the in-text markers cannot
drift apart. And the bibliography itself is rendered from the same JSON, so
a reference that is cited but absent -- or present but never cited -- is
detectable rather than merely unlikely.

    .\\backend\\.venv\\Scripts\\python.exe -m molfusion_backend.benchmark.manuscript_cli
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DOCS = Path("docs/manuscript")

#: Section drafts, in manuscript order. Each is spliced in at the marker
#: named beside it; text outside the marked span is drafting scaffolding
#: (evidence-identity headers and so on) and is deliberately not carried
#: into the assembled manuscript.
SECTIONS = (
    ("INTRODUCTION_DRAFT.md", r"^# 1\. Introduction$", None),
    ("METHODS_DRAFT.md", r"^## 2\.1 ", r"^---\s*$\n\Z"),
    ("RESULTS_DRAFT.md", r"^## 3\.1 ", None),
    ("DISCUSSION_DRAFT.md", r"^# 4\. Discussion$", None),
)

PLACEHOLDER = re.compile(r"\[CITATION: ([^\]]+)\]")


def load_references(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_reference(entry: dict[str, Any]) -> str:
    """One bibliography line, in a neutral numbered style."""
    authors = ", ".join(entry["authors"])
    parts = [f"{authors}. {entry['title']}."]
    container = entry.get("container")
    if container and container != "Software":
        parts.append(f"*{container}*")
    if entry.get("year"):
        parts.append(f"{entry['year']}")
    if entry.get("volume"):
        volume = entry["volume"]
        if entry.get("issue"):
            volume += f"({entry['issue']})"
        parts.append(volume)
    if entry.get("pages"):
        parts.append(entry["pages"])
    line = ", ".join(p for p in parts[1:] if p)
    text = parts[0] + (" " + line if line else "")
    tail = []
    if entry.get("doi"):
        tail.append(f"doi:{entry['doi']}")
    if entry.get("pmid"):
        tail.append(f"PMID:{entry['pmid']}")
    if entry.get("arxiv"):
        tail.append(f"arXiv:{entry['arxiv']}")
    if entry.get("isbn"):
        tail.append(f"ISBN {entry['isbn']}")
    if entry.get("jstor"):
        tail.append(f"JSTOR {entry['jstor']}")
    if entry.get("url") and not entry.get("doi"):
        tail.append(entry["url"])
    if tail:
        text += ". " + ". ".join(tail)
    return text.rstrip(".") + "."


def resolve_citations(text: str, resolution: dict[str, list[int]],
                      cited: set[int]) -> tuple[str, list[str]]:
    """Replace every ``[CITATION: key]`` with its numeric reference marker."""
    unresolved: list[str] = []

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        # Placeholders wrap across lines in the drafts, and "Bemis-Murcko"
        # is written with an en dash; normalise both before lookup.
        lookup = re.sub(r"\s+", " ", key).replace("–", "-").replace("’", "'")
        numbers = resolution.get(lookup)
        if numbers is None:
            unresolved.append(key)
            return match.group(0)
        cited.update(numbers)
        return "[" + ", ".join(str(n) for n in numbers) + "]"

    return PLACEHOLDER.sub(replace, text), unresolved


def extract(path: Path, start: str, end: str | None) -> str:
    body = path.read_text(encoding="utf-8")
    match = re.search(start, body, re.M)
    if not match:
        raise SystemExit(f"{path.name}: start marker {start!r} not found")
    text = body[match.start():]
    if end:
        stop = re.search(end, text, re.M)
        if stop:
            text = text[: stop.start()]
    return text.rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    docs = args.docs
    out = args.out or docs / "MANUSCRIPT_DRAFT.md"

    bibliography = load_references(docs / "references.json")
    resolution = bibliography["placeholder_resolution"]
    entries = {e["id"]: e for e in bibliography["references"]}
    cited: set[int] = set()

    header = (docs / "TITLE_AND_ABSTRACT.md").read_text(encoding="utf-8")
    title = re.search(r"^\*\*Recommended: (T\d+) — \*(.+?)\*\*\*$", header, re.M)
    if not title:
        title_line = "MolFusion: Probe-Dependent Performance of Molecular Representations Across 22 ADMET Endpoints"
    else:
        title_line = title.group(2).replace("*", "")
    abstract = re.search(r"^## 4\. Abstract[^\n]*$\n\n(.*?)\n\n### Numerical",
                         header, re.S | re.M)
    if not abstract:
        raise SystemExit("abstract block not found in TITLE_AND_ABSTRACT.md")
    abstract_text = re.sub(r"^> ?", "", abstract.group(1), flags=re.M)
    keywords = re.search(r"^## 5\. Keywords.*?```\n(.*?)```", header, re.S | re.M)
    keyword_line = "; ".join(
        k.strip() for k in keywords.group(1).strip().splitlines()) if keywords else ""

    chunks: list[str] = [
        f"# {title_line}\n",
        "> Generated by `molfusion_backend.benchmark.manuscript_cli` from the "
        "section drafts in `docs/manuscript/`. Edit the drafts, not this file.\n",
        "## Abstract\n\n" + abstract_text.strip() + "\n",
        "## Keywords\n\n" + keyword_line + "\n",
    ]

    for name, start, end in SECTIONS:
        chunks.append(extract(docs / name, start, end))

    body = "\n---\n\n".join(chunk.rstrip() + "\n" for chunk in chunks)
    body, unresolved = resolve_citations(body, resolution, cited)

    captions = []
    for name in ("FIGURE_CAPTIONS.md", "TABLE_CAPTIONS.md"):
        path = docs / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            text = re.sub(r"\A.*?^(## )", r"\1", text, count=1, flags=re.S | re.M)
            resolved, more = resolve_citations(text, resolution, cited)
            unresolved += more
            captions.append(resolved.rstrip() + "\n")

    lines = ["## References\n"]
    for number in sorted(entries):
        marker = "" if number in cited else "  <!-- not cited in main text -->"
        lines.append(f"{number}. {format_reference(entries[number])}{marker}")
    references_block = "\n".join(lines) + "\n"

    document = body + "\n---\n\n" + references_block
    if captions:
        document += "\n---\n\n" + "\n---\n\n".join(captions)

    out.write_text(document, encoding="utf-8", newline="\n")

    # Standalone bibliography, generated from the same JSON so it cannot
    # drift from the copy embedded in the assembled manuscript.
    standalone = [
        "# References",
        "",
        f"Generated from `references.json` by `manuscript_cli`. "
        f"{len(entries)} references, verified {bibliography['verified_on']}.",
        "",
        bibliography["verification_policy"],
        "",
        f"Numbering: {bibliography['numbering']}.",
        "",
    ]
    for number in sorted(entries):
        entry = entries[number]
        standalone.append(f"{number}. {format_reference(entry)}")
        standalone.append(f"   - *type*: {entry['type']}")
        standalone.append(f"   - *verified via*: {entry['verified_via']}")
        standalone.append(f"   - *supports*: {entry['supports']}")
        standalone.append("")
    (docs / "REFERENCES.md").write_text(
        "\n".join(standalone).rstrip() + "\n", encoding="utf-8", newline="\n")

    words = len(re.sub(r"[#*`>|\[\]]", " ", re.sub(r"^\|.*$", "", document, flags=re.M)).split())
    print(f"wrote {out}")
    print(f"  words           {words}")
    print(f"  references      {len(entries)} defined, {len(cited)} cited")
    uncited = sorted(set(entries) - cited)
    if uncited:
        print(f"  UNCITED         {uncited}")
    if unresolved:
        print(f"  UNRESOLVED      {sorted(set(unresolved))}")
        return 1
    print("  placeholders    0 unresolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
