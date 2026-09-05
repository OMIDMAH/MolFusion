# Journal compliance report — *Journal of Cheminformatics*, Research article

Checked **2026-09-05** against current journal sources, not against
recollection.

Status key: **PASS** · **FAIL** · **AUTHOR INPUT REQUIRED** ·
**ACTION BEFORE SUBMISSION** · **NOT APPLICABLE**

---

## Sources consulted

| Source | URL | What it provided |
| --- | --- | --- |
| Official Research article Markdown template | `https://github.com/jcheminform/markdown-jcheminf` (`article.Rmd`, updated 2026-01-06) | section order, structured abstract, 350-word limit, keyword placement, declarations list, graphical abstract spec, reproducibility policy |
| Official build configuration | same repository, `Makefile` | pandoc ≥ 2.12 route to DOCX/PDF, CSL and CiTO filters |
| Official citation style | same repository, `journal-of-cheminformatics.csl` (+ `springer-basic-brackets.csl` parent) | numeric bracketed reference style, eISSN 1758-2946 |
| Journal submission guidelines | `https://jcheminf.biomedcentral.com/submission-guidelines` → redirects to `https://link.springer.com/journal/13321/submission-guidelines` | graphical abstract dimensions; data availability requirement |

**Discrepancy recorded.** The BMC-hosted guideline pages now 301-redirect to
Springer Link, and the Springer Link pages require a cookie/authorisation
handshake that could not be completed non-interactively. The
journal-maintained Markdown template repository was therefore used as the
primary structural authority. It is the journal's own artifact, is current
(updated January 2026), and is more specific than the general guideline
pages for the questions that matter here. **The authors should confirm
length limits and any submission-system-specific requirements on the live
Springer page before submitting.**

---

## Requirement-by-requirement

### Article type and structure

| Requirement | Source | Status | Artifact | Remaining action |
| --- | --- | --- | --- | --- |
| Research article type | template | **PASS** | `article.md` | — |
| Section order: Abstract, Introduction, Methods, Results, Discussion, Conclusions, Declarations, References | template `article.Rmd` | **PASS** | `article.md`; Limitations placed between Discussion and Conclusions | — |
| List of abbreviations section | template (optional heading) | **NOT APPLICABLE** | — | no abbreviation list needed; terms defined at first use |

### Abstract and keywords

| Requirement | Source | Status | Artifact | Remaining action |
| --- | --- | --- | --- | --- |
| Structured abstract: **Background**, **Methods**, **Results** | template | **PASS** | `article.md` | — |
| Abstract ≤ 350 words | template: "should include an abstract that does not exceed 350 words" | **PASS** — 220 words | `article.md` | — |
| Keywords inside the abstract block, semicolon-separated | template | **PASS** — 8 keywords | `article.md` | — |
| No new claims introduced by restructuring | project guard | **PASS** — prose identical to the frozen abstract, verified by test | — | — |

### References

| Requirement | Source | Status | Artifact | Remaining action |
| --- | --- | --- | --- | --- |
| Journal CSL style (numeric, bracketed) | `journal-of-cheminformatics.csl` | **PASS** | CSL vendored; `article.md` declares it | apply via pandoc |
| BibTeX bibliography | template | **PASS** — 25 entries | `bibliography.bib` | — |
| Citations as keys, renumbered by the style | template | **PASS** — 48 `@key` citations, 0 hard-coded numbers | `article.md` | — |
| CiTO annotation supported | template + filters | **NOT APPLICABLE** — permitted, not required; plain citations used | filters vendored if wanted | optional |
| No fabricated bibliographic metadata | project guard | **PASS** — all 25 verified against publisher/Crossref/project sources | `REFERENCE_EVIDENCE_MAP.md` | — |

### Figures

| Requirement | Source | Status | Artifact | Remaining action |
| --- | --- | --- | --- | --- |
| Figures referenced and captioned | template | **PASS** — 4 main + 1 supplementary, all captioned | `article.md`, `figures/` | — |
| Vector format acceptable | template uses standard image embedding; SVG accepted for graphical abstract | **PASS** — all figures are SVG | `figures/*.svg` | confirm the submission system accepts SVG, else export to EPS/TIFF |
| Underlying data available | journal reproducibility policy | **PASS** | `figures/figure_0*_data.csv` | — |
| Figures embedded in the manuscript file | submission system | **ACTION BEFORE SUBMISSION** | — | embed via pandoc or upload separately |
| Exact resolution/dimension limits | Springer page (not retrievable non-interactively) | **ACTION BEFORE SUBMISSION** | — | confirm on the live guidelines page |

### Supplementary material

| Requirement | Source | Status | Artifact | Remaining action |
| --- | --- | --- | --- | --- |
| Supplementary information provided | journal policy | **PASS** | `supplementary/` (MD, DOCX, PDF) | — |
| Supplementary tables captioned | project standard | **PASS** — S1–S11 | SI document | export S1–S9 data files |
| Supplementary figure captioned | project standard | **PASS** — S1 | SI document; `figures/` | — |

### Reproducibility and availability

| Requirement | Source | Status | Artifact | Remaining action |
| --- | --- | --- | --- | --- |
| "will only publish research or software that is entirely reproducible by third parties" | template, Methods section | **PASS, with a caveat** | Declarations; SI §S4 | see repository note below |
| Source code must be provided | template | **PASS** — public repository named | Declarations | merge or tag recommended |
| Datasets accessible without registration or restrictive licence | template | **PASS** — TDC and ChEMBL are public | Declarations | — |
| "Availability of data and materials" section required | Springer guidelines | **PASS** | Declarations | — |
| Software/version reporting | project standard | **PASS** — Python, RDKit, NumPy, scikit-learn, SciPy all pinned | SI §S4 | — |
| Result identity reporting | project standard | **PASS** — all three frozen identities quoted | Declarations; SI §S4 | — |
| Historical provenance gap disclosed | project standard | **PASS** — 338/616 stated with per-track breakdown | SI §S4.2 | — |

**Repository caveat.** The repository is public, but its default branch
(`main`) does not yet contain the code that produced these results. The
availability statement names the `develop` branch and the exact commits
rather than claiming the default branch reproduces the article. This is
accurate but is weaker than the journal's reproducibility expectation
deserves; merging or tagging before submission is recommended.

### Declarations

| Requirement | Source | Status | Remaining action |
| --- | --- | --- | --- |
| Availability of data and materials | template | **PASS** | — |
| Competing interests | template | **AUTHOR INPUT REQUIRED** | authors to supply |
| Funding | template | **AUTHOR INPUT REQUIRED** | authors to supply |
| Authors' contributions | template | **AUTHOR INPUT REQUIRED** | authors to supply |
| Acknowledgements | template | **AUTHOR INPUT REQUIRED** | authors to supply |
| Authors' information (optional) | template | **NOT APPLICABLE** | — |
| Ethics approval and consent to participate | Springer policy | **PASS** — not applicable, stated with justification | — |
| Consent for publication | Springer policy | **PASS** — not applicable | — |

### Graphical abstract

| Requirement | Source | Status | Remaining action |
| --- | --- | --- | --- |
| Optional; 920 × 300 px, max 150 KB, jpeg/png/svg | template comment | **OPTIONAL — not created** | specification in `metadata/GRAPHICAL_ABSTRACT_SPEC.md`; produce if the authors want one |

### Author metadata

| Requirement | Status |
| --- | --- |
| Author list, order, affiliations | **AUTHOR INPUT REQUIRED** |
| Corresponding author and email | **AUTHOR INPUT REQUIRED** |
| ORCID iDs | **AUTHOR INPUT REQUIRED** |

### File formats for initial submission

| Requirement | Source | Status | Remaining action |
| --- | --- | --- | --- |
| DOCX for journal submission | template README: "the Word version for journal submission" | **PASS** | — |
| PDF for preprint/review | template README | **PASS**, plain-text typeset | regenerate via pandoc for typeset quality |
| Pandoc ≥ 2.12 build | template Makefile | **ACTION BEFORE SUBMISSION** | pandoc is not installed in this environment |

---

## Summary

| Status | Count |
| --- | --- |
| PASS | 27 |
| AUTHOR INPUT REQUIRED | 8 |
| ACTION BEFORE SUBMISSION | 6 |
| NOT APPLICABLE | 3 |
| **FAIL** | **0** |

No requirement is failed. Every outstanding item is either a fact only the
authors can supply, or a mechanical step (install pandoc, merge or tag,
export supplementary data files, confirm figure limits on the live
guidelines page).
