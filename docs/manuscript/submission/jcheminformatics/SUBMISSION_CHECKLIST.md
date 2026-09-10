# Submission checklist — *Journal of Cheminformatics*, Research article

Status key: **PASS** · **AUTHOR INPUT REQUIRED** · **HUMAN PRE-SUBMISSION
CHECK REQUIRED** · **OPTIONAL** · **NOT APPLICABLE**

Updated Phase 6C.7. Every item is classified; nothing is left implicit.

Nothing marked AUTHOR INPUT REQUIRED was guessed. The generated artifacts
carry visible `[AUTHOR INPUT REQUIRED]` placeholders wherever such a value
belongs, so no file silently invents one.

---

## Manuscript files

| Item | File | Status |
| --- | --- | --- |
| Journal Markdown source | `article.md` | PASS |
| **Submission DOCX** | `MolFusion_JCheminformatics_Manuscript_FINAL.docx` | **PASS** — pandoc 3.11 + journal CSL |
| **Submission PDF** | `MolFusion_JCheminformatics_Manuscript_FINAL.pdf` | **PASS** — 35 pages, embedded fonts |
| Review DOCX (6C.6) | `MolFusion_JCheminformatics_Manuscript.docx` | PASS — retained as a review artifact |
| Review PDF (6C.6) | `MolFusion_JCheminformatics_Manuscript.pdf` | PASS — retained as a review artifact |
| Supplementary Markdown | `supplementary/supplementary_information.md` | PASS |
| **Supplementary DOCX** | `supplementary/MolFusion_Supplementary_Information_FINAL.docx` | **PASS** |
| **Supplementary PDF** | `supplementary/MolFusion_Supplementary_Information_FINAL.pdf` | **PASS** |
| CSL-rendered plain text (audit trail) | `validation/article_rendered.txt` | PASS |
| Cover letter | `COVER_LETTER.md` | **AUTHOR INPUT REQUIRED** — scientific content final; signature block and reviewers outstanding |
| BibTeX bibliography | `bibliography.bib` | **PASS** — 25 entries, all rendered by the CSL |
| Journal CSL | `journal-of-cheminformatics.csl` + parent style | **PASS** — applied by pandoc |
| Pandoc build | `Makefile` | **PASS** — pandoc 3.11, typst 0.15.1, commands verified |

**Two build routes, both retained.** The `_FINAL` artifacts are produced by
pandoc 3.11 through the journal's own template, CSL and CiTO filters, with
typst 0.15.1 as the PDF engine; these are the submission files. The Phase
6C.6 artifacts, produced by dependency-free writers, are kept as review
copies and as evidence that the package builds without a toolchain. Both
routes read the same `article.md`, and the guard suite runs against both.

The `_FINAL` DOCX preserves real Unicode notation; the review PDF
transliterates it to ASCII, which is why the two are not interchangeable
for submission.

## Figures

| Item | File | Status |
| --- | --- | --- |
| Figure 1 — rank heatmap | `figures/figure_01_rank_heatmap.svg` | PASS — vector |
| Figure 2 — mean rank + bootstrap CI | `figures/figure_02_mean_rank_ci.svg` | PASS — vector |
| Figure 3 — A1 → A2 robustness | `figures/figure_03_rank_robustness.svg` | PASS — vector |
| Figure 4 — rank vs nonlinear cost | `figures/figure_04_rank_vs_cost.svg` | PASS — vector |
| Supplementary Figure S1 — endpoint stability | `figures/figure_05_endpoint_stability.svg` | PASS — vector |
| Underlying data for every figure | `figures/figure_0*_data.csv` | PASS |
| Figures embedded in DOCX/PDF | — | **ACTION BEFORE SUBMISSION** — supplied as separate vector files; most submission systems prefer separate upload |
| Graphical abstract | `metadata/GRAPHICAL_ABSTRACT_SPEC.md` | OPTIONAL — specified, not created |

## Tables

| Item | Location | Status |
| --- | --- | --- |
| Table 1 — representation characteristics | main text caption; `tables/table1_*.csv` | PASS |
| Table 2 — Track A1 performance | main text caption; `tables/table2_*.csv` | PASS |
| Table 3 — Track A2 robustness | main text caption; `tables/table3_*.csv` | PASS |
| Table 4 — key statistical contrasts | main text caption; `tables/table4_*.csv` | PASS |
| Table 5 — computational cost | main text caption; `tables/table5_*.csv` | PASS |
| Table 6 — endpoint stability | **moved to Supplementary (S10)**; `tables/table6_*.csv` | PASS |
| Table 7 — 22 vs 19 endpoint subsets | **moved to Supplementary (S11)**; `tables/table7_*.csv` | PASS |
| Supplementary Tables S1–S11 | `supplementary/tables/*.csv` + `SUPPLEMENTARY_TABLE_MANIFEST.json` | **PASS** — all eleven exported, each with row/column counts and a SHA-256 |

## Declarations

| Item | Status |
| --- | --- |
| Availability of data and materials | PASS — public sources named; frozen release verifiable by checksum; scientific identities quoted |
| Availability of code | PASS, with a caveat that must stay: the repository is public but its **default branch does not contain this code**; the statement names `develop` and the exact commits |
| Competing interests | **AUTHOR INPUT REQUIRED** |
| Funding | **AUTHOR INPUT REQUIRED** |
| Authors' contributions | **AUTHOR INPUT REQUIRED** |
| Acknowledgements | **AUTHOR INPUT REQUIRED** |
| Ethics approval and consent to participate | PASS — not applicable, stated and justified |
| Consent for publication | PASS — not applicable |

## Author and submission metadata

| Item | Status |
| --- | --- |
| Author list | **AUTHOR INPUT REQUIRED** |
| Author order | **AUTHOR INPUT REQUIRED** |
| Affiliations | **AUTHOR INPUT REQUIRED** |
| Corresponding author + email | **AUTHOR INPUT REQUIRED** |
| ORCID iDs | **AUTHOR INPUT REQUIRED** |
| Suggested reviewers | **AUTHOR INPUT REQUIRED**, if invited |
| Excluded reviewers | **AUTHOR INPUT REQUIRED**, if applicable |
| Title | PASS |
| Structured abstract (Background/Methods/Results), 220 words | PASS — within the 350-word limit |
| Keywords (8) | PASS |

Full detail: `metadata/AUTHOR_METADATA.md`.

## Recommended before submission

| Item | Status | Why it matters |
| --- | --- | --- |
| Install pandoc and build | **PASS** — done in Phase 6C.7 | pandoc 3.11 + typst 0.15.1 |
| Tag the manuscript commit | **AUTHOR INPUT REQUIRED** — plan prepared, nothing created | `metadata/RELEASE_PLAN.md`; recommended tag `molfusion-jcheminform-2026-09` |
| Archive a release (e.g. Zenodo) | **OPTIONAL** — checklist prepared | `metadata/RELEASE_PLAN.md` §7; Zenodo must be linked *before* the release is published |
| RDKit release DOI | **NOT APPLICABLE** — re-checked 2026-09-11 | no Zenodo record exists for Release_2026_03_5; concept DOI retained and the exact version recorded in the reference and in Methods |
| Export Supplementary Tables S1–S11 | **PASS** — done in Phase 6C.7 | 11 CSVs with a hash manifest |
| Embed or upload figures | **ACTION BEFORE SUBMISSION** | vector files ready; most systems prefer separate upload |

## Scientific guards — all verified against the generated artifacts

| Guard | Result |
| --- | --- |
| Confidence intervals `[1.45, 2.41]` / `[1.32, 2.27]`; `2.47` absent | PASS |
| Compute shares 35.4 % / 30.7 %; `29.8 %` never paired with 35.4 % | PASS |
| Linear probe: difference detected, leadership unresolved | PASS |
| No regression-specific nonlinear superiority claim; p = 0.079 disclosed | PASS |
| Track A2 never called external validation | PASS |
| C9 absent from abstract and conclusions; exploratory in Limitations | PASS |
| No information-content claim; "predictive accessibility" used | PASS |
| No endpoint-specific superlative for the six low-stability endpoints | PASS |
| `vdss_lombardo` remains BORDERLINE | PASS |
| Provenance hardening never credited with producing A1/A2 | PASS |
| No universal-superiority or state-of-the-art claim | PASS |
| No external numerical comparison against published models | PASS |
| Cover letter free of prohibited claims | PASS |

Enforced by `backend/tests/test_submission_package.py` (78 tests) and
`backend/tests/test_submission_final.py` (80 tests) against the extracted
text of **both** build routes, not merely against the Markdown source.

Two defects were caught this way in Phase 6C.7, both invisible in the
Markdown: an unescaped `%` in the SELFIES title started a BibTeX comment
and truncated that reference, and the journal style emits no DOI for any
non-article type, which dropped the RDKit identifier from the bibliography.
Both are fixed and both are now guarded.
