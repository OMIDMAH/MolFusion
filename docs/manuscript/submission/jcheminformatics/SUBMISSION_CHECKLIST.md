# Submission checklist — *Journal of Cheminformatics*, Research article

Status key: **READY** · **AUTHOR INPUT REQUIRED** · **ACTION BEFORE
SUBMISSION** · **OPTIONAL**

Nothing marked AUTHOR INPUT REQUIRED was guessed. The generated artifacts
carry visible `[AUTHOR INPUT REQUIRED]` placeholders wherever such a value
belongs, so no file silently invents one.

---

## Manuscript files

| Item | File | Status |
| --- | --- | --- |
| Journal Markdown source | `article.md` | READY |
| Manuscript DOCX | `MolFusion_JCheminformatics_Manuscript.docx` | READY |
| Review PDF | `MolFusion_JCheminformatics_Manuscript.pdf` | READY (see note below) |
| Supplementary Markdown | `supplementary/supplementary_information.md` | READY |
| Supplementary DOCX | `supplementary/MolFusion_Supplementary_Information.docx` | READY |
| Supplementary PDF | `supplementary/MolFusion_Supplementary_Information.pdf` | READY |
| Cover letter | `COVER_LETTER.md` | AUTHOR INPUT REQUIRED (signature block, reviewers) |
| BibTeX bibliography | `bibliography.bib` | READY — 25 entries |
| Journal CSL | `journal-of-cheminformatics.csl` + parent style | READY (vendored from the official template) |
| Pandoc build | `Makefile` | ACTION BEFORE SUBMISSION — pandoc not installed here |

**Note on the PDF.** The review PDF is produced by a dependency-free writer
and is complete and auditable in content, but it is plain-text typeset:
tables render as pipe-delimited text and figures are not embedded. For a
submission-quality PDF, run `make pandoc-pdf` once pandoc ≥ 2.12 is
available. The scientific content of both routes comes from the same
`article.md`.

## Figures

| Item | File | Status |
| --- | --- | --- |
| Figure 1 — rank heatmap | `figures/figure_01_rank_heatmap.svg` | READY (vector) |
| Figure 2 — mean rank + bootstrap CI | `figures/figure_02_mean_rank_ci.svg` | READY (vector) |
| Figure 3 — A1 → A2 robustness | `figures/figure_03_rank_robustness.svg` | READY (vector) |
| Figure 4 — rank vs nonlinear cost | `figures/figure_04_rank_vs_cost.svg` | READY (vector) |
| Supplementary Figure S1 — endpoint stability | `figures/figure_05_endpoint_stability.svg` | READY (vector) |
| Underlying data for every figure | `figures/figure_0*_data.csv` | READY |
| Figures embedded in DOCX/PDF | — | ACTION BEFORE SUBMISSION — supplied as separate files; embed via the pandoc route or at upload |
| Graphical abstract | `metadata/GRAPHICAL_ABSTRACT_SPEC.md` | OPTIONAL — specified, not created |

## Tables

| Item | Location | Status |
| --- | --- | --- |
| Table 1 — representation characteristics | main text caption; `tables/table1_*.csv` | READY |
| Table 2 — Track A1 performance | main text caption; `tables/table2_*.csv` | READY |
| Table 3 — Track A2 robustness | main text caption; `tables/table3_*.csv` | READY |
| Table 4 — key statistical contrasts | main text caption; `tables/table4_*.csv` | READY |
| Table 5 — computational cost | main text caption; `tables/table5_*.csv` | READY |
| Table 6 — endpoint stability | **moved to Supplementary (S10)**; `tables/table6_*.csv` | READY |
| Table 7 — 22 vs 19 endpoint subsets | **moved to Supplementary (S11)**; `tables/table7_*.csv` | READY |
| Supplementary Tables S1–S9 | captions in SI; data in the publication evidence package | ACTION BEFORE SUBMISSION — export the nine data files alongside the SI |

## Declarations

| Item | Status |
| --- | --- |
| Availability of data and materials | READY — public sources named; frozen release verifiable by checksum; scientific identities quoted |
| Availability of code | READY, with a caveat that must stay: the repository is public but its **default branch does not contain this code**; the statement names `develop` and the exact commits |
| Competing interests | **AUTHOR INPUT REQUIRED** |
| Funding | **AUTHOR INPUT REQUIRED** |
| Authors' contributions | **AUTHOR INPUT REQUIRED** |
| Acknowledgements | **AUTHOR INPUT REQUIRED** |
| Ethics approval and consent to participate | READY — not applicable, stated and justified |
| Consent for publication | READY — not applicable |

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
| Title | READY |
| Structured abstract (Background/Methods/Results), 220 words | READY — within the 350-word limit |
| Keywords (8) | READY |

Full detail: `metadata/AUTHOR_METADATA.md`.

## Recommended before submission

| Item | Status | Why it matters |
| --- | --- | --- |
| Install pandoc ≥ 2.12 and run `make` | ACTION BEFORE SUBMISSION | produces the journal-typeset DOCX/PDF with CSL-formatted references and CiTO support |
| Merge `develop` → `main`, or tag the manuscript commit | ACTION BEFORE SUBMISSION | readers currently must be told to use a non-default branch |
| Archive a release (e.g. Zenodo) and cite its DOI | RECOMMENDED | gives a permanent identifier; the journal requires third-party reproducibility |
| Replace the RDKit concept DOI with a release DOI | NOT POSSIBLE YET | no Zenodo record exists for RDKit 2026.03.5 (checked 2026-09-05); the concept DOI is retained per the project's own recommendation |
| Export Supplementary Tables S1–S9 as data files | ACTION BEFORE SUBMISSION | captions exist; the data lives in the publication evidence package |
| Embed figures into the manuscript file | ACTION BEFORE SUBMISSION | or upload separately if the submission system prefers it |

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

Enforced by `backend/tests/test_submission_package.py` (78 tests) against
the extracted text of the generated DOCX and PDF, not merely against the
Markdown source.
