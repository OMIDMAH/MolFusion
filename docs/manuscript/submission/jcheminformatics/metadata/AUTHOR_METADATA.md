# Author and submission metadata — unresolved

Nothing on this page was inferred. Authorship, order, affiliations,
corresponding author, funding, competing interests and contributions are
facts about people, and guessing any of them would be a fabrication in the
submission record. Each is marked and must be supplied by the authors.

The manuscript, DOCX, PDF and cover letter all carry visible
`[AUTHOR INPUT REQUIRED]` placeholders where these values belong, so no
generated artifact silently invents them.

---

## Required before submission

| Field | Status | Where it appears |
| --- | --- | --- |
| Author list | **[AUTHOR INPUT REQUIRED]** | `article.md` YAML header; DOCX; PDF; cover letter |
| Author order | **[AUTHOR INPUT REQUIRED]** | as above |
| Affiliations | **[AUTHOR INPUT REQUIRED]** | `article.md` YAML header |
| Corresponding author | **[AUTHOR INPUT REQUIRED]** | `article.md` correspondence line; cover letter |
| Corresponding author email | **[AUTHOR INPUT REQUIRED]** | as above |
| ORCID iDs | **[AUTHOR INPUT REQUIRED]** | `article.md` YAML header (`orcid:` field per author) |
| Authors' contributions | **[AUTHOR INPUT REQUIRED]** | Declarations |
| Funding | **[AUTHOR INPUT REQUIRED]** | Declarations |
| Competing interests | **[AUTHOR INPUT REQUIRED]** | Declarations |
| Acknowledgements | **[AUTHOR INPUT REQUIRED]** | Declarations |
| Suggested reviewers | **[AUTHOR INPUT REQUIRED]**, if invited | submission system |
| Excluded reviewers | **[AUTHOR INPUT REQUIRED]**, if applicable | submission system |

## Resolved without author input

| Field | Value | Basis |
| --- | --- | --- |
| Article type | Research article | frozen for this phase |
| Journal | *Journal of Cheminformatics* | frozen for this phase |
| Title | *MolFusion: Probe-Dependent Performance of Molecular Representations Across 22 ADMET Endpoints* | Phase 6C.1 freeze |
| Abstract | 220 words, structured Background/Methods/Results | Phase 6C.1 freeze, restructured to the journal template |
| Keywords | 8, semicolon-separated inside the abstract block | Phase 6C.1 freeze |
| Ethics approval | Not applicable — public molecular property data only, no human participants, human data or animals | verifiable from the study design |
| Consent for publication | Not applicable | as above |
| Code availability | `https://github.com/OMIDMAH/MolFusion`, `develop` branch, commits listed in Declarations | repository verified public, default branch `main`, `develop` at `d1bba81` |
| Data availability | TDC ADMET group and ChEMBL 37, both public; frozen release verifiable by checksum | verified public sources |

## A note on the repository state

The repository is **public**, but its **default branch (`main`) does not
contain the code that produced these results**. A pull request from
`develop` to `main` is open and unmerged, and merging it was outside the
scope of this work.

The Declarations therefore direct readers to the `develop` branch and to
the specific commits, rather than claiming that the default branch
reproduces the article. If the authors merge before submission, the
availability statement should be updated to name the merge commit or a tag.

**Recommended but not created:** a Git tag on the manuscript commit and an
archived release DOI (for example via Zenodo) would give readers a
permanent identifier. Neither was created, because creating a tag, release
or external archive was not authorised for this phase.
