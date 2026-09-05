# Reference evidence map (Phase 6C.5)

Every reference was checked against a publisher page, the Crossref API, or
the project's own recommended citation before entering the bibliography.
No bibliographic field was filled from memory, and no reference was added
to make the bibliography look current.

Structured source of truth: [`references.json`](references.json). The
rendered bibliography ([`REFERENCES.md`](REFERENCES.md)) and the copy
embedded in [`MANUSCRIPT_DRAFT.md`](MANUSCRIPT_DRAFT.md) are both generated
from it by `molfusion_backend.benchmark.manuscript_cli`, so they cannot
drift apart.

Evidence identity: `5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`.

---

## Verification table

| # | Reference | Supported statement | Section(s) | Verification source | DOI / identifier | Class | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Bemis & Murcko 1996 | Bemis–Murcko scaffold definition used for Track A2 repartitioning | Intro, Methods §2.6, Table 3 caption | ACS publisher page | doi:10.1021/jm9602928 | primary | verified |
| 2 | Cereto-Massagué et al. 2015 | fingerprints encode local substructure presence; fingerprint choice affects performance | Intro, Discussion §4.2 | ScienceDirect; PubMed | doi:10.1016/j.ymeth.2014.08.005 · PMID 25132639 | review | verified |
| 3 | David et al. 2020 | representation choice is a primary design decision; landscape of molecular representations incl. learned classes | Intro, Discussion §4.2, §4.5 | PMC7495975; PubMed | doi:10.1186/s13321-020-00460-5 · PMID 33431035 | review | verified |
| 4 | Demšar 2006 | Friedman omnibus followed by corrected post-hoc tests when comparing methods across datasets | Intro, Methods §2.10 | JMLR official BibTeX | — (JMLR 7:1-30) | primary | verified |
| 5 | Durant et al. 2002 | MDL/MACCS structural key set | Intro, Methods §2.2 | PubMed; ACS | doi:10.1021/ci010132r · PMID 12444722 | primary | verified |
| 6 | Efron 1979 | nonparametric bootstrap resampling | Methods §2.10 | Project Euclid | doi:10.1214/aos/1176344552 | primary | verified |
| 7 | Friedman 1937 | Friedman rank-based omnibus test | Intro, Methods §2.10, Table 4 & S8 captions | Taylor & Francis | doi:10.1080/01621459.1937.10503522 | primary | verified |
| 8 | Gedeck et al. 2006 | Avalon fingerprint | Intro, Methods §2.2 | ACS publisher page | doi:10.1021/ci050413p | primary | verified |
| 9 | Holm 1979 | Holm step-down multiple-comparison correction | Methods §2.10, Table 4 caption | JSTOR record | JSTOR 4615733 (no DOI) | primary | verified |
| 10 | Huang et al. 2021 | TDC dataset collection and benchmark task definitions | Intro, Methods §2.3 | arXiv 2102.09548; TDC project citation | arXiv:2102.09548 | primary | verified |
| 11 | Huang et al. 2022 | primary peer-reviewed TDC publication | Intro, Methods §2.3 | **Crossref API, full record** | doi:10.1038/s41589-022-01131-2 | primary | verified |
| 12 | Kendall & Babington Smith 1939 | Kendall coefficient of concordance (W) | Methods §2.10, Table 6 caption | **Crossref API, exact match** | doi:10.1214/aoms/1177732186 | primary | verified |
| 13 | Kerby 2014 | matched-pairs rank-biserial correlation as an effect size | Methods §2.10, Table 4 caption | SAGE publisher page | doi:10.2466/11.IT.3.1 | primary | verified |
| 14 | Krenn et al. 2020 | SELFIES robust molecular string representation | Intro, Methods §2.1 | **Crossref API**; IOPscience | doi:10.1088/2632-2153/aba947 | primary | verified |
| 15 | Landrum (RDKit) | all cheminformatics computation | Methods §2.2 | RDKit project documentation | doi:10.5281/zenodo.591637 | software | verified |
| 16 | Morgan 1965 | Morgan atom-relaxation algorithm underlying circular fingerprints | Intro, Methods §2.2 | ACS via DOI; Wikidata Q28837925 | doi:10.1021/c160017a018 | primary | verified |
| 17 | Pedregosa et al. 2011 | linear and gradient-boosting probe implementations | Intro, Methods §2.7 | scikit-learn citation page; dblp | — (JMLR 12:2825-2830) | software | verified |
| 18 | Rogers & Hahn 2010 | extended-connectivity (ECFP) fingerprints | Intro, Methods §2.2 | ACS; PubMed | doi:10.1021/ci100050t · PMID 20426451 | primary | verified |
| 19 | Stiefl et al. 2006 | extended reduced graph (ErG) representation | Intro, Methods §2.2 | ACS; PubMed | doi:10.1021/ci050457y · PMID 16426057 | primary | verified |
| 20 | Todeschini & Consonni 2009 | physicochemical and constitutional molecular descriptors | Intro, Discussion §4.2, Table 1 | Wiley-VCH publisher page | ISBN 978-3-527-31852-0 | book | verified |
| 21 | Virtanen et al. 2020 | statistical computation | Methods §2.10 | Nature; PubMed | doi:10.1038/s41592-019-0686-2 · PMID 32015543 | software | verified |
| 22 | Weininger 1988 | SMILES line notation | Intro, Methods §2.2 | ACS publisher page | doi:10.1021/ci00057a005 | primary | verified |
| 23 | Wilcoxon 1945 | Wilcoxon signed-rank test | Methods §2.10, Table 4 caption | JSTOR DOI record | doi:10.2307/3001968 | primary | verified |
| 24 | Wu et al. 2018 | benchmark datasets, metrics and featurisation/splitting choices affect molecular ML conclusions | Intro, Discussion §4.5 | **Crossref API**; RSC | doi:10.1039/c7sc02664a · PMID 29629118 | primary | verified |
| 25 | Zdrazil et al. 2024 | ChEMBL bioactivity database, source of the TF-IDF fitting corpus | Methods §2.2, Table 1 & S5 captions | Oxford Academic | doi:10.1093/nar/gkad1004 | primary | verified |

**Totals:** 25 references — **18 primary**, **3 review/book context**
(2, 3, 20), **4 software/resource** (15, 17, 21, and 10 as a
datasets/benchmark paper). All 25 are cited in the assembled manuscript;
none is uncited.

---

## Notes on specific references

**Morgan versus ECFP (16, 18).** These are related but not
interchangeable. Morgan 1965 describes the atom-relaxation algorithm for
generating a canonical machine description; Rogers & Hahn 2010 describes the
extended-connectivity fingerprint formulation and the ECFP naming. The
manuscript cites both together where circular fingerprints are introduced,
and does not attribute ECFP terminology to Morgan.

**RDKit (15).** The RDKit project states that no RDKit publication exists
and recommends citing the software together with the Zenodo DOI for the
version used. The bibliography records the concept DOI
(10.5281/zenodo.591637); **the release-specific DOI for version 2026.03.5
should replace it at submission.** No journal article was invented for it.

**Holm 1979 (9).** No DOI has been assigned to this article. The JSTOR
stable identifier is recorded instead, and the DOI field is deliberately
left empty rather than filled with a plausible-looking value.

**Avalon (8).** The Avalon fingerprint is described in the supporting
information of Gedeck et al. 2006, which is the citation the RDKit
documentation points to for this fingerprint. There is no separate
standalone Avalon paper.

**TDC (10, 11).** Both the peer-reviewed *Nature Chemical Biology* article
and the NeurIPS Datasets and Benchmarks paper are cited, in line with the
brief's requirement not to rely on a repository page alone. Reference 11 is
the primary peer-reviewed publication.

**Two statistical citations added in this phase.** Kerby 2014 (13) for the
matched-pairs rank-biserial effect size and Efron 1979 (6) for the
bootstrap were not present as placeholders in the earlier drafts; they were
added to Methods §2.10 where those procedures are described, because the
phase brief lists both among the references to verify.

---

## Literature-claim check

Each literature-supported statement was checked against what the cited
source actually establishes, not against its title.

| Statement | Reference | Does the source support it? |
| --- | --- | --- |
| Representation choice is an early, consequential design decision | 3 | Yes — a review of molecular representations and their role in AI-driven drug discovery |
| Circular fingerprints enumerate substructural environments to a fixed radius | 16, 18 | Yes — 18 defines ECFP generation; 16 the underlying Morgan algorithm |
| Substructure key fingerprints record presence of predefined patterns | 5 | Yes — MDL/MACCS keyset construction and reoptimisation |
| Reduced graphs abstract to pharmacophoric nodes for scaffold-level similarity | 19 | Yes — ErG is introduced explicitly for scaffold hopping |
| Physicochemical descriptors compute interpretable continuous properties | 20 | Yes — the standard reference work defining molecular descriptors |
| SELFIES adds syntactic robustness to string encodings | 14 | Yes — the paper's central claim is 100 % syntactic robustness |
| Fingerprint choice affects downstream performance | 2 | Yes — the review states different fingerprints represent different aspects and this affects search performance |
| Benchmark collections fix datasets, metrics and splits to reduce variance | 10, 11, 24 | Yes — both TDC papers and MoleculeNet establish standardised datasets, metrics and splits |
| Featurisation and splitting choices materially affect ML conclusions | 24 | Yes — MoleculeNet evaluates featurisations and splitting and reports their effect |
| Friedman then corrected post-hoc tests is the appropriate design for multi-dataset comparison | 4, 7 | Yes — 7 introduces the test; 4 recommends exactly this procedure for comparing methods across datasets |
| Learned representations adapt the encoding to the task | 3 | Yes — covered in the review's treatment of learned representations |

**No source is cited for a claim it does not make**, and **no external
numerical performance comparison is drawn from any cited work.** The
manuscript nowhere compares its measured performance against results
reported in the literature; the literature is used only to establish
context and to attribute methods.
