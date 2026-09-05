# Supplementary Information

**MolFusion: Probe-Dependent Performance of Molecular Representations
Across 22 ADMET Endpoints**

This document contains the Supplementary Methods, the supplementary tables
and figure, and the reproducibility record. Every value here is taken from
the frozen publication evidence package
(identity `5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`);
nothing was recomputed for this document.

---

# S1. Supplementary Methods

## S1.1 Representation configurations

Every representation is produced by a versioned component and was frozen at
version 1.0.0 for the duration of the benchmark. Configurations are given
in full so that each vector can be regenerated exactly.

| Representation | Configuration |
| --- | --- |
| `morgan_ecfp4_1024` | RDKit `rdFingerprintGenerator` Morgan generator; radius 2; 1024 bits (ECFP4) |
| `maccs_keys_167` | RDKit MACCS implementation; 167 bits |
| `rdkit_physchem_descriptors` | 217 RDKit physicochemical descriptors; descriptor name list and order frozen at construction |
| `avalon_1024` | RDKit `pyAvalonTools.GetAvalonFP`; 1024 bits; `bitFlags = 15761407` |
| `erg_reduced_graph_315` | RDKit ErG; `atomTypes = 0`, `fuzzIncrement = 0.3`, `minPath = 1`, `maxPath = 15`; output width 315 derived from these parameters |
| `rdkit_fragment_descriptors` | 85 RDKit fragment-count descriptors; names and order frozen |
| `smiles_tfidf_4096` | frozen artifact `chembl37_token_ngrams_1_3` v1.0.0; token n-grams of order 1–3; absolute `min_df = 5`; sublinear TF × smoothed IDF; L2-normalised; 4096 dimensions |

The Avalon bit-flag value is the `GetAvalonFP` default in RDKit 2026.03.5
and is recorded explicitly because it determines which structural features
are enumerated and could change between library releases.

**TF-IDF vocabulary selection.** Candidate n-grams with an absolute corpus
document frequency of at least 5 were ranked by descending document
frequency, ties broken by ascending lexicographic n-gram token tuple, and
the top 4096 retained. Vector indices follow lexicographic order of the
selected token tuples. An n-gram outside the vocabulary contributes nothing
and triggers no vocabulary growth, refit or error. A molecule that
tokenises successfully but retains no vocabulary n-gram yields an exact
zero vector, which is a valid representation rather than a failure.

**Canonicalisation and tokenisation.** Molecular identity is canonical
isomeric SMILES under the frozen contract
`rdkit_canonical_isomeric_smiles_v1`. TF-IDF tokenisation uses the frozen
lexer `rdkit_smiles_lexer_v1` and is lossless with respect to the canonical
SMILES: concatenating the emitted tokens reproduces the input exactly.

## S1.2 Dataset release identity

The frozen release is `TDC-ADMET-2026-09`. Each endpoint file is serialised
under a deterministic contract — fixed column order, UTF-8 without
byte-order mark, LF line endings, source row order preserved,
round-trip-exact floating-point representation — so the same logical
dataset always produces the same bytes. Per-endpoint SHA-256 checksums and
the release identity are distributed in the provenance manifest
(`backend/benchmark_manifests/tdc_admet_group.json`).

Across the 22 endpoints, 81,809 official records yield 79,712 usable
molecules under the MolFusion curation applied in Track A2. Two records
could not be parsed by RDKit, both in `solubility_aqsoldb`, of 9,982
records for that endpoint; all other endpoints parsed completely.

## S1.3 Full hyperparameter grid

Four candidates per probe and task type, selected on validation performance
under the endpoint's primary metric. All other estimator settings remained
at library defaults with the seed fixed per realisation.

| Probe | Task | Candidates |
| --- | --- | --- |
| Linear | classification | `C` ∈ {0.01, 0.1, 1.0, 10.0} (logistic regression, `max_iter = 5000`) |
| Linear | regression | `alpha` ∈ {0.1, 1.0, 10.0, 100.0} (ridge regression) |
| Nonlinear | classification | `learning_rate` ∈ {0.05, 0.1} × `max_leaf_nodes` ∈ {15, 31} |
| Nonlinear | regression | `learning_rate` ∈ {0.05, 0.1} × `max_leaf_nodes` ∈ {15, 31} |

**Preprocessing is representation- and probe-specific, not uniform.** Under
the linear probe: a stateless fold of ±∞ to NaN, then median imputation
fitted on the training split alone, then standardisation for
`rdkit_physchem_descriptors`, `erg_reduced_graph_315` and
`rdkit_fragment_descriptors` only. The binary fingerprints already share a
0/1 scale and `smiles_tfidf_4096` is already L2-normalised by its own
contract, so standardising them would add a fitted step that changes
nothing while creating a further opportunity for information to cross the
split boundary. Under the nonlinear probe no scaler or imputer is applied:
the model is scale-invariant and consumes missing values natively.

**No class-imbalance handling was applied.** No class weighting,
resampling, SMOTE or threshold adjustment was used; the `class_weight`
parameter was left unset for both classifiers. Imbalance is addressed at
evaluation, where AUPRC is reported for every classification endpoint.

## S1.4 Statistical procedures

The endpoint is the unit of statistical inference throughout. The five seed
realisations within an endpoint are not treated as independent
observations: in Track A1 they share a single test partition, and in Track
A2 they are five views of the same molecules. Treating them as independent
would inflate the effective sample size from 22 to 110.

Friedman omnibus tests were computed on within-endpoint ranks with
endpoints as blocks, per probe and task family. Pairwise tests were
conducted only within families whose omnibus test rejected at α = 0.05,
using paired Wilcoxon signed-rank tests with Holm correction applied within
each family, and matched-pairs rank-biserial correlation reported as effect
size for every contrast. Bootstrap intervals used 10,000 resamples with a
fixed seed of 0 and the endpoint as the resampling unit; they are marginal
per-representation intervals rather than a simultaneous band and were not
used as a test of difference.

---

# S2. Supplementary tables

The following tables are provided as separate data files alongside this
document. Captions are given at the end of this document.

| Table | Content | Rows |
| --- | --- | --- |
| S1 | Complete pairwise contrasts, Track A2 | 126 |
| S2 | Complete pairwise contrasts, Track A1 | 105 |
| S3 | Reproduction of Track A1 contrasts under Track A2 | 11 |
| S4 | Endpoint-level ranks, both tracks | 616 |
| S5 | ChEMBL 37 corpus exposure by endpoint | 44 |
| S6 | Per-partition stability detail | 308 |
| S7 | Curation effects and partition distinctness | 22 |
| S8 | Friedman omnibus results, both tracks | 12 |
| S9 | Execution provenance audit | 2 tracks |

Two tables moved here from the main text under the journal layout, and are
also provided as data files:

| Table | Content |
| --- | --- |
| S10 (main Table 6) | Endpoint stability, Kendall's W per endpoint and probe |
| S11 (main Table 7) | All-endpoint and repartitioned-subset summaries side by side |

---

# S3. Supplementary figure

**Supplementary Figure S1** — stability of representation ordering across
partitions, by endpoint. The caption is given at the end of this document
and the figure is supplied as a vector file with its underlying data table.

---

# S4. Reproducibility record

A reader should not have to infer any of the following from prose.

| Item | Value |
| --- | --- |
| Benchmark release | `TDC-ADMET-2026-09` |
| Endpoints | 22 (13 classification, 9 regression) |
| Representations | 7 fixed-vector, all agent version 1.0.0 |
| TF-IDF artifact | `chembl37_token_ngrams_1_3` v1.0.0 |
| Canonicalisation contract | `rdkit_canonical_isomeric_smiles_v1` |
| Tokenisation contract | `rdkit_smiles_lexer_v1` |
| Track A1 split semantics | shipped TDC test partition held fixed; train/validation re-split at fractions (0.875, 0.125, 0.0) with seeds 1–5 |
| Track A2 split semantics | independent Bemis–Murcko scaffold repartitioning, 70/10/20, `includeChirality = True`, seeds 0–4, full curation |
| Probes | regularised logistic regression / ridge; histogram-based gradient boosting |
| Hyperparameter budget | 4 candidates per probe × task, identical for every representation |
| Primary metrics | AUROC (classification), MAE (regression) |
| Secondary metrics | AUPRC, balanced accuracy, MCC; RMSE, R², Spearman |
| Statistical plan | Friedman → Holm-corrected paired Wilcoxon → rank-biserial → endpoint bootstrap (10,000 resamples, seed 0) |
| Python | 3.11.15 |
| RDKit | 2026.03.5 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.0 |
| SciPy | 1.17.1 (analysis environment) |
| Track A1 execution commits | `459653b`, `ddabb42`, `2bcb467` |
| Track A2 execution commit | `e6ae297` |
| Track A1 analysis commit | `fe4bc60` |
| Track A2 analysis commit | `15b78a2` |
| Provenance hardening commit (subsequent) | `89335dc` |
| Track A1 scientific identity | `d40ef09b398f47914aa51f99fd6a4f5893f7778b50c0cca04404b575632de868` |
| Track A2 scientific identity | `9dd5dfa6067c8a760b0bb8fb39648f71f662f2fa1bbf4cc5d7cb0cd495a69f14` |
| Publication evidence identity | `5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18` |

## S4.1 Result identity

Each completed result set carries two distinct digests: a file digest over
the serialised result table, and a **scientific identity** computed over
only those columns that constitute a scientific result — release, track,
endpoint, split, seed, representation, model family, probe,
hyperparameters, metric, metric value, partition sizes and feature
accounting. Timing columns and cache-hit flags are deliberately excluded,
so re-collecting results on a different machine reproduces the same
scientific identity even though wall-clock values differ.

## S4.2 Execution provenance

The reported results were produced by the benchmark runner as it existed at
execution time. A post-run provenance audit identified a logging defect in
that runner: each worker process resolved the Git commit independently and
recorded a null value when the subprocess call failed under load.
Consequently **338 of 616 result shards lack an embedded Git commit
identifier**:

| Track | Shards | With a recorded commit | Without |
| --- | --- | --- | --- |
| A1 | 308 | 181 (`459653b` 167, `ddabb42` 12, `2bcb467` 2) | **127** |
| A2 | 308 | 97 (`e6ae297`) | **211** |
| **Total** | **616** | **278** | **338** |

The consequences of this defect are bounded and are stated explicitly:

- **The scientific result values were unaffected.** The same code produced
  every shard; `protocol_version` and `benchmark_release` agree across all
  616.
- **The scientific identities were unchanged**, and are the values recorded
  in the table above.
- **Historical attribution was verified after the fact**, using run-level
  metadata, the content-derived scientific identities, immutable dataset
  and result checksums, and the source-code history of the runner. The
  runner sources are byte-identical to the execution commits.
- **No result shard was backfilled or modified.** Writing a reconstructed
  commit into a shard would have destroyed the only evidence that the
  defect existed and would have made a null shard indistinguishable from
  one that recorded its commit honestly.

The provenance mechanism was subsequently hardened (commit `89335dc`) so
that execution provenance is captured once in the orchestrating process,
requires a non-null commit before any scientific work begins, distinguishes
tracked source modifications from unrelated untracked files, and propagates
a single immutable record to all workers. **That hardened mechanism applies
to subsequent executions only. It did not produce the results reported in
this article**, and it was not applied retroactively to the historical
runs.

## S4.3 Curation applied per track

Track A1 applies **no cleaning**: official records are consumed exactly as
shipped, so that results remain comparable with published work on this
benchmark. Duplicate and conflicting-label structure was audited and is
reported rather than silently corrected. Track A2 applies the full
MolFusion curation policy: duplicate canonical molecules collapsed,
molecules carrying conflicting labels removed, unparseable structures
excluded. The two treatments are never mixed, and the tracks are never
pooled.
