# 2. Materials and Methods — draft (Phase 6C.2)

Evidence source: Phase 6B publication package, identity
`5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`
(verified before drafting). Every technical and numerical statement below
was read from committed source or frozen output, not from recollection; the
mapping is in [`METHODS_EVIDENCE_MAP.md`](METHODS_EVIDENCE_MAP.md).

This document contains the Materials and Methods section only. No result is
stated anywhere in it.

---

## 2.1 MolFusion framework

MolFusion is a framework for reproducible, systematic comparison of
heterogeneous molecular representations. It is not a predictive model: it
supplies the infrastructure under which different molecular encodings can
be compared on equal terms, and the benchmark reported here is its first
large-scale application. The design goal is that every reported number be
attributable to a specific molecular input, a specific representation
implementation, and a specific analysis procedure, each identified by
content rather than by file location or execution time.

The evaluation workflow is uniform across representations:

```
molecule
  → canonical molecular identity
  → FeatureAgent (versioned representation implementation)
  → fixed-length numeric representation
  → frozen predictive probe
  → endpoint-specific evaluation
  → cross-endpoint statistical comparison
```

Representations are supplied by *FeatureAgents* registered in a central
registry. Each agent declares a stable identifier, a semantic version, a
value type (binary, count, or continuous), and, where the representation
has a fixed width, an output dimension. Registration is the only route by
which a representation enters a benchmark, so the set of representations
under comparison is enumerable and versioned rather than assembled ad hoc
at analysis time. An agent's declared identity participates in the cache
key of any computed feature matrix, so a change in agent version cannot
silently reuse features produced by an earlier implementation.

Two properties of the framework are relied upon throughout. First,
molecular identity is canonical and explicit, so the same compound is
recognised as the same compound across endpoints and representations.
Second, benchmark configuration — datasets, partitions, probe families,
hyperparameter budget, metrics, and the statistical plan — was frozen and
committed before execution, so no analysis decision reported here was made
after observing outcomes.

### Representation contract

Track A of the benchmark comprises **seven fixed-vector representations**.
A fixed-vector representation maps one molecule to a numeric vector of
predetermined length, which allows all seven to be consumed by an identical
downstream probe without any architectural difference between them.

MolFusion also implements a SELFIES sequence representation
[CITATION: SELFIES], which was deliberately **excluded from Track A**.
SELFIES produces a variable-length categorical token sequence rather than a
fixed-length numeric vector, so it cannot be supplied to the same probe as
the other seven. Evaluating it would require an additional learned sequence
encoder, and any performance difference would then confound the
representation with the capacity and training of that encoder. Excluding it
keeps the comparison between representations rather than between model
architectures. This is a scope decision, not a negative finding about
SELFIES: no SELFIES model was trained or evaluated, and no claim about its
utility is made.

No learned or pretrained molecular representation was benchmarked in this
study.

---

## 2.2 Molecular representations

The seven Track A representations are summarised below. All were computed
with RDKit [CITATION: RDKit] version 2026.03.5, and each was frozen at
agent version 1.0.0 for the duration of the benchmark.

| Representation | Family | Dimension | Value type |
| --- | --- | --- | --- |
| `morgan_ecfp4_1024` | circular fingerprint | 1024 | binary |
| `maccs_keys_167` | substructure key fingerprint | 167 | binary |
| `rdkit_physchem_descriptors` | physicochemical descriptors | 217 | continuous |
| `avalon_1024` | substructure fingerprint | 1024 | binary |
| `erg_reduced_graph_315` | reduced-graph features | 315 | continuous |
| `rdkit_fragment_descriptors` | fragment counts | 85 | count |
| `smiles_tfidf_4096` | SMILES token n-gram TF-IDF | 4096 | continuous |

These encodings are not interchangeable in kind, and are described
individually rather than under a single collective label.

**Morgan / ECFP.** Circular fingerprints [CITATION: Morgan fingerprint;
ECFP] computed with **radius 2** and folded to **1024 bits**, corresponding
to ECFP4. Generated through RDKit's `rdFingerprintGenerator` Morgan
generator rather than the deprecated convenience function, so that bit
generation follows the current supported implementation.

**MACCS keys.** The RDKit implementation of the **167-bit** MACCS
structural key set [CITATION: MACCS keys], a binary vector in which each
position indicates the presence of a predefined substructural pattern.

**Physicochemical descriptors.** The **217** RDKit physicochemical
descriptors available in the pinned RDKit release, computed as a continuous
vector. The descriptor name list and its ordering were frozen at agent
construction, so vector position *i* refers to the same descriptor in every
molecule, endpoint and track. These are computed molecular properties, not
learned embeddings. Descriptors that are undefined for a given molecule, or
that diverge numerically, are handled at the preprocessing stage (§2.8)
rather than by discarding the molecule.

**Avalon.** The Avalon fingerprint [CITATION: Avalon] at **1024 bits**,
computed with RDKit's `pyAvalonTools`. The bit-flag configuration was
pinned to **15761407**, the `GetAvalonFP` default in RDKit 2026.03.5,
recorded explicitly because the value determines which structural features
are enumerated and the library default could change between releases.

**ErG reduced graph.** Extended reduced graph descriptors
[CITATION: ErG] with the frozen parameters **`atomTypes = 0`,
`fuzzIncrement = 0.3`, `minPath = 1`, `maxPath = 15`**, yielding **315**
continuous dimensions. The output width is a deterministic function of
these parameters and was derived from them rather than assumed.

**Fragment descriptors.** The **85** RDKit fragment-count descriptors,
each counting occurrences of a defined functional-group pattern. Feature
names and ordering were frozen, as for the physicochemical set.

**SMILES token n-gram TF-IDF.** A frozen, externally fitted, unsupervised
text representation of the canonical SMILES string, of dimension **4096**.
Canonical SMILES are tokenised (§ below), token **n-grams of order 1–3**
are extracted, and each molecule is encoded as **sublinear term frequency
weighted by smoothed inverse document frequency, L2-normalised**.

The vocabulary and IDF weights are supplied by a versioned artifact,
`chembl37_token_ngrams_1_3` version `1.0.0`, fitted on a frozen ChEMBL 37
reference corpus [CITATION: ChEMBL]. Vocabulary selection was
deterministic: candidate n-grams with an absolute document frequency of at
least **`min_df = 5`** in the corpus were ranked by descending corpus
document frequency, ties broken by ascending lexicographic n-gram token
tuple, and the top 4096 retained; vector indices follow lexicographic order
of the selected token tuples. An n-gram absent from the frozen vocabulary
contributes nothing and does not trigger vocabulary growth, refitting, or
an error; a molecule that tokenises successfully but retains no vocabulary
n-gram yields an exact zero vector, which is a valid representation rather
than a failure.

Critically:

> The TF-IDF vocabulary and IDF weights were frozen before downstream ADMET
> evaluation and were not fitted using endpoint labels or endpoint data.

Because the fitting corpus is a general-purpose compound library, some
benchmark molecules also occur in it. This is **external unsupervised
corpus exposure**, quantified per endpoint and reported in the Results; no
benchmark label was visible at any point during fitting.

### Canonical SMILES and tokenisation

All molecular identity in MolFusion is defined by **canonical isomeric
SMILES** [CITATION: SMILES] under the frozen normalisation contract
`rdkit_canonical_isomeric_smiles_v1`. The same canonical form is used for
duplicate detection, scaffold assignment, feature-cache keying, and TF-IDF
tokenisation, so these operations cannot disagree about which molecules are
identical.

SMILES tokenisation for the TF-IDF representation uses the frozen lexer
`rdkit_smiles_lexer_v1`, which segments a canonical SMILES string into
chemically meaningful tokens (multi-character element symbols, bracketed
atoms, ring closures, bond and branch symbols). Tokenisation is **lossless
with respect to the canonical SMILES**: concatenating the emitted tokens
reproduces the input string exactly, so no structural information is
discarded before n-gram extraction, and a tokenisation failure is an error
rather than a silent truncation.

---

## 2.3 TDC ADMET benchmark

Evaluation used the ADMET benchmark group of the Therapeutics Data Commons
[CITATION: TDC; TDC ADMET], comprising **22 endpoints**: **13
classification** and **9 regression** tasks spanning absorption,
distribution, metabolism, excretion and toxicity.

**All 22 endpoints were retained.** Inclusion was determined by the frozen
protocol before any model was fitted: an endpoint was included if it
belonged to the ADMET benchmark group, carried a supported task type, and
provided the official partition files required for Track A1. No endpoint
was added, removed, or reweighted on the basis of observed model
performance.

Benchmark data were obtained once from the Therapeutics Data Commons using
PyTDC, and were immediately re-serialised into a frozen MolFusion dataset
release. **PyTDC is not a runtime dependency of the benchmark**: it was
used only for the initial acquisition and for reconciling MolFusion's
partition semantics against the official implementation, and it is not
imported by any code that computes a reported number. This isolates the
benchmark from upstream changes to the acquisition library.

The frozen release is named **`TDC-ADMET-2026-09`**. Every endpoint file is
serialised under a deterministic contract — fixed column order, UTF-8
without byte-order mark, LF line endings, source row order preserved,
round-trip-exact floating-point representation — so that the same logical
dataset always produces the same bytes. Each file carries a SHA-256
checksum, and the release as a whole carries a release identity derived
from those checksums. Any downstream result therefore names the exact
dataset bytes it consumed. Full per-endpoint checksums and the release
identity are given in Supplementary Methods rather than here.

---

## 2.4 Molecular standardization, identity, and data curation

A molecule's identity is its canonical isomeric SMILES (§2.2). Beyond
canonicalisation, **no structural standardisation was applied**:

- no salt stripping,
- no charge neutralisation,
- no tautomer canonicalisation,
- no stereochemistry removal,
- no largest-fragment selection.

This is deliberate. Each of these transformations changes which compound is
being modelled, and applying them silently would make results
non-comparable with the published benchmark while appearing to be a
formatting step. Where a source benchmark had already applied such a
transformation, the records were consumed as shipped.

**Duplicate and conflict auditing.** For every endpoint, three conditions
were quantified: exact duplicate input strings; distinct input strings that
canonicalise to the same molecule; and canonical molecules carrying
conflicting labels — disagreeing classes for classification endpoints, or
values differing by more than a fixed relative tolerance (1%) for
regression endpoints.

**The two tracks treat this audit differently, and the difference is
intentional.** In Track A1 the audit is *reported only*: the official rows
are consumed exactly as shipped, with no de-duplication or conflict
resolution, so that results remain comparable with the published benchmark
(§2.5). In Track A2 the audit is *enforced*: duplicate canonical molecules
are collapsed and conflicting molecules removed before partitioning (§2.6).
The same measurement therefore underlies both a comparability-preserving
and a stricter-curation evaluation, and the two are never merged.

**Invalid structures.** Records whose SMILES cannot be parsed by RDKit are
excluded, because no representation can be computed for them and their
inclusion would silently change the molecule set between representations.
Across the frozen release this affected **2 records, both in
`solubility_aqsoldb`** (of 9,982 records for that endpoint); all other
endpoints parsed completely. Under the full MolFusion curation applied in
Track A2, 81,809 official records across the 22 endpoints yield 79,712
usable molecules; per-endpoint accounting is reported in the Results and in
Supplementary Methods.

---

## 2.5 Track A1 — official TDC evaluation

Track A1 is the **primary, TDC-comparable, headline evaluation**. Its
purpose is to measure representation performance under exactly the
partition that other work using this benchmark reports against.

The official TDC ADMET protocol ships a fixed held-out test partition per
endpoint and provides a routine that re-splits only the remaining
train/validation pool. MolFusion's implementation was reconciled against
the official implementation, and Track A1 reproduces its semantics:

> For Track A1, the shipped TDC test partition was held fixed, whereas five
> deterministic scaffold-based train/validation realizations were generated
> using seeds 1–5.

The five seeds therefore index five train/validation realisations against
**one** test set. They do **not** generate five independent test sets, and
the five resulting scores for a given cell are five measurements of
model-selection variability on a common evaluation set, not five
independent replicates. Every statistical procedure in §2.10 depends on
this distinction.

**Partition structure.** The official release divides each endpoint into a
train/validation pool and a fixed test partition in approximately 80% / 20%
proportion. Within the pool, the official re-splitting routine allocates
**87.5% to training and 12.5% to validation** (fractions
`(0.875, 0.125, 0.0)`, the third component being zero because no further
test partition is drawn). This corresponds to approximately **70% train,
10% validation, 20% fixed test** of each endpoint. It should not be read as
an independently generated 70/10/20 partition: only the train/validation
boundary moves with the seed.

**Curation.** Track A1 applies **no cleaning**; official records are
consumed exactly as shipped. Duplicate and conflicting-label structure was
audited and is reported (§2.4, and in the Results), but was not silently
corrected, because doing so would alter the evaluation set and break
comparability with published results on this benchmark. The consequences of
the stricter alternative are measured separately in Track A2.

---

## 2.6 Track A2 — independent scaffold robustness evaluation

Track A2 is a **supplementary robustness analysis**. It asks whether
conclusions drawn under the official partition survive when the molecules
are partitioned differently and curated more strictly. It is **not external
validation** — the molecules are the same — and it does not replace Track
A1 as the headline evaluation.

**Partitioning.** Each endpoint was independently repartitioned into
approximately **70% train, 10% validation, 20% test** by Bemis–Murcko
scaffold [CITATION: Bemis–Murcko scaffold], using seeds **0–4**. Scaffolds
were computed with RDKit's Murcko scaffold implementation with
**`includeChirality = True`**, so stereoisomers sharing a constitutional
framework are assigned to distinct scaffolds. This differs from the
official TDC convention, which computes scaffolds without chirality; the
difference is a property of the two protocols and is recorded rather than
reconciled, since forcing agreement would defeat the purpose of an
independent repartitioning. Molecules sharing a scaffold are always
assigned to the same partition, so no scaffold spans a partition boundary.

**Curation.** Track A2 applies the full MolFusion curation policy described
in §2.4: duplicate canonical molecules collapsed, conflicting-label
molecules removed, unparseable structures excluded.

**A necessary caveat, stated here rather than deferred.** Track A2 changes
two things at once relative to Track A1 — the partitioning scheme *and* the
curation policy. **The effects of repartitioning and of stricter cleaning
are therefore not fully separable in this design.** A sensitivity analysis
excluding the endpoints most affected by curation is reported alongside the
main A2 analysis, but it bounds the confound rather than resolving it. A
design that varied the two independently would require a third track and
was not undertaken.

**Partition distinctness.** The frozen scaffold splitter orders scaffold
groups by descending size, with the seed permuting only within groups of
equal size. For endpoints whose largest scaffold groups already fill the
training and validation targets, the seed therefore moves the test
partition very little, and **not every endpoint produced five distinct test
partitions**. This behaviour was measured, not assumed: mean pairwise
Jaccard similarity between the five test molecule sets was computed for
every endpoint, and endpoints at or below a frozen similarity threshold of
0.50 were designated *genuinely repartitioned*. **19 of the 22 endpoints**
meet this criterion.

All 22 endpoints remain in the headline A2 analysis. The 19-endpoint set is
reported alongside it as a **sensitivity subset**, not as a replacement,
and no endpoint was removed from the benchmark on this basis. The subset
criterion and its threshold were fixed before the A2 analysis and no
alternative subset was introduced.

---

## 2.7 Predictive probes

Each representation was evaluated through two deliberately different model
families, referred to as **probes**. Using more than one probe is central
to the design: a representation's measured utility is a joint property of
the encoding and the model consuming it, and a single model family cannot
distinguish information that is absent from information that is present but
not accessible to that model.

**Linear probe.** Regularised logistic regression for classification
endpoints and ridge regression for regression endpoints
[CITATION: scikit-learn]. Logistic regression was fitted with an iteration
limit of 5,000 to allow convergence at low regularisation strength. This
probe measures predictive information accessible through a linear decision
function of the representation.

**Nonlinear probe.** Histogram-based gradient boosting —
`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` — for
classification and regression respectively. This probe measures predictive
information accessible after nonlinear feature interactions and
thresholding.

Performance under either probe is a statement about **predictive
accessibility under that probe**, not about the intrinsic information
content of the representation. Representation rankings are conditional on
the probe throughout, and are reported per probe rather than pooled.

**Equal budget.** Within a track, every representation received identical
treatment: the same two model families, the same hyperparameter grid and
tuning budget, the same endpoint partitions, and the same five seeds. No
representation-specific model, architecture, or enlarged search was
introduced for any encoding. Any performance difference between
representations is therefore attributable to the representation and its
required preprocessing, not to differences in model capacity or tuning
effort.

---

## 2.8 Hyperparameter selection and preprocessing

**Data usage.** The three partitions have strictly separated roles:

| Partition | Role |
| --- | --- |
| Train | fitting the model and every fitted preprocessing step |
| Validation | hyperparameter selection |
| Test | final evaluation only |

The test partition was never used for model fitting, preprocessing
estimation, hyperparameter selection, early stopping, or any other choice.
Each cell was evaluated on test exactly once, after selection had
concluded. Preprocessing and estimator are composed into a single
scikit-learn `Pipeline`, so that fitting occurs on the training fold alone
and is applied unchanged to validation and test; this makes the separation
structural rather than a matter of discipline.

**Hyperparameter grid.** Four candidates per probe and task type, selected
on validation performance under the endpoint's primary metric:

| Probe | Task | Candidates |
| --- | --- | --- |
| Linear | classification | `C` ∈ {0.01, 0.1, 1.0, 10.0} |
| Linear | regression | `alpha` ∈ {0.1, 1.0, 10.0, 100.0} |
| Nonlinear | classification and regression | `learning_rate` ∈ {0.05, 0.1} × `max_leaf_nodes` ∈ {15, 31} |

All other estimator settings remained at their library defaults, with the
seed fixed per realisation. If a candidate could not be scored on the
validation partition — for example because a validation fold contained a
single class — it was skipped rather than assigned a substitute score.

**Preprocessing is representation- and probe-specific, not uniform.**
Applying one pipeline to all seven encodings would have handicapped some of
them, so the policy was fixed per representation in advance:

| Probe | Steps |
| --- | --- |
| Linear | non-finite fold → median imputation → standardisation *for `rdkit_physchem_descriptors`, `erg_reduced_graph_315`, `rdkit_fragment_descriptors` only* |
| Nonlinear | non-finite fold only |

The binary fingerprints (`morgan_ecfp4_1024`, `maccs_keys_167`,
`avalon_1024`) already share a common 0/1 scale, and
**`smiles_tfidf_4096` is already L2-normalised by its frozen
representation contract**, so standardising them would add a fitted step
that changes nothing while creating an additional opportunity for
information to leak across partitions. Under the nonlinear probe no scaler
or imputer is applied at all: gradient boosting is scale-invariant and
consumes missing values natively.

The *non-finite fold* is a stateless transform mapping ±∞ to NaN. RDKit
descriptors legitimately emit NaN where a descriptor is undefined and, for
a small number of molecules, ±∞ where one diverges; both mean "not
computable". The fold makes them a single condition, which the imputer then
resolves from **training-split medians only**. Because the transform fits
nothing, it cannot carry information between partitions.

**Class imbalance.** **No class-imbalance handling was applied.** No class
weighting, resampling, SMOTE, or threshold adjustment was used; the
`class_weight` parameter was left unset for both classifiers. Imbalance is
instead addressed at the evaluation stage: AUPRC is reported for every
classification endpoint precisely because AUROC can appear healthy for a
model that ranks the minority class poorly (§2.9).

---

## 2.9 Evaluation metrics

**Classification.** Primary metric: **AUROC**. Secondary metrics: **AUPRC,
balanced accuracy, Matthews correlation coefficient**. AUPRC is reported
for every endpoint and is not optional, because ADMET classification
endpoints are frequently imbalanced.

**Regression.** Primary metric: **mean absolute error (MAE)**. Secondary
metrics: **RMSE, R², Spearman correlation**. MAE was chosen as primary
because it is interpretable in the endpoint's own units and is not
dominated by a small number of large errors; Spearman is reported alongside
it because, being unit-free, it remains interpretable when comparing
behaviour across endpoints measured on different scales.

Metric direction is recorded explicitly: MAE and RMSE are lower-is-better,
all other metrics higher-is-better. Direction is applied by the analysis
code rather than assumed by the reader.

### Cross-endpoint ranking

The 22 endpoints are measured on incompatible scales and in opposite
directions: an AUROC of 0.85 and an MAE of 0.42 are not commensurable, and
averaging them would be meaningless.

**Raw AUROC and MAE values were therefore never averaged across
endpoints.** Instead, for each endpoint, probe and seed, the seven
representations were ranked *within that endpoint* on its primary metric
after applying the metric's direction, with tied values receiving the
average of the positions they span. Ranks were then aggregated across the
five seeds, and only these within-endpoint ranks entered cross-endpoint
summaries.

Cross-endpoint summaries comprise **mean rank**, **median rank**, **win
count** (endpoints at which a representation attained rank 1), and
**top-3 frequency**. Ranking is performed separately for each probe, and
separately for classification and regression endpoints where a task-family
summary is reported; the three summaries are never combined.

---

## 2.10 Statistical analysis

**Unit of inference.** The **endpoint** is the unit of statistical
inference throughout. The five seed realisations within an endpoint are
**not** treated as five independent experimental units: in Track A1 they
share a single test partition (§2.5), and in Track A2 they are five views
of the same molecules. Treating them as independent would inflate the
effective sample size from 22 to 110 and understate every p-value. Seed
variation is used descriptively — to characterise variability — and never
as replication.

Analyses were performed separately for each probe, and A1 and A2 result
rows were never pooled into a single dataset.

**Omnibus test.** For each probe and task family, a **Friedman test**
[CITATION: Friedman test] was applied to the within-endpoint ranks of the
seven representations, with endpoints as blocks. This tests whether the
representations are exchangeable within that family.

**Pairwise comparison.** Pairwise tests were conducted **only within
families whose Friedman test rejected** at α = 0.05, avoiding pairwise
testing where the omnibus provided no evidence of any difference. Pairwise
comparisons used the **paired Wilcoxon signed-rank test**
[CITATION: Wilcoxon signed-rank test] over endpoint-level scores, with
**Holm correction** [CITATION: Holm correction] applied across the pairwise
family. Every pairwise result is reported with its **matched-pairs
rank-biserial correlation** as an effect size; significance is never
reported without magnitude and direction.

**Bootstrap intervals.** Uncertainty in cross-endpoint mean rank was
characterised by a nonparametric bootstrap with **10,000 resamples**, a
fixed random seed of **0**, and the **endpoint** as the resampling unit,
consistent with the unit of inference. These intervals are marginal
per-representation intervals, not a simultaneous band. They are used for
**uncertainty visualisation only**; interval overlap or non-overlap was not
used as a significance test, and all inferential statements rest on the
Friedman → Holm-corrected Wilcoxon → rank-biserial chain above.

**Track comparison.** Each track was aggregated under its own sampling
interpretation before any comparison: Track A1 as five train/validation
realisations against a fixed test partition, Track A2 as five independent
scaffold repartitions. **No seed-to-seed pairing was performed between
tracks.** Although the two seed sets overlap numerically (1–4 occur in
both), a seed value carries no cross-track meaning; rows are disambiguated
by track identifier and split identifier, never by seed. Comparison between
tracks was made at the level of derived endpoint-level and rank-level
summaries.

**Endpoint stability.** Because Track A2's five partitions are genuinely
different evaluation sets, the stability of a representation ordering can
be quantified there. **Kendall's coefficient of concordance (W)**
[CITATION: Kendall's W] was computed across the five partitions for each
endpoint and probe, with W = 1 indicating identical orderings across
partitions. An endpoint was designated low-stability if its **weaker**
probe fell below a frozen threshold of **W < 0.35** — taking the minimum
across probes so that a stable ordering under one probe cannot mask an
unstable one under the other.

Six endpoints were pre-registered as low-stability on this criterion:
**`herg`, `cyp2c9_substrate_carbonmangels`, `clearance_hepatocyte_az`,
`cyp2d6_substrate_carbonmangels`, `cyp3a4_substrate_carbonmangels`, and
`bioavailability_ma`**. These endpoints **remain in the overall benchmark
and in every cross-endpoint analysis**; the designation restricts only
endpoint-specific interpretation, for which their orderings do not provide
a stable basis.

A seventh endpoint, **`vdss_lombardo`**, falls marginally below the same
threshold. Because this was identified only after the analysis was run, it
is labelled **BORDERLINE** and reported transparently, but was **not
excluded**: widening a pre-registered exclusion set on the basis of
observed results would convert a pre-specified caveat into a post-hoc
filter.

Statistical computation used SciPy [CITATION: SciPy] and scikit-learn.

---

## 2.11 Computational cost measurement

Computational cost was recorded as a separate dimension of evidence and was
measured, not estimated. Three components were timed independently and are
never summed into a single figure of merit:

1. **Feature-generation cost** — wall-clock time to compute a
   representation's feature matrix for an endpoint, recorded on first
   computation. Matrices are cached and reused across probes and seeds, so
   the reported cost is the cost of computing each matrix once.
2. **Model-fitting cost** — wall-clock time for hyperparameter selection
   (four candidate fits per cell and seed) and for the final fit, recorded
   separately.
3. **Prediction cost** — wall-clock time to score the validation and test
   partitions.

**Hardware and execution context.** All benchmark execution was performed
on a single host, a 2-core Windows workstation, with two concurrent worker
processes; Track A2 required approximately 37.8 hours of wall-clock time at
that concurrency. Because timings were collected under a fixed
hyperparameter grid on one machine, they reflect relative cost within this
study and are not portable performance figures.

> Computational timings are intended for relative within-study comparison
> and were measured on the same execution host.

**Cost did not influence any predictive result.** Timing measurements are
excluded from the scientific identity of the result set (§2.12), are not an
input to any ranking, statistical test, or model-selection decision, and
did not alter any representation's rank. **No composite efficiency or
cost-adjusted performance score was defined**, because the exchange rate
between compute and predictive accuracy is application-specific and not
something this benchmark can establish.

---

## 2.12 Reproducibility, result identity, and provenance

**Frozen inputs.** Every benchmark run names the dataset release it
consumed by content-derived identity (§2.3), and refuses to proceed if the
release does not match the expected value.

**Result identity.** Each completed result set carries two distinct
digests: a *file digest* over the serialised result table, and a
**scientific identity** computed over only those columns that constitute a
scientific result — release, track, endpoint, split, seed, representation,
model family, probe, hyperparameters, metric, metric value, partition sizes
and feature accounting. Timing columns and cache-hit flags are deliberately
excluded, so that re-collecting results on a different machine reproduces
the same scientific identity even though wall-clock values differ. Analysis
outputs and the publication evidence package carry their own derived
identities in the same manner, each incorporating the identity of the
inputs it consumed.

**Deterministic caching and execution.** Feature matrices are cached under
a key covering the dataset release, endpoint, agent identity and version,
output dimension, canonicalisation contract, an order-sensitive digest of
the exact molecule list, and, for artifact-backed representations, the
artifact identity. A cached matrix is validated against this key before
reuse and never silently recomputed over a stale entry. Results are written
as atomic per-cell shards, validated before reuse, enabling interrupted
runs to resume without recomputation and without partially written output.

**Historical execution provenance.** The reported Track A1 and Track A2
results were produced by the benchmark runner as it existed at execution
time:

| | Commits |
| --- | --- |
| Track A1 execution | `459653b`, `ddabb42`, `2bcb467` |
| Track A2 execution | `e6ae297` |
| Analysis | `15b78a2` |
| Provenance hardening (subsequent) | `89335dc` |

A post-run provenance audit identified a logging defect whereby some result
shards lacked an embedded Git commit identifier. The numerical results and
their content-derived scientific identities were unaffected; run-level
metadata, immutable checksums, source-code audits, and historical execution
records were used to verify attribution. The provenance mechanism was
subsequently hardened for future executions. Exact per-shard counts are
given in Supplementary Methods.

**The hardened provenance mechanism did not produce the results reported
here.** It was implemented after execution completed. In subsequent
executions it captures execution provenance once in the orchestrating
process rather than independently in each worker, requires a non-null
commit identifier before any scientific work begins, distinguishes tracked
source modifications from unrelated untracked files, and propagates a
single immutable provenance record to all workers. **This mechanism applies
to subsequent executions and was not applied retroactively to the
historical Track A1 and Track A2 runs**, whose results were instead audited
as described above. No result value was reconstructed, and no shard was
retrospectively modified.

**Software versions.** Recorded in the frozen run reports of both tracks:
**Python 3.11.15, RDKit 2026.03.5, NumPy 2.4.6, scikit-learn 1.9.0**.
Statistical analysis additionally used **SciPy 1.17.1**; the SciPy version
is recorded in the analysis environment rather than in the execution run
report.

**Availability.** The framework, the frozen benchmark protocol, the
analysis code, and the complete result tables, together with per-endpoint
checksums, the full hyperparameter and cost tables, and the provenance
audit, are available in the project repository and in Supplementary
Material.
