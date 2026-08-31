# MolFusion benchmark protocol (Phase 6A, version 6A.1)

**Status:** frozen. No benchmark has been executed. This document and
`molfusion_backend/benchmark/protocol.py` are the two halves of one
specification — the prose here, the machine-readable constants there — and
tests keep them in step. It is written to be readable without the source,
and to become the Methods section of a MolFusion manuscript.

## 1. The question

Not *which pipeline wins a leaderboard*. A leaderboard result confounds the
representation with the model, the tuning budget and the split, and
optimising all four jointly answers a question nobody asked.

The benchmark asks:

> Given the **same** downstream model and the **same** tuning budget, how
> much predictive information does each fixed-length molecular
> representation make accessible, how stable is that across endpoints, and
> what does it cost to compute?

Redundancy and complementarity between representations (Phase 6B) and
fusion (Phase 6C) build on this; neither is in scope here.

## 2. Representation tracks

### Track A — seven fixed-length vectors

| Representation | Type | Dimension |
| --- | --- | --- |
| `morgan_ecfp4_1024` | binary | 1024 |
| `maccs_keys_167` | binary | 167 |
| `rdkit_physchem_descriptors` | continuous | 217 |
| `avalon_1024` | binary | 1024 |
| `erg_reduced_graph_315` | continuous | 315 |
| `rdkit_fragment_descriptors` | count | 85 |
| `smiles_tfidf_4096` | continuous | 4096 |

All seven are fixed-length numeric vectors, so one shared prediction head
can consume them unchanged. This is the primary benchmark.

### Track B — SELFIES, deferred

**`selfies_sequence` is excluded from Track A.** It is a variable-length
categorical token sequence, and entering it into a fixed-vector comparison
would first require choosing an encoder — bag-of-tokens, one-hot, a mean
embedding, a learned encoder. The resulting score would measure *that
encoder*, not the representation MolFusion ships, while appearing in the
same table as seven representations that received no such treatment.

A dedicated sequence-model track is future work. Its scores will not be
directly comparable with Track A without explicitly accounting for the
learned encoder and head, and any table combining them must say so.

## 3. Datasets

### Current inventory

A repository scan found **no molecular property datasets present locally**.
The only molecular data in the project is the ChEMBL 37 reference corpus
(`backend/corpus_data/chembl37/canonical_smiles.smi`, 2,897,639 canonical
SMILES), which is **unlabelled** and exists to fit the TF-IDF vocabulary.
It is not a benchmark dataset and is not used as one.

Phase 6A therefore designs acquisition rather than performing it.

### Recommended suite: the TDC ADMET benchmark group

| | |
| --- | --- |
| Source | Therapeutics Data Commons, ADMET benchmark group |
| Endpoints | 22 (9 regression, 13 classification) |
| Domain | absorption, distribution, metabolism, excretion, toxicity |
| Split | official scaffold split, 5 seeds, provided by the harness |
| Sizes | ~500 to ~13,000 molecules per endpoint |
| Acquisition | `PyTDC`, as a **dev-only** dependency |

Chosen because it satisfies the requirements simultaneously: it is an
established suite with documented provenance and licensing, it spans both
task types, its size (22 endpoints) supports a cross-task robustness claim
rather than an anecdote, and its official protocol is *already* a
scaffold split with five seeds — so following it costs nothing in rigour
and buys comparability with published numbers.

MoleculeNet was considered and not chosen as primary: several of its
endpoints have known label-quality problems, and its split conventions vary
between papers, which weakens exactly the cross-task comparison this
benchmark is for.

**Acquisition protocol.** Download once, canonicalize, record per-endpoint
SHA-256 and molecule counts, and freeze that manifest. The benchmark then
runs against the frozen local copy, never a live download — a run whose
inputs can change under it is not reproducible.

### Inclusion criteria

- structure available as a single SMILES string per record
- single-molecule property prediction (no reactions, no mixtures)
- one clearly defined target label per endpoint
- **≥ 100** usable molecules after validity filtering
- classification: **≥ 20** minority-class molecules
- parseable by the pinned RDKit build
- documented provenance and a research-permitting licence

The two thresholds are scientific, not cosmetic. A 20% test fold of 100
molecules is 20 compounds, where a single prediction moves AUROC by several
points; and below ~20 minority examples AUPRC is decided by a handful of
compounds. An endpoint failing a criterion is reported as **excluded with
the reason** — never dropped quietly, and never dropped for scoring badly.

## 4. Molecule identity and duplicates

Canonicalization reuses the frozen Phase 5F-A contract,
`rdkit_canonical_isomeric_smiles_v1`, so dataset identity and representation
input are canonicalized identically.

**No additional standardization.** No salt stripping, neutralization,
tautomer canonicalization, largest-fragment selection or stereo removal.
Each is a modelling decision that changes what is being predicted; applying
one here would benchmark a different dataset than the source defines. If a
source dataset itself specifies preprocessing, that is reproduced and
recorded per dataset.

**Duplicates.** Records are grouped by canonical SMILES:

| Case | Action |
| --- | --- |
| Same molecule, same label | collapse to one record, counted |
| Same molecule, conflicting labels | **drop every copy**, counted |

Dropping is the only reproducible option that adds nothing. Averaging
conflicting labels asserts a value the source never did; keeping one
arbitrarily makes the dataset depend on row order. The cost is reported so
it can be argued with. For regression, "conflicting" means differing by more
than 1% of the endpoint's own label spread — a relative tolerance, because
endpoints differ in units by orders of magnitude.

Deduplication happens **after** canonicalization. Deduplicating raw strings
would miss two spellings of one molecule, which is precisely the leakage
this prevents.

## 5. Splits

**Bemis–Murcko scaffold split is the default.** A random split asks whether
a model can interpolate among scaffolds it has already seen, which
overstates the generalisation that matters and rewards memorising a
dataset's scaffold inventory. Whole scaffold groups are assigned to one
partition, so no scaffold straddles the train/test boundary.

Where a dataset publishes an **official split**, that is used instead, so
results stay comparable with published numbers. The split source is
recorded per dataset and the two are never mixed silently.

- Fractions: **70 / 10 / 20** train / validation / test (approximate —
  a partition boundary never splits a scaffold group)
- Acyclic molecules have an empty Bemis–Murcko scaffold and are grouped
  under one explicit key; that group's size is reported per dataset
- Determinism is structural: group order is a hash of *(seed, scaffold)*,
  so assignment cannot drift with input order, hash randomisation, or
  dictionary iteration

**Seeds.** Five deterministic splits, seeds `0–4`, with model training fixed
at seed `0`. Split variability and training variability are deliberately
separated — the reported spread is data variability alone, not the two
mixed. A training-seed sweep is a different experiment.

## 6. Downstream models

The same model families and the same tuning budget for every
representation. Letting one representation receive a random forest and
another a neural network would measure the pairing, not the representation.

### Two probes, because they answer different questions

| Probe | Classification | Regression |
| --- | --- | --- |
| Linear | `LogisticRegression` | `Ridge` |
| Nonlinear | `HistGradientBoostingClassifier` | `HistGradientBoostingRegressor` |

The **linear** probe measures how much predictive information is *linearly
accessible* — a property of the representation's geometry. The
**nonlinear** probe measures what a capable tabular model can extract
regardless of geometry. A representation only the second can use is
informative but poorly shaped, and one number would hide that.

HistGradientBoosting is the nonlinear family because it is already present
via scikit-learn, is scale-invariant, handles the NaNs RDKit descriptors
legitimately produce natively, and copes with 4096 columns. Adding XGBoost
or LightGBM would mean a new heavy dependency for no argued scientific gain.

### Tuning budget

| Probe | Grid | Candidates |
| --- | --- | --- |
| Linear (classification) | `C ∈ {0.01, 0.1, 1, 10}` | 4 |
| Linear (regression) | `alpha ∈ {0.1, 1, 10, 100}` | 4 |
| Nonlinear | `learning_rate ∈ {0.05, 0.1}` × `max_leaf_nodes ∈ {15, 31}` | 4 |

Identical for every representation — the grid function does not accept a
representation, so an unequal budget cannot be introduced by accident.
Selection is on the **validation split only**, by the endpoint's primary
metric. **The test partition is read exactly once**, after selection.

The grids are deliberately small. The study compares representations, and an
open-ended or unequal search lets tuning effort masquerade as representation
quality.

## 7. Preprocessing

Chosen per representation *and* per probe, never uniformly:

| Representation | Linear probe | Nonlinear probe |
| --- | --- | --- |
| Morgan, MACCS, Avalon (binary) | none | none |
| RDKit descriptors (continuous) | standardize | none |
| ErG (continuous) | standardize | none |
| RDKit fragments (count) | standardize | none |
| TF-IDF (L2-normalized) | none | none |

Standardizing a bit vector destroys sparsity and gives an *absent* bit a
nonzero value, which is not what absence means. Leaving physicochemical
descriptors unscaled lets one descriptor's units dominate a penalized linear
model. TF-IDF is already L2-normalized by its frozen weighting contract, and
rescaling would undo a deliberate part of the representation. Trees are
invariant to monotone rescaling, so any scaler there would be a fitted step
that changes nothing while adding a place for leakage to hide.

**NaN.** RDKit descriptors emit NaN where a descriptor cannot be computed.
The linear probe imputes with the **training split's** median; the nonlinear
probe consumes NaN natively and no value is invented for it.

**Every fitted step sees the training split only.** This is enforced
structurally by `sklearn.Pipeline` rather than by remembering: `fit` fits the
steps on training data, and validation/test are only transformed.

## 8. Metrics

| | Primary | Secondary |
| --- | --- | --- |
| Classification | **AUROC** | **AUPRC** (mandatory), balanced accuracy, MCC |
| Regression | **MAE** | **Spearman** (mandatory), RMSE, R² |

AUROC is primary by convention, but **AUPRC is reported for every
classification endpoint and is not optional**: ADMET endpoints are often
heavily imbalanced, and AUROC can look respectable for a model that ranks
the minority class badly — usually the class the endpoint exists to find.

MAE is primary for regression because it is interpretable in the endpoint's
own units and is not dominated by a few large errors. **Spearman always
travels with it** because MAE is meaningless to compare across endpoints
measured in different units, and a suite spanning log solubility and percent
protein binding needs at least one metric that is unit-free.

Where a source suite defines its own official per-endpoint metric, that is
additionally recorded so results remain comparable with published
leaderboards; the uniform set above is what the protocol ranks on, so the
comparison stays internally consistent.

## 9. Cross-endpoint aggregation

**Raw AUROC and MAE are never averaged together.** They have different
units, directions and attainable ranges, and their mean is not a quantity.

Representations are ranked 1–7 **within each endpoint** on that endpoint's
primary metric after direction normalization, and only the ranks are
aggregated. Ties receive the average rank. Reported aggregates: **mean rank,
median rank, win count**.

## 10. Statistics

Comparisons are paired — every representation is evaluated on the same
endpoints and the same splits.

1. **Friedman omnibus** across all seven representations
2. Pairwise **Wilcoxon signed-rank** tests, *only if the omnibus rejects*
3. **Holm correction** over the 21 pairwise comparisons
4. **Matched-pairs rank-biserial correlation** as effect size
5. **Bootstrap confidence intervals** over endpoints (10,000 resamples)

α = 0.05. **No p-value is reported without an effect size.** A p-value says
an ordering is detectable, not that it is large enough to act on, and with
seven representations the number of possible pairwise claims makes
uncorrected testing meaningless.

**Uncertainty.** Split-level scores are retained, never only their mean.
Each cell reports mean, standard deviation and a 95% interval across splits,
and the raw per-split rows stay in the result table so later analysis
recomputes rather than trusts.

## 11. Class imbalance

**No synthetic resampling.** SMOTE and its relatives invent molecules that
do not exist, which is indefensible in a chemical benchmark and alters
training data in a representation-dependent way.

Imbalance is characterised and reported per endpoint. Where handling is
needed, `class_weight="balanced"` is a model-side option applied identically
to every representation, **training-side only**. The test distribution is
never altered.

## 12. Failures and the evaluation universe

The evaluation universe for an endpoint is fixed **before any model is
fitted**: molecules RDKit parses, deduplicated by canonical SMILES, and for
which **every** Track A representation succeeds.

Comparing representations on different molecule sets would let one look
better by having failed on the hard compounds. Per-representation failure
counts are reported. If an endpoint loses more than **1%** of its molecules
to the intersection, that is flagged and a per-representation full-set
sensitivity analysis is reported alongside the primary result.

Recorded per dataset and agent: input records, RDKit-invalid, duplicates
collapsed, conflicting duplicates dropped, missing labels dropped,
representation failures, and the final common evaluation set.

## 13. External corpus exposure

`smiles_tfidf_4096` carries a vocabulary and IDF fitted on the frozen ChEMBL
37 corpus, so some benchmark molecules may have contributed to it. This is
**unsupervised exposure, not label leakage** — no benchmark label was ever
read — but it is an asymmetry the other six representations do not have.

The audit canonicalizes benchmark molecules with the frozen contract and
counts membership in the corpus, reporting overlap per endpoint.
**Overlapping molecules are never removed**: changing the benchmark to suit
one representation would be a worse distortion than the exposure it
corrects. The audit reads no labels.

**The TF-IDF artifact is frozen and never refitted** on benchmark data, per
endpoint or otherwise. Refitting would evaluate a different representation
from the one MolFusion ships — and one fitted on the evaluation
distribution at that. The artifact identity
`smiles_tfidf/chembl37_token_ngrams_1_3/1.0.0` is recorded in every run
manifest.

## 14. Cost

Feature generation cost and model cost are recorded **separately
throughout**; a blended number would hide which dominates.

| Per representation | Per model fit |
| --- | --- |
| feature computation time (total and per molecule) | fit time |
| dimension, dtype | predict time |
| sparsity / nonzero fraction | selected hyperparameters |

## 15. Results and reproducibility

**Long format**, one row per *(dataset, endpoint, task, split, seed,
representation, model, metric)*. Wide tables are projections generated later
for publication; the long rows are the source of truth. A wide table would
have to be rewritten whenever the representation set changed, could not hold
per-split rows, and would force an aggregation choice at write time.

Columns: `protocol_version, dataset, endpoint, task_type, split_id,
split_strategy, seed, representation, representation_version, model, probe,
metric, value, n_train, n_valid, n_test, feature_dim, feature_failures,
hyperparameters, fit_seconds, predict_seconds, feature_seconds`.

Each run also writes a manifest recording protocol version, dataset identity
and checksums, split identifiers, canonicalization ID, agent IDs and
versions, artifact identity, Python / RDKit / NumPy / scikit-learn / SciPy
versions, MolFusion git commit and working-tree cleanliness, and the
timestamp. Nothing depends on notebook state.

## 16. Feature cache contract

Representations are deterministic, so caching across repeated splits is
safe — but only if the key covers everything the value depends on. The key
is a SHA-256 over:

```
cache_key_version │ normalization_id │ agent_id │ agent_version │
artifact_identity │ canonical_smiles
```

separated by `0x1f` so no component can impersonate another. It is
explicitly **not** keyed on filename, dataset, split, label or row order. A
filename-keyed cache would happily serve a vector computed by a different
agent version or a superseded artifact with nothing downstream able to
notice. Phase 6A freezes the key; the store is later work.

## 17. Planned outputs

**Tables.** A: representation specification (type, dimension, dtype,
sparsity, generation cost). B: per-endpoint predictive performance with
split-level uncertainty. C: cross-endpoint mean/median rank and wins.
D: performance against computational cost.

**Figures.** 1: endpoint × representation performance heatmap. 2: mean rank
across endpoints with confidence intervals. 3: performance against
dimension and against compute cost. 4: classification and regression
behaviour shown separately, never pooled.

## 18. What Phase 6A did not do

No benchmark was executed. No dataset was downloaded. No representation was
added or modified. The protocol validation used synthetic fixtures whose
scores are wiring evidence and carry no scientific meaning.

Execution — all endpoints × seven representations × two probes × five
splits — is the next phase.
