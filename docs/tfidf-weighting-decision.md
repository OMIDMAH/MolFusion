# Decision record: SMILES TF-IDF weighting contract

**Scope:** this record decides **how a retained feature becomes a number**.
Which token n-grams are features was decided separately in
[`tfidf-vocabulary-decision.md`](tfidf-vocabulary-decision.md) (Phase
5F-C) and is not reopened here. Packaging both into a reproducible
artifact is Phase 5F-D.

**Status:** recommended, awaiting review. **Nothing is fitted or frozen.**
No vocabulary payload, IDF payload, artifact metadata, feature agent, or
registry entry exists.

**Evidence:** the Phase 5F-C.1 study, run 2026-08-30 against the frozen
ChEMBL 37 corpus (`fit_corpus_sha256`
`b2c4b81160df05c95f8421582bb4b1c95fdf5964a4edaff24a7c1ddd43e2a5de`,
verified at run start; 2,897,639 documents). Method is documented in
[`reproducibility.md`](reproducibility.md); tables are in the git-ignored
`backend/corpus_data/chembl37/studies/tfidf_weighting/`.

```
python -m molfusion_backend.corpus.study.weighting \
    --corpus     backend/corpus_data/chembl37/canonical_smiles.smi \
    --output-dir backend/corpus_data/chembl37/studies/tfidf_weighting
```

## The recommendation

| Parameter | Recommendation |
| --- | --- |
| `tf_mode` | **sublinear** — `1 + ln(count)` for `count > 0`, else `0` |
| `use_idf` | **true** |
| `idf_mode` / `smooth_idf` | **smoothed** — `ln((1 + N) / (1 + df(t))) + 1` |
| `norm` | **l2**, zero-safe |
| `internal_arithmetic_dtype` | **float64** |
| `internal_idf_dtype` (artifact storage) | **float64** |
| `runtime_output_dtype` | **float32** |
| Vector index ordering | **lexicographic token tuple, after DF selection** |
| Vocabulary encoding | **JSON array of token strings** |
| Runtime transformer | **MolFusion-owned NumPy** (not sklearn) |

Order of operations is fixed: `tf(counts) → multiply by idf → normalize`.
Logarithms are natural. The provisional expectation stated in the brief
(sublinear / use_idf / smooth_idf / l2) is **accepted**, and each part is
now backed by a measurement rather than convention.

## The vocabulary these numbers apply to

Deriving the frozen Phase 5F-C rule against the **full corpus** (not the
5F-C analysis subset):

| | |
| --- | --- |
| eligible terms at `min_df = 5` | **9,383** |
| selected terms | **4,096** |
| cap is binding | **yes** |
| effective `min_df` imposed by the cap | **133** |
| composition | 85 unigrams, 689 bigrams, 3,322 trigrams |
| DF range in vocabulary | 133 … 2,882,503 |
| IDF range (smoothed) | 1.005237 … 10.981567 |

The 9,383 figure reproduces the Phase 5F-C corpus-scope prediction
exactly. This is why artifact metadata must show `min_df` **and**
`dimension`: `min_df = 5` alone would leave 9,383 features, so quoting it
as if it produced the dimension would misstate the pruning by more than a
factor of two.

## Why sublinear TF

Raw counts are not bounded in any useful way. In the sampled molecules the
single most frequent feature reaches:

| stratum | median tokens | mean nonzero features | mean max feature TF | p99 | max |
| --- | --- | --- | --- | --- | --- |
| small | 28 | 50.4 | 8.9 | 16 | 28 |
| typical | 47 | 78.6 | 13.9 | 24 | 33 |
| large | 75 | 110.6 | 21.9 | 53 | 81 |
| very_long | 350 | 135.6 | 90.1 | 173 | **319** |

One motif occurring 319 times in one molecule is a syntactic fact about a
long SMILES string, not 319 times the chemical evidence. The decisive
measurement is how much of a molecule's own weight that single feature
takes, and how that changes with length:

| stratum | raw top-share | sublinear top-share | raw / sublinear |
| --- | --- | --- | --- |
| small | 0.11480 | 0.05132 | 2.24× |
| typical | 0.10126 | 0.03538 | 2.86× |
| large | 0.08630 | 0.02540 | 3.40× |
| very_long | 0.07682 | 0.01788 | **4.30×** |

The same pattern in the Herfindahl index (1.56× → 1.82× → 2.09× →
2.59×). Raw TF's excess concentration over sublinear **grows
monotonically with molecule length**. That is precisely the
length-dependence Phase 5F-C found in corpus TF *rankings*, reappearing
inside individual vectors, and sublinear damping is what removes it. Under
sublinear TF no single feature exceeds ~2.5% of a long molecule's weight
at p99; under raw TF it reaches ~11%.

Note the raw top-share *falls* with length in absolute terms (0.115 →
0.077) simply because long molecules have more nonzero features to divide
weight among. That is why the ratio, not the level, is the meaningful
statistic here — reading the level alone would suggest, wrongly, that long
molecules are less concentrated.

## Why smoothed IDF, even though nothing forces it

With `min_df = 5` (effective 133) there is no zero-DF term, so smoothing
is **not** needed to avoid division by zero. The reflex justification does
not apply and the choice has to stand on numbers:

| | smoothed | unsmoothed |
| --- | --- | --- |
| min | 1.005237 | 1.005237 |
| median | 8.396163 | 8.396725 |
| mean | 7.893409 | 7.894917 |
| max | 10.981567 | 10.989058 |

| absolute difference | value |
| --- | --- |
| mean | 0.001507 |
| median | 0.000562 |
| p95 | 0.006042 |
| max | **0.007490** |
| max *relative* difference | **0.068%** |
| rank order identical | **yes** |

Where the difference lives, by document frequency:

| DF band | terms | mean abs. difference | max abs. difference |
| --- | --- | --- | --- |
| (100, 1 000] | 1,636 | 3.41e-3 | 7.49e-3 |
| (1 000, 10 000] | 1,308 | 4.27e-4 | 9.98e-4 |
| (10 000, 100 000] | 769 | 4.35e-5 | 9.93e-5 |
| (100 000, 1 000 000] | 328 | 4.47e-6 | 9.62e-6 |
| (1 000 000, ∞) | 55 | 2.70e-7 | 6.54e-7 |

**Conclusion: the difference is real but numerically trivial**, and it is
concentrated entirely in the rarest surviving terms — exactly the terms
whose DF estimate is least stable. Both formulas are strictly decreasing
in DF, so they induce the *same* term ordering; only the spacing differs,
by at most 0.068%.

So smoothing is chosen not because it changes results but because it
costs nothing and buys stability at the boundary the vocabulary is
pruned at. It is a convention-and-reproducibility choice, and this record
says so plainly rather than implying a numerical necessity that the
measurement does not support. Had the difference been material, the
argument would have had to be made on other grounds.

## Why L2 normalization

This is the least ambiguous result in the phase. Without normalization,
vector magnitude is essentially a measurement of molecule size:

| TF mode | median magnitude | max | spread | Pearson vs token count | Spearman | Pearson vs SMILES length |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 45.4 | 1872.9 | 41.2× | **0.98428** | 0.97558 | 0.98754 |
| sublinear | 34.4 | 233.0 | 6.8× | **0.91494** | 0.94476 | 0.92518 |

Within a single stratum, where length varies only across a narrow band,
the association persists:

| stratum | raw Pearson | sublinear Pearson |
| --- | --- | --- |
| small (≤32 tokens) | 0.519 | 0.387 |
| typical (33–64) | 0.742 | 0.587 |
| large (65–256) | 0.920 | 0.692 |
| very_long (>256) | 0.927 | 0.669 |

(The pooled correlations are computed over a deliberately size-stratified
sample that over-represents long molecules, so the within-stratum figures
are the conservative ones. Both point the same way.)

**Answer to the question the brief posed: yes.** `norm=None` would make
the first principal direction of the representation essentially "how big
is this molecule", a fact already available for free from a token count
and one that would dominate any distance or linear model built on the
vectors. Sublinear TF reduces the effect but does not remove it (0.915 is
not meaningfully better than 0.984 for this purpose). L2 removes it *by
construction*: every molecule retaining any term has magnitude exactly
1.0, so size cannot be encoded in magnitude at all. `norm=None` must not
be the default representation contract.

L1 was not studied further; nothing in these results suggests a reason to
prefer it, and L2 is what cosine-similarity consumers expect.

## Precision: float64 internally and stored, float32 emitted

All arithmetic in float64, always. The two open questions were storage and
output.

**Runtime output as float32** — measured on 20,635 molecules, comparing
the float64 result against the same result rounded to float32:

| stratum | max element diff | mean element diff | L2 vector diff (max) | min cosine |
| --- | --- | --- | --- | --- |
| small | 2.95e-8 | 2.78e-9 | 3.94e-8 | 0.999999999999999 |
| typical | 1.49e-8 | 2.25e-9 | 3.42e-8 | 0.999999999999999 |
| large | 1.49e-8 | 1.92e-9 | 3.31e-8 | 0.999999999999999 |
| very_long | 1.49e-8 | 1.71e-9 | 3.37e-8 | 0.999999999999999 |

Cosine similarity is 1.0 to fifteen decimal places in every stratum. A
float32 output halves the payload of every API response and every CSV
export for a difference around 1e-8 — far below any chemically or
statistically meaningful threshold. **Accepted.**

**Artifact IDF as float64** — this is the opposite call, and it is worth
separating because the brief warned against accepting the pairing
automatically. Storing the IDF vector as float32 would:

| | |
| --- | --- |
| save | **16,384 bytes** (32,768 → 16,384) |
| cost | max absolute IDF error **4.77e-7**, max relative error 5.91e-8 |
| worst downstream cosine (2,000 synthetic supports) | 0.999999999999999 |

The downstream effect is as negligible as the output cast. The saving,
however, is 16 KB on an artifact that will also carry a 4,096-term
vocabulary payload — a rounding error in the artifact's own size. There is
no reason to introduce a lossy step into the *stored, canonical* value
when the entire benefit is 16 KB. Storage keeps full precision; the lossy
step happens once, at the boundary where it buys something.

So the asymmetry is deliberate and justified separately at each end:
**store exactly, emit cheaply.**

## Zero-vector semantics

```
valid molecule + successful tokenization + no retained vocabulary term
    -> all-zero vector of dimension 4,096
```

This is a **valid result, not a representation failure**, and must not be
reported as an error. L2 normalization of a zero vector leaves it exactly
zero: the implementation substitutes 1 for a zero norm before dividing, so
no `NaN` or `Inf` can be produced. Dividing a zero vector by its own norm
would create `NaN`s that propagate silently through every downstream
similarity, mean, and model input — a failure that surfaces far from its
cause.

Phase 5F-C measured zero all-zero molecules in the ChEMBL holdout at this
configuration, and this phase measured **zero** among 20,635 stratified
sample molecules. But arbitrary future input is not ChEMBL, so the
behaviour is specified rather than assumed away, and is covered by tests.

## Out-of-vocabulary semantics

An n-gram outside the frozen 4,096-term vocabulary **contributes
nothing**. It does not expand the vocabulary, does not receive an `UNK`
dimension, does not trigger fitting, and does not raise.

This is explicitly distinct from tokenization failure. A string MolFusion
cannot tokenize violates the Phase 5F-A contract and remains an error;
a molecule composed of unfamiliar motifs is simply a molecule, and
vocabulary OOV is normal runtime behaviour.

## Vector index ordering: lexicographic, after selection

Selection and indexing are separate rules, and the recommendation is the
one the brief expected — **select by DF ranking, then index the selected
features lexicographically** — with no reason found against it.

| | ranking order | lexicographic order |
| --- | --- | --- |
| column 0 means | the highest-DF term | the lexicographically first term |
| stable under a DF re-fit? | **no** — any DF shift can permute columns | **yes** — only membership changes matter |
| auditable by eye? | needs the DF column to verify | sorted, diffable directly |

The deciding argument is stability. Under ranking order, re-fitting on a
later ChEMBL release would permute columns even for terms whose membership
never changed, because two terms swapping DF by one document swaps their
indices. Under lexicographic order the index of a surviving term depends
only on which *other* terms are in the vocabulary, so a re-fit that keeps
the same term set produces the same column layout regardless of how DF
moved. That makes two artifact versions comparable column-by-column, which
matters for anything cached, compared, or persisted downstream.

The counter-argument for ranking order — "the most important features come
first" — has no operational value here: the vector is dense and fixed at
4,096, nothing truncates it, and every consumer indexes by name through
the vocabulary map.

Verified in the study output: index 0 is `["#"]` (DF 203,631, selection
rank 237) and index 4,095 is `["s","s","c"]` (DF 283, selection rank
3,416). The index order is demonstrably independent of DF, which is the
property being bought.

Each term retains its `selection_rank` alongside its `index`, so the
ranking that chose a feature stays auditable after the ordering that
positions it.

## Vocabulary serialization

Each feature is a **JSON array of token strings**:

```json
{"index": 123, "tokens": ["C", "(", "="], "order": 3, "document_frequency": 1048576}
```

Never a joined string. `("Cl","C")` and `("C","lC")` both concatenate to
`"ClC"`, so a concatenated key cannot distinguish two genuinely different
features; a separator character only relocates the problem to any token
containing it. A JSON array has no separator to collide with and
round-trips exactly — including for tokens containing spaces, commas,
quotes, or brackets, which is covered by tests.

`order` is redundant with `len(tokens)`, and is stored anyway so the file
is self-checking: a loader can reject a payload whose records disagree
with themselves rather than silently indexing a corrupted vocabulary.

## Runtime: MolFusion-owned NumPy, not sklearn

**Recommendation: option B.**

By the time the artifact exists, MolFusion owns the tokenizer, the
vocabulary, the index ordering, the IDF vector, the TF rule, and the
normalization rule. sklearn would contribute only `tf * idf` followed by a
division — arithmetic that is roughly fifteen lines, already implemented
in `study/weighting/weights.py`, and already pinned by hand-derived tests.

Concretely:

- **sklearn is not a MolFusion dependency and would be a new one.** The
  backend declares fastapi, pydantic, rdkit, selfies and uvicorn. Adding
  sklearn pulls in scipy, joblib and threadpoolctl — tens of megabytes —
  to perform a multiply and a divide.
- **NumPy already is one in practice.** Every existing feature agent
  (`base`, `morgan`, `maccs`, `avalon`, `erg`, `fragments`,
  `descriptors`) imports NumPy directly today, so a NumPy transformer adds
  no new exposure.
- **sklearn would not do the tokenization anyway.** MolFusion's analyzer
  is its own frozen tokenizer plus n-gram generation, passed in as a
  callable. sklearn's own analyzer and `token_pattern` are explicitly out
  of scope.
- **Artifact auditability.** A float64 `.npy` IDF array plus a vocabulary
  JSON can be read by anyone. A pickled or joblib-serialized vectorizer
  cannot, and unpickling across sklearn versions is a known fragility —
  which conflicts directly with an artifact meant to be rebuildable years
  later.
- **Injecting a frozen vocabulary and IDF into `TfidfVectorizer` at
  transform time** means setting private-ish state (`vocabulary_`,
  `_tfidf.idf_`), whose form has changed across sklearn releases. Pinning
  that is pinning an internal, not a contract.

The disadvantage the brief names — reimplementing standard mathematics
incorrectly — is real and is why this phase wrote the arithmetic out in
full and pinned it against a fully worked example computed by hand
(`test_corpus_study_weighting_parity.py`: a three-document corpus with
every DF, IDF, TF and normalized output derived from the formulas with
plain `math`, and the IDF constants written as decimal literals).

**Requirement carried to Phase 5F-D:** add a parity test against sklearn's
`TfidfTransformer` as a **dev-only, skip-if-absent** dependency. The
recommended contract corresponds exactly to
`TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)`
applied to a count matrix, so parity is directly checkable without sklearn
becoming a runtime dependency.

## What Phase 5F-D should do

1. Freeze the vocabulary: top 4,096 `(1,3)` terms of the full corpus by
   `(-document_frequency, ngram_tuple)` after `min_df = 5`; index
   lexicographically; serialize each term as a JSON token array.
2. Compute IDF as `ln((1 + N) / (1 + df)) + 1` with `N = 2,897,639`, store
   as float64.
3. Record in artifact metadata: `fit_corpus_sha256`, contract IDs,
   `ngram_orders`, `min_df` **and** `dimension` (both — the cap binds),
   the selection ranking, the index ordering, and the full weighting
   contract including `tf_mode`, `smooth_idf`, `norm`, log base, order of
   operations, and both dtypes.
4. Verify a deterministic rebuild reproduces the vocabulary and IDF
   byte-for-byte.
5. Add the sklearn parity test as a skip-if-absent dev dependency.

## Issues carried forward

- **NumPy is undeclared.** Every existing agent imports it, and it arrives
  transitively via RDKit, but `pyproject.toml` does not list it. A runtime
  transformer makes this load-bearing; 5F-D should declare it explicitly.
  This is pre-existing, not introduced here.
- **The effective `min_df` is 133, not 5.** Metadata that shows only
  `min_df = 5` understates the pruning by more than a factor of two
  (9,383 eligible → 4,096 selected). Both numbers must appear.
- **Phase 5F-C's effective `min_df` was 126, measured on the 95% analysis
  subset; the full-corpus value is 133.** The two are consistent, and the
  full-corpus value is the one the production artifact must record.
- **This phase measures numerics, not predictive utility.** Sublinear vs
  raw TF and smoothed vs unsmoothed IDF are justified on stability and
  length-robustness grounds using ChEMBL alone, since no downstream label
  was read. If a later phase evaluates these on real endpoints and finds a
  different ordering, this record should be revised rather than defended.

## Reproducibility of the study run

| | |
| --- | --- |
| Python | 3.11.15 |
| RDKit | 2026.03.5 |
| NumPy | 2.4.6 |
| sklearn | not installed |
| MolFusion commit | `e5d7730` |
| corpus pass | 1,643 s |
| diagnostics pass | 38 s |
| total | 1,687 s |
| sample | 20,635 molecules (5,001 / 5,205 / 5,267 / 5,162 by stratum) |
| all-zero molecules in sample | 0 |
