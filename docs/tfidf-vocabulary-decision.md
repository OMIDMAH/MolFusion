# Decision record: SMILES TF-IDF vocabulary policy

**Status:** recommended, awaiting review. **Nothing is fitted or frozen.**
No vectorizer, IDF vector, vocabulary payload, artifact, agent, or registry
entry exists yet; Phase 5F-D creates those, and only after this policy is
accepted.

**Evidence:** the Phase 5F-C study, run on 2026-08-30 against the frozen
ChEMBL 37 reference corpus. Method, contracts and output schema are
documented in [`reproducibility.md`](reproducibility.md); the raw tables
live in the (git-ignored) study output directory
`backend/corpus_data/chembl37/studies/ngram_vocabulary/`. Every number
below is reproducible with:

```
python -m molfusion_backend.corpus.study \
    --corpus     backend/corpus_data/chembl37/canonical_smiles.smi \
    --output-dir backend/corpus_data/chembl37/studies/ngram_vocabulary
```

## Corpus the decision rests on

| | |
| --- | --- |
| `fit_corpus_sha256` | `b2c4b81160df05c95f8421582bb4b1c95fdf5964a4edaff24a7c1ddd43e2a5de` (verified at run start) |
| documents | 2,897,639 |
| tokens | 153,258,518 |
| `normalization_id` | `rdkit_canonical_isomeric_smiles_v1` |
| `tokenizer_id` | `rdkit_smiles_lexer_v1` |
| study split | 2,752,823 fit / 144,816 holdout (4.998%), hash-defined |

Study fit tokens (145,592,643) plus holdout tokens (7,665,875) reproduce
the frozen corpus total exactly.

## The recommendation

| Decision | Recommendation |
| --- | --- |
| `ngram_range` | **(1, 3)** |
| `min_df` | **5** (absolute molecule count) as a stated rarity floor |
| Dimension cap needed? | **Yes** — not for feasibility, for value per dimension |
| Dimension | **4,096** |
| Feature ranking | **descending document frequency**, ties broken by the lexicographic n-gram token tuple |
| Protect all unigrams? | **No** |
| Chosen by sklearn? | **No** — MolFusion freezes its own vocabulary |

Under DF ranking the two knobs are one knob: `{DF >= min_df}` is exactly a
prefix of the ranking. On this corpus the 4,096-term cap binds and imposes
an **effective `min_df` of 126**; the stated floor of 5 is currently inert
and exists so the semantic ("a motif seen in fewer than 5 of 2.9M
molecules never becomes a feature") survives a future corpus in which the
cap does not bind.

## Why

### The vocabulary is small, so nothing is forced

| order | distinct n-grams | `DF <= 1` | `DF <= 10` |
| --- | --- | --- | --- |
| 1 | 291 | 65 (22.3%) | 153 (52.6%) |
| 2 | 2,517 | 419 (16.7%) | 1,247 (49.5%) |
| 3 | 12,404 | 2,231 (18.0%) | 6,179 (49.8%) |

The full `(1,3)` vocabulary is **15,212 terms** (15,109 on the fit
subset) — already below the largest candidate dimension. So a cap is not
needed to make the representation tractable, and the cap question becomes
purely one of value, not feasibility. Roughly half of every order is
`DF <= 10`, a long tail by count that carries almost no mass.

### The unseen-molecule OOV floor is negligible

Motifs the holdout contains that the 95% fit subset never saw:

| order | unseen distinct n-grams | unseen occurrences | of holdout occurrences |
| --- | --- | --- | --- |
| 1 | 2 | 3 | 0.00004% |
| 2 | 15 | 18 | 0.00024% |
| 3 | 86 | 89 | 0.00121% |

No pruning policy can recover these, and no pruning policy needs to. On a
ChEMBL-only estimate, unseen-molecule OOV is a non-problem; the OOV a
vocabulary actually suffers is entirely self-inflicted by pruning, which
makes the pruning choice the whole decision.

### `(1,3)`, not `(1,1)`, `(1,2)` or `(2,3)`

Measured on the holdout at each policy's full vocabulary:

| policy | dimension | mean nonzero features | all-zero molecules | occurrence coverage |
| --- | --- | --- | --- | --- |
| `(1,1)` | 289 | 12.10 | 0 | 1.000000 |
| `(1,2)` | 2,791 | 41.19 | 0 | 0.999999 |
| `(1,3)` | 15,109 | 80.65 | 0 | 0.999995 |
| `(2,3)` | 14,820 | 68.56 | **1** | 0.999993 |

- **`(1,1)` is rejected as too coarse.** 289 features and 12 nonzero per
  molecule is an atom/bond-symbol histogram: it encodes composition and no
  local connectivity at all. Its perfect coverage is trivial, not a merit —
  a vocabulary of 289 symbols covers everything because the alphabet is
  the vocabulary.
- **`(2,3)` is rejected on all-zero risk.** Dropping order 1 makes
  single-token molecules structurally unrepresentable: one holdout molecule
  has no n-gram of order 2 or 3 at all, and 4 molecules are all-zero at
  1,024 dimensions. This is a correctness failure, not a coverage
  shortfall — an all-zero vector is indistinguishable from every other
  all-zero vector. It remains a useful diagnostic: it shows that removing
  isolated token composition costs both representation (68.56 vs 80.65
  nonzero) and safety, which is the argument for keeping unigrams in.
- **`(1,3)` over `(1,2)`** doubles the nonzero features per molecule
  (41.19 → 80.65) at zero all-zero risk. Trigrams resolve local context —
  branch openings, ring closures, bond-atom-bond motifs — that bigrams
  cannot. Coverage cannot arbitrate between them (both are ~0.999999), so
  the decision rests on representational resolution per dimension, and
  trigrams are where it is.

### 4,096 is the elbow, and it was measured, not assumed

`(1,3)`, DF ranking, scored against the 144,816-molecule holdout:

| dimension | effective `min_df` | occurrence coverage | mean OOV | OOV p95 | OOV p99 | mean nonzero | % of full nonzero | sparsity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1,024 | 12,886 | 0.983515 | 0.015843 | 0.06000 | 0.10714 | 78.21 | 96.977% | 0.9236 |
| 2,048 | 1,684 | 0.996750 | 0.002999 | 0.01754 | 0.04348 | 80.17 | 99.408% | 0.9609 |
| **4,096** | **126** | **0.999591** | **0.000392** | **0.00000** | **0.01010** | **80.59** | **99.928%** | **0.9803** |
| 8,192 | 7 | 0.999955 | 0.000058 | 0.00000 | 0.00000 | 80.65 | 99.993% | 0.9902 |
| 15,109 | 1 | 0.999995 | 0.000012 | 0.00000 | 0.00000 | 80.65 | 100.000% | 0.9947 |

Three things converge on 4,096:

1. **Diminishing returns are sharp there.** Going 2,048 → 4,096 buys
   0.520 percentage points of the full vocabulary's nonzero-feature
   content. Going 4,096 → 8,192 buys **0.065** — eight times less for the
   same doubling. Beyond 4,096 the extra dimensions are being spent on
   motifs that almost no molecule contains.
2. **4,096 is where p95 OOV reaches exactly zero.** At 4,096, 95% of
   unseen molecules lose *no* n-gram mass whatsoever, and 99% lose at most
   1.01%. At 2,048 the 95th percentile is still 1.75% and the 99th is
   4.35%.
3. **All-zero risk is zero** at 4,096 and at every wider cut. No molecule
   in the holdout is annihilated.

4,096 retains 99.928% of the representational content of the uncapped
vocabulary at 27.1% of its dimension. Per section 25 of the brief, the
remaining 0.072% does not justify quadrupling the dimension — and this
happens to land on the value the brief warned against assuming, which is
worth stating plainly: it was not assumed, the sweep was run over
1,024/2,048/4,096/8,192/16,384 and 4,096 is where the curve turns.

Expected runtime density at 4,096: mean 80.59 nonzero features, median 79,
p95 118, p99 148, max 309 — a **98.03% sparse** dense vector. That is
dense enough to be meaningful and a reason not to go to 16,384, where the
same ~81 features would sit in a 99.5% empty vector.

### DF ranking, not TF

Holdout coverage cannot separate them — the difference is in the fifth
decimal place, and the two rankings agree on 98.3–98.9% of terms:

| policy | dimension | DF occurrence coverage | TF occurrence coverage | top-K overlap |
| --- | --- | --- | --- | --- |
| `(1,3)` | 2,048 | 0.996750 | 0.996780 | 98.93% |
| `(1,3)` | 4,096 | 0.999591 | 0.999601 | 98.54% |
| `(1,3)` | 8,192 | 0.999955 | 0.999955 | 98.28% |

So the choice is made on the long-molecule evidence instead, which is
decisive. Molecules of more than 256 tokens are **0.452%** of the fit
subset but carry **3.435%** of all n-gram occurrences — a 7.6-fold
over-representation in TF space, because TF counts every repetition while
DF caps each molecule's contribution at one. Re-ranking with those
molecules removed churns the TF ranking more than the DF ranking at every
dimension:

| policy | dimension | TF terms changed | DF terms changed |
| --- | --- | --- | --- |
| `(1,3)` | 2,048 | 8 (0.39%) | 2 (0.10%) |
| `(1,3)` | 4,096 | 52 (1.27%) | 39 (0.95%) |
| `(1,3)` | 8,192 | 325 (3.97%) | 263 (3.21%) |

DF ranking is the more stable of the two under exactly the perturbation
the corpus makes likely, at no measured cost in coverage. No molecule is
excluded or length-filtered to achieve this — the bias is quantified, not
engineered away.

The tie-break is the lexicographic token tuple, which makes the order
**total**: no two distinct n-grams can tie, so no vocabulary slot is ever
awarded by insertion order, hash seed, sort stability or locale. This is
why MolFusion should freeze its own vocabulary rather than call
`max_features`: the rule above can be re-applied years from now from the
recorded counts alone, whereas a library's internal selection is
reproducible only as long as its internals are, and its tie behaviour is
not part of its public contract.

### Unigrams should not be protected

A global 4,096-term cut does exclude 204 of 289 unigrams, so the concern
in section 12 is real and worth checking. It does not survive measurement.
Forcing every unigram in and giving higher orders the remainder is
**worse at every dimension tested**, never better:

| policy | dimension | global-cap coverage | unigram-protected coverage | global mean OOV | protected mean OOV |
| --- | --- | --- | --- | --- | --- |
| `(1,2)` | 1,024 | 0.999957 | 0.999918 | 0.000058 | 0.000089 |
| `(1,3)` | 1,024 | 0.983515 | 0.972796 | 0.015843 | 0.026441 |
| `(1,3)` | 2,048 | 0.996750 | 0.995577 | 0.002999 | 0.004109 |
| `(1,3)` | 4,096 | 0.999591 | 0.999533 | 0.000392 | 0.000442 |
| `(1,3)` | 8,192 | 0.999955 | 0.999954 | 0.000058 | 0.000058 |

All-zero risk is zero under both, so protection buys nothing and costs
coverage. Inspecting the boundary explains why: at an effective `min_df`
of 126 the excluded unigrams are radio-isotope labels (`[11CH3]`,
`[125I]`, `[3H]`, `[123I]`, `[14C]`, `[211AtH]`, `[42K+]`), exotic charge
states (`[te+]`, `[se+]`, `[SH+]`, `[c+]`), rare counter-ions (`[Ca+2]`,
`[Zn+2]`, `[Mg+2]`, `[NaH]`) and high ring-closure indices (`%12`–`%24`,
which occur only in molecules holding more than eleven rings open at
once). Every core structural token — `C c O N ( ) = 1 2 [nH] Cl Br` — sits
orders of magnitude above the cut. The tokens protection would rescue are
rarer than the trigrams it would evict, which is precisely why it loses.

## What Phase 5F-D should do

1. Freeze the vocabulary as the **top 4,096 `(1,3)` token n-grams of the
   full 2,897,639-molecule corpus** by `(-document_frequency, ngram_tuple)`,
   with `min_df = 5` applied first.
2. Fit IDF on the **full corpus**, not the study's 95% subset. The split in
   this study is an analysis device only.
3. Record in the artifact metadata: `fit_corpus_sha256`, the contract IDs,
   the ranking rule, `min_df`, the dimension, and the resulting vocabulary
   digest.
4. Verify a deterministic rebuild reproduces the vocabulary and IDF
   byte-for-byte.

Two things to carry forward:

- **The study vocabulary and the production vocabulary will differ
  slightly.** This study ranked on the 95% fit subset (15,109 terms); the
  production fit uses 100% (15,212 terms), and the corpus-scope `min_df`
  table in `df_thresholds.csv` is the one Phase 5F-D should size against.
  At `min_df = 5` the corpus-scope `(1,3)` vocabulary is 9,383 terms,
  which the 4,096 cap then binds.
- **This study cannot measure predictive utility.** It measures coverage,
  OOV, density and stability on ChEMBL alone — deliberately, since no
  downstream label was read. The claim "trigrams are worth 4,096
  dimensions" rests on representational resolution, not on any measured
  gain in an ADMET endpoint. If a later phase evaluates `(1,2)` at 2,791
  dimensions against `(1,3)` at 4,096 on real tasks and finds no
  difference, `(1,2)` would be the cheaper choice and this record should be
  revised rather than defended.

## Reproducibility of the study run itself

| | |
| --- | --- |
| Python | 3.11.15 |
| RDKit | 2026.03.5 |
| MolFusion commit | `29b8f4564194ce1804484141652bacac5789a21e` |
| Peak memory | 188.7 MB |
| Counting pass | 1,680 s |
| Holdout pass | 325 s |
| Total | 2,010 s |

Exact counts throughout — no sketching, no sampling, no approximation. The
memory figure is why that was affordable: a token alphabet of 291 symbols
yields a three-order vocabulary of 15,212 entries, so exact dictionaries
cost less than 200 MB on a corpus of 153 million tokens.
