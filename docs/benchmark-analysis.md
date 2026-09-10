# Track A1 analysis (Phase 6A.3)

**Status:** analysis complete on the frozen Track A1 matrix
(`d40ef09b…de868`, 308 cells, 6,160 rows). Track A2 has **not** been run.
Raw results were treated as read-only and verified unchanged afterwards.

This document records the analysis contract and the amendment review. The
numbers themselves live in the derived tables under
`backend/benchmark_runs/track_a1/analysis/` (git-ignored, reproducible from
the raw matrix plus committed source).

---

## 1. Three decisions that shape everything

**The statistical unit is the endpoint.** Track A1 has five
train/validation realizations scored on *one fixed test set*. Those five
values are not five independent observations; treating them as such would be
pseudoreplication and would inflate every test by a factor of five. They are
aggregated to one value per endpoint × representation × probe before any
cross-endpoint comparison.

**Linear and nonlinear probes are never pooled.** How much information a
representation makes *linearly accessible* and how much a flexible model can
extract from it are different questions. Averaging them answers neither —
and, as it turns out, they give different answers here.

**Raw metric values never cross endpoint boundaries.** AUROC, AUPRC, MAE and
Spearman differ in units, direction and attainable range. Only ranks —
computed within an endpoint, direction-aware, average ties — are aggregated
across endpoints.

---

## 2. Amendment review

### Amendment A — unparseable SMILES

| | |
| --- | --- |
| Endpoint | `solubility_aqsoldb` only |
| Count | 2 |
| Rows | 9144, 9145 — **both in the TEST partition** |
| Handling, leakage audit | excluded from the canonical and scaffold sets |
| Handling, evaluation | counted as representation failures, not scored |
| Denominator | `n_test = 1995`, not the 1997 rows shipped |

The exclusion **is** reflected in reported denominators, and identically for
all 7 representations and both probes — verified: `n_test` takes exactly one
distinct value across every `solubility_aqsoldb` result row.

Confirmed: no computed score changed (the cells had previously failed
outright, producing nothing); the official test identity still matches the
frozen manifest value `71bcb5db…b9e8`, because Phase 6A.1's audit excluded
the same two rows when computing it; and molecule and scaffold overlap
remain 0 / 0 / 0.

### Amendment B — non-finite descriptor values

| | |
| --- | --- |
| Endpoint | `solubility_aqsoldb` only |
| Row | dataset row 3274 — **TRAIN_VAL partition** |
| Representation | `rdkit_physchem_descriptors` |
| Descriptors | `MaxPartialCharge`, `MaxAbsPartialCharge` |
| Sign | **+inf** (both) |
| Scope | 2 values across all 152 feature matrices |

**Sanitization rule:** `x → NaN where not isfinite(x)`, applied as the first
pipeline step for every representation and both probes. It is a pure
function of the value, uses **no statistic of any kind**, and fits nothing —
so it cannot transport information between partitions.

The resulting NaN is then handled by the already-frozen policy: the linear
probe imputes with `SimpleImputer(strategy="median")` **inside an
`sklearn.Pipeline`**, so the median is fitted on the training split only and
validation/test are transform-only; the nonlinear probe consumes NaN
natively with no imputation at all.

The affected row is in `train_val`, never in test, so under every seed it is
a training or validation row — the imputation statistic is train-derived and
applied to a non-test row.

**No score changed as a result of an after-the-fact patch.** The fold is the
identity on finite input, and `solubility_aqsoldb`'s descriptor matrix is
the only one in the benchmark containing inf, so the 306 cells computed
before the amendment are numerically unaffected. A test asserts the identity
property directly.

**Assessment: both amendments are scientifically acceptable.** Each is the
minimal available response, neither drops a molecule from an official
partition, neither uses a validation- or test-derived statistic, and both
are recorded in `protocol.py` and here rather than absorbed silently.

---

## 3. Aggregation and ranking

- **Seed aggregation:** mean of the five realizations, with SD, median, min
  and max reported alongside rather than folded in. The mean rather than the
  median because five is a small sample and the median of five discards most
  of it.
- **Primary metric:** read from each row's own `molfusion_primary_metric`
  column, which the runner wrote from the frozen protocol *before any score
  existed*. Never selected after seeing results.
- **Ranking:** direction-aware within endpoint, 1 = best, **average ranks
  for exact ties**. Ties are never broken by representation name or any
  other property of the contestant — that would manufacture an ordering the
  data does not contain.

---

## 4. Statistics

| Stage | Method |
| --- | --- |
| Omnibus | Friedman across all 7, on endpoint-level values, per probe and per task |
| Post-hoc | paired Wilcoxon signed-rank, **only where the omnibus rejects** |
| Correction | Holm over the 21 pairs |
| Effect size | matched-pairs rank-biserial, reported with **every** p-value |
| Uncertainty | bootstrap over **endpoints**, 10,000 resamples, seed 0 |
| α | 0.05 |

Bootstrap resamples endpoints, not molecules: the claim being bounded is a
cross-endpoint one, and resampling molecules would answer a different
question while badly understating the uncertainty.

---

## 5. Nonlinear gain

Two transparent measures, because neither alone suffices:

- **`normalised_gain`** — within each endpoint, all 14 (representation,
  probe) oriented scores are min-max scaled to [0, 1] and the gain is
  nonlinear minus linear. Scaling *within* the endpoint is what makes AUROC
  and MAE endpoints comparable at all; scaling across all 14 rather than per
  probe keeps both probes on one common scale.
- **`rank_gain`** — linear rank minus nonlinear rank. Immune to metric
  scale, but measures only position relative to competitors.

---

## 6. A caveat that matters for interpretation

The hyperparameter audit surfaced something that qualifies the linear
results. Under the linear probe, most representations select the **most
regularized end of the frozen grid** most of the time — `alpha = 100`
(the grid maximum) on 58–98% of Ridge fits for six of seven
representations, and `C = 0.01` (the grid minimum) on 48–75% of logistic
fits for four of seven.

`smiles_tfidf_4096` is the exception: its selections sit in the grid
interior (`alpha = 1.0` most often; `C` spread toward the permissive end).

So the linear comparison may under-serve the representations that are
regularization-saturated, and TF-IDF's linear advantage should be read with
that in mind. **The grid was not expanded** — it is frozen, and widening it
after seeing scores would be exactly the kind of post-hoc tuning the
protocol exists to prevent. It is recorded as a diagnostic and as a
hypothesis for a later phase.

A second note on the same audit: the nonlinear grid has only **two values
per parameter**, so every selection is simultaneously at the minimum and the
maximum and the "boundary" flag is `True` by construction. That column is
uninformative for the nonlinear probe and must not be read as saturation.
The actual split is balanced (56% / 44% on both parameters), which is the
meaningful statement.

---

## 7. Outputs

Derived tables in `backend/benchmark_runs/track_a1/analysis/`:
`endpoint_summary.csv`, `endpoint_ranks.csv`, `representation_ranks.csv`,
`representation_characteristics.csv`, `pairwise_wins.csv`,
`nonlinear_gain.csv`, `nonlinear_gain_summary.csv`,
`hyperparameter_audit.csv`, `cost_summary.csv`,
`endpoint_instability.csv`, `tfidf_exposure_analysis.csv`, `friedman.csv`,
`statistical_tests.csv`, `bootstrap_mean_rank.csv`, plus six figure-ready
tables under `figures/` and `analysis_report.json`.

Figures are emitted as **figure-ready data** rather than rendered images:
matplotlib is not a project dependency, and a CSV that any plotting tool can
consume is more reproducible than a PNG anyway.

**Analysis identity** covers the raw scientific identity, analysis version,
protocol version and the full statistical configuration (α, correction,
effect size, bootstrap seed and resamples, resampling unit, flagged endpoint
list). It excludes timestamps, paths and machine names.

---

## 8. What Phase 6A.3 did not do

No model was retrained. No representation was added or changed. The raw
result matrix was not modified — its identity was re-verified before
analysis and its size and mtime re-checked afterwards. Track A2 was not
started.
