# Track A1 execution (Phase 6A.2)

**Status:** execution infrastructure frozen; Track A1 run against release
`TDC-ADMET-2026-09` (`10bda5f0…35e3`). Track A2 has **not** been run.

Phase 6A designed the experiment; Phase 6A.1 froze the data and established
what TDC's split actually does. This phase executes Track A1 — the official,
leaderboard-comparable half — and nothing else.

---

## 1. What Track A1 is, precisely

Five **train/validation realizations evaluated on one fixed external test
set**. Not five independent test splits.

```
official test.csv ──────────────────────────────► scored 5 times
                                                   (one per realization)
official train_val.csv
   └── seed 1..5 ──► train 0.875 / validation 0.125
```

This distinction survives into the result rows: `split_id` is
`tdc_official/seed=N`, `split_strategy` is `tdc_official_fixed_test`, and
the seed values are `1–5` — disjoint from Track A2's `0–4`, so a bare seed
can never make a row ambiguous.

The consequence for later statistics: spread across the five runs is
**variability due to the train/validation and model-selection realization on
a common external test set**. It is not variability across independent test
sets, and must never be described as such.

**No cleaning is applied.** Official rows are consumed exactly as shipped,
duplicates included, per the Phase 6A.1 decision. Applying MolFusion's
conflicting-label rule here would remove 58% of `ppbr_az`'s test set, and a
score on that is not comparable with any published number.

---

## 2. Execution needs no PyTDC

`molfusion_backend.benchmark.a1`, `.runner` and `.feature_store` import no
`tdc` module, and a test asserts it. Verification that required the tool
which produced the data would prove very little.

Everything is read from the frozen directory, and every frozen file's
SHA-256 is re-checked against the tracked manifest before its rows are used.

---

## 3. Feature cache

Phase 6A froze a *per-molecule* key. Execution needs a coarser one: a matrix
per `endpoint × representation`, reused across five seeds, two probes and
four hyperparameter candidates — forty fits that would otherwise recompute
identical features forty times.

### Key

```
sha256(
  cache_schema_version │ release_identity │ endpoint │
  agent_id │ agent_version │ output_dim │ normalization_id │
  row_identity_sha256 │ artifact_identity
)
```

joined by `0x1f`. Each field earns its place: the release and endpoint say
which rows, the agent id/version/dimension say which code, the
normalization id says which canonicalization produced the inputs, the row
identity says which molecules in which order, and the artifact identity
covers the fitted payload for `smiles_tfidf_4096` (its type, id, version,
and the artifact's own checksum-verified corpus identity).

**Not** keyed on: filename, directory, timestamp, split, seed, probe,
hyperparameters, or labels. A path is not an identity, and features do not
depend on the model that will consume them.

### Row identity

An **order-sensitive** SHA-256 over `(index, canonical_smiles)` pairs — not
the set hash used elsewhere. The claim being protected is "matrix row *i* is
molecule *i*", and a set hash would happily accept a permuted matrix.
Duplicates are preserved rather than collapsed, because A1 consumes official
rows as shipped.

### Validation on load

`cache_schema_version`, `release_identity`, `endpoint`, `agent_id`,
`agent_version`, `output_dim`, `row_identity_sha256`, `artifact_identity`,
matrix dtype, matrix shape, and row-index count against matrix rows.

A **missing** entry is a cold cache and returns `None`. A **present** entry
that disagrees raises `FeatureCacheError`. That asymmetry is deliberate: a
stale entry is evidence of a problem, and silently recomputing over it would
hide the problem instead of surfacing it.

### Atomicity

Writes stage into a sibling temporary directory and finalize with
`os.replace`. A killed process leaves either a complete entry or nothing —
never a half-written matrix that validates.

---

## 4. Checkpointing and resume

One shard per `endpoint × representation × probe`, written atomically via
`mkstemp` + `os.replace`. A shard is reused **only** if it carries the
current schema version, the current release identity, the current protocol
version, `status == "complete"`, exactly the expected seeds, exactly the
expected row count, and a `cell_identity` that recomputes to the same value.

Every rejection path means a recomputed cell. That is the cheap direction to
be wrong in: recomputing a good cell costs minutes, while trusting a
truncated or stale one silently corrupts the result table.

The work *unit* is `endpoint × representation` (both probes together) so the
feature matrix is computed once rather than twice. One damaged shard
therefore rebuilds its sibling as well — slightly wasteful, and much simpler
to reason about than partial-unit bookkeeping.

**Verified by interrupting a real run.** A truncated shard, a shard left at
`status: running`, and a shard relabelled with a foreign release identity
were all rejected and rebuilt; the five intact cells were reused; feature
matrices came from cache (0.1 s versus 6.5 s cold); zero duplicate rows.

---

## 5. Result identity

Two digests, because they answer different questions:

| Digest | Covers | Stable across reruns? |
| --- | --- | --- |
| `results_file_sha256` | the whole CSV, timings included | no |
| `scientific_identity_sha256` | the science only | **yes** |

The scientific identity excludes every duration and the cache-hit flag —
properties of the machine and the run order, not of the result. Metric
values are hashed via `repr()`, the shortest round-trip form, so the digest
is exact rather than rounded. Rows are canonically sorted first, so shard
completion order cannot change it.

`cell_identity` follows the same rule at cell granularity: release,
protocol version, track, endpoint, representation, probe and seed set —
never a timestamp, path, or worker count.

---

## 6. Parallelism and worker count

Parallelism is across cells, never inside the scientific logic, so the
worker count cannot change a single number.

The count was measured, not assumed. The worst-case worker — the
4096-dimensional TF-IDF matrix on the largest endpoint (`cyp2d6_veith`,
13,130 rows, a 430 MB matrix) — peaked at **894 MB** resident. The host has
**2 physical cores** and roughly 2.5 GB free, so **2 workers** is both the
CPU limit and comfortably within memory.

This corrects the Phase 6A.1 suggestion of 8-way parallelism, which assumed
a machine this one is not.

---

## 7. Leakage discipline

Re-verified per endpoint before any model is fitted, rather than trusted
from the earlier phase:

- each frozen file's SHA-256 against the manifest;
- the official test molecule set identity against the frozen manifest value;
- zero canonical molecule overlap between `train_val` and `test`;
- zero scaffold overlap, under TDC's chirality-excluding key **and**
  MolFusion's chirality-including key.

Any violation aborts the endpoint rather than proceeding with a caveat.

Within a cell the tune-then-test ordering is straight-line code, not a
convention a caller must honour: every candidate is scored on validation,
the winner is refit, and only then is the test partition touched — once.
Fitted preprocessing lives inside an `sklearn.Pipeline`, so `fit` sees
training rows only and validation and test are transformed with the
training-fitted parameters.

---

## 8. Representation failures

Phase 6A.1 found zero Track A representation failures. Execution does not
assume that holds: each unit compares its observed failure count against the
frozen expectation, and a mismatch **fails the cell** rather than proceeding.

Comparing models trained on silently different molecule sets is the specific
failure this prevents.

---

## 8a. Amendment: non-finite descriptor values

Phase 6A defined NaN handling for RDKit descriptors. It did not anticipate
**±inf**, which RDKit also emits: `MaxPartialCharge` and
`MaxAbsPartialCharge` diverge for certain structures. Exactly one molecule
in the entire 22-endpoint suite triggers it — a `solubility_aqsoldb`
training row — producing **two infinite values across 152 feature
matrices**.

scikit-learn tolerates NaN in both probes and rejects inf in both, so those
two values failed all seven of that endpoint's cells.

**Amendment:** ±inf is folded onto NaN, then handled by the already-frozen
NaN policy. An infinite descriptor carries the same information as a missing
one — the quantity is not meaningfully computable for that molecule — so
this reuses existing machinery rather than inventing a second mechanism.

The alternatives were worse. Dropping the molecule is unavailable: Track A1
may not alter the official partitions. Clipping to a finite bound would
assert a value the descriptor never produced.

The fold is **stateless** (fits nothing, so it cannot leak), applied
**uniformly** to every representation and both probes (so no representation
gets special treatment), and is the **identity on finite input** — which is
why it invalidates none of the 294 cells computed before it existed. A test
asserts each of those three properties.


## 9. TF-IDF cost is a measurement, not a problem

The nonlinear probe on `smiles_tfidf_4096` is the most expensive cell in the
matrix by a wide margin. It was **kept**. Substituting a cheaper model for
one representation would break the comparison fairness the benchmark exists
to provide, and the cost itself is an input to the planned
performance-versus-cost analysis.

Recorded per cell: feature dimension, nonzero fraction, feature seconds
(and whether they came from cache), hyperparameter-selection seconds, final
fit seconds, validation predict seconds and test predict seconds — kept
separate throughout, because a blended number would hide which dominates.

---

## 10. ChEMBL exposure carry-forward

Each shard carries its endpoint's frozen `chembl37_exposure` block, so
TF-IDF result tables join to the exposure audit without a second lookup. The
overlap is disclosure metadata: no molecule is removed, no label changed, and
the artifact is never refitted.

---

## 11. Outputs

```
backend/benchmark_runs/track_a1/          (git-ignored)
  shards/<endpoint>/<representation>__<probe>.json
  results_track_a1.csv                    long format
  run_report.json                          audit + identities + environment
  timings.json                             per-seed timing detail
backend/benchmark_cache/features/         (git-ignored)
```

Everything here is reproducible from the frozen release plus committed
source, so none of it is tracked. The runner, its tests, and this document
are.

---

## 12. What Phase 6A.2 did not do

Track A2 was not run. No significance test was performed, no pairwise
comparison was made, and no representation is described as better than
another — that analysis belongs to the next phase, on the completed and
audited matrix.
