# MolFusion benchmark data: TDC ADMET acquisition and freeze (Phase 6A.1)

**Status:** frozen. No predictive benchmark has been run. This document
records what was acquired, how it was frozen, what TDC's official split
actually does, and where that contradicted an assumption in
[`benchmark-protocol.md`](benchmark-protocol.md).

Phase 6A designed the experiment against a dataset suite that was not yet on
disk. Phase 6A.1 put it on disk and checked the design against it. One
assumption did not survive, and §5 below is the amendment.

---

## 1. Acquisition

| | |
| --- | --- |
| Source | Therapeutics Data Commons, ADMET benchmark group |
| Package | `PyTDC` **1.0.0** |
| Endpoints | 22 (13 classification, 9 regression) |
| Download | one 1.47 MB archive, once |
| Local staging | `backend/benchmark_data/` — **git-ignored** |
| Tracked | the manifest, the code, this document |

### PyTDC is not a MolFusion dependency

`PyTDC` is installed in a **throwaway virtual environment outside the
project**, and appears in neither `pyproject.toml` nor `uv.lock`. MolFusion's
dependency surface is unchanged by this phase.

The reason is the transitive closure. Current PyTDC (1.1.15) resolves to
**123 packages including `torch`, `transformers`, `scanpy`,
`cellxgene-census` and `tiledbsoma`** — none of which have anything to do
with serving FeatureAgents, and one of which the protocol explicitly rules
out. PyTDC 1.0.0 is the newest release that avoids `torch` and
`transformers`, and even it pulls 117 packages if installed normally.

So it was installed with `--no-deps` plus only what the ADMET group actually
touches: `pandas`, `numpy`, `requests`, `tqdm`, `scikit-learn`,
`fuzzywuzzy`, `packaging`, `setuptools<81`, `biopython`, `mygene`,
`huggingface_hub`, and `rdkit==2026.3.5`.

Two pins worth recording: `setuptools<81`, because 81 removed
`pkg_resources` which PyTDC 1.0.0 still imports; and `rdkit==2026.3.5`,
**deliberately matched to MolFusion's pinned version** so that any
difference observed between TDC's scaffolds and MolFusion's is a difference
of convention rather than of RDKit build.

**PyTDC's role ends at acquisition.** Every audit downstream reads the
frozen files. Nothing in `molfusion_backend` imports `tdc` — if verification
required the tool that produced the data, the checksums would prove very
little.

---

## 2. The frozen serialization contract

A checksum is meaningless unless the bytes it covers are defined. The
downloaded CSVs are therefore not hashed as they arrive; every endpoint is
re-serialized through one explicit contract (`molfusion_frozen_csv_v1`,
`benchmark/release.py`) and *that* is hashed.

| Clause | Value | Why |
| --- | --- | --- |
| Encoding | UTF-8, no BOM | a BOM changes the checksum on some editors |
| Line terminator | LF | the `csv` module emits CRLF on Windows by default |
| Column order | `Drug_ID,Drug,Y` | fixed here, never taken from frame order |
| Row order | source order preserved | row order is part of dataset identity |
| Quoting | `QUOTE_MINIMAL` | one rule, applied by the stdlib |
| Float format | `repr()` | shortest form that reads back identically |
| Final newline | present | a well-formed text file |

`repr()` rather than a fixed precision because it is the shortest string
that round-trips to the identical float — `%.6f` would silently truncate
labels, and a dataframe library's default formatting is not a contract.

Only `csv` and `json` from the standard library are used. pandas is not a
MolFusion dependency and its serialization defaults are not something to
build a checksum on.

---

## 3. What TDC's official split actually does

This is the central output of the phase, established from
`tdc/benchmark_group/base_group.py` and then confirmed against all 22
endpoints.

```
benchmark
   ├── train_val.csv   ── shipped file
   │      └── get_train_valid_split(seed) → train 0.875 / valid 0.125
   └── test.csv        ── shipped file, never re-drawn
```

From the source:

- `get()` and `__next__()` read `train_val.csv` and `test.csv` from disk.
  **Neither accepts a seed.**
- `get_train_valid_split(seed, benchmark, split_type="default")` reads
  `train_val.csv` **only**, and splits it with `frac = [0.875, 0.125, 0.0]`.
  The trailing `0.0` is the test fraction: no test set is drawn, because one
  was already held out.
- `split_type="default"` resolves per dataset via
  `metadata.bm_split_names`; **all 22 ADMET endpoints resolve to
  `"scaffold"`.**
- TDC's scaffold key is `MurckoScaffoldSmiles(..., includeChirality=False)`,
  and groups are shuffled with `random.Random(seed)` — the sort-by-size line
  in that function is commented out upstream.
- `evaluate_many()` enforces `min_requirement = 5` runs.

### Seed semantics, proven rather than asserted

For every endpoint the canonical molecule set of each partition was hashed
at every seed. Across all 22 endpoints:

- **the test-set SHA-256 is identical at all five seeds** — the test set is
  a file, and a seed cannot move it;
- the train and validation set hashes differ at every seed.

So the seed moves the train/validation boundary **and nothing else**.

### Seed values

PyTDC fixes the run *count* (≥ 5) but **not the seed values** — the caller
passes them. TDC's own documentation demonstrates `1–5`, so Track A1 uses
`1–5`. Phase 6A's `0–4` is kept for Track A2, deliberately different, so a
seed value alone can never make a result row ambiguous.

---

## 4. Official partition audit

Measured with MolFusion's pinned RDKit, per endpoint, over the official
partitions:

- **canonical molecule overlap between `train_val` and `test`: 0 in all 22
  endpoints.** Checked after canonicalization, so two spellings of one
  molecule could not have hidden.
- **scaffold overlap: 0 in all 22 endpoints, under both conventions** —
  TDC's chirality-excluding key and MolFusion's chirality-including key.

TDC's official ADMET splits are clean. That is a positive result and it is
worth stating plainly, because the audit was built to find the opposite.

The two scaffold conventions do differ in general — a fused bicyclic whose
ring-fusion stereocentres survive scaffold reduction gets different keys —
which is why both are reported rather than one being assumed to answer for
the other.

---

## 5. Amendment to the Phase 6A split protocol

> **Phase 6A said:** five independent 70/10/20 Bemis–Murcko scaffold splits,
> seeds 0–4, each drawing its own test partition.
>
> **What is true:** TDC ships a fixed held-out test set and re-splits only
> the remainder. The seed never touches the test set.

The **fractions in Phase 6A were right**, which is worth noting because it
would be easy to over-correct. Nesting TDC's numbers reproduces them
exactly: `0.8 × 0.875 = 0.70` train and `0.8 × 0.125 = 0.10` validation
against a `0.20` test set.

What was wrong was the assumption that the test partition is re-drawn per
seed. That is not cosmetic — it changes what the spread across runs
*measures*. Under TDC's protocol the spread is sensitivity to the
train/validation boundary on a fixed evaluation set. Under Phase 6A's
original reading it would also have included which molecules get evaluated
at all.

Both are legitimate questions, so both are kept — as two tracks that are
never mixed and never share a label.

### Track A1 — official, TDC-comparable *(primary)*

| | |
| --- | --- |
| Test set | the shipped `test.csv`, unchanged, identical at every seed |
| Train/validation | `get_train_valid_split`, 0.875 / 0.125, TDC's own scaffold splitter |
| Seeds | 1, 2, 3, 4, 5 |
| Cleaning | **none** — official rows consumed exactly as shipped |
| Split ID | `tdc_official/seed=N` |
| Purpose | comparability with published TDC leaderboard numbers |

### Track A2 — MolFusion robustness *(supplementary)*

| | |
| --- | --- |
| Splits | five independent 70/10/20 Bemis–Murcko splits |
| Seeds | 0, 1, 2, 3, 4 |
| Cleaning | full Phase 6A policy |
| Split ID | `molfusion_scaffold/seed=N` |
| Purpose | does the A1 ranking survive a different scaffold partition? |

A2 answers something A1 cannot: whether a representation's ranking is a
property of the representation or an artefact of the one partition TDC
happened to publish. It is **not** comparable with TDC leaderboard numbers
and is never to be labelled as official TDC results.

**A2's placement: supplementary.** A1 is the headline because a reader can
check it against published work. A2 is evidence about the stability of the
A1 ranking, not a second opinion competing with it. Promoting it to the main
manuscript would invite exactly the confusion this separation prevents.

Both tracks are enforced in code: every result row carries
`protocol.split_id(track, seed)`, which names its track, and the two tracks
use disjoint seed values as a second line of defence.

---

## 6. Why Track A1 applies no cleaning

MolFusion's duplicate policy — collapse agreeing duplicates, drop
conflicting groups entirely — is right for MolFusion's own analysis and
wrong for a leaderboard comparison. That was decided **after** measuring it
against the official partitions, not before.

Applying it to the shipped rows removes:

| Impact | Endpoints |
| --- | --- |
| 0% of both partitions | 8 endpoints |
| under 6% | 12 endpoints |
| **30%** (`train_val` 30.7%, `test` 30.0%) | `clearance_hepatocyte_az` |
| **53–58%** (`train_val` 53.3%, `test` 58.0%) | `ppbr_az` |

`ppbr_az` and `clearance_hepatocyte_az` carry many replicate measurements of
the same compound whose values differ by more than 1% of the label spread.
A test set missing 58% of its molecules is **not the test set every
published number was computed on**. Scoring on it and presenting the result
as TDC-comparable would be false.

So Track A1 consumes the official rows as shipped and reports the duplicate
structure as a caveat; Track A2 applies the full policy. Any endpoint where
cleaning would move more than **5%** of either official partition is flagged
wherever A1 and A2 results appear together.

---

## 7. Metrics: two concepts, both recorded

TDC defines a primary metric per endpoint, and it is not uniform: `roc-auc`,
`pr-auc`, `mae`, and `spearman` all appear. MolFusion's protocol ranks on a
uniform set so that endpoints can be compared with each other.

Overwriting one with the other would lose something either way, so every
endpoint records **both**:

- `tdc_official_metric` — source-defined, for leaderboard comparability
- `molfusion_primary_metric` — AUROC (classification) or MAE (regression),
  for cross-endpoint ranking

with the full MolFusion secondary set (AUPRC, balanced accuracy, MCC / RMSE,
R², Spearman) recorded alongside.

Task type is **derived from the official metric** rather than hard-coded:
`roc-auc` and `pr-auc` are classification-only, `mae` and `spearman`
regression-only. The package already states it, so nothing is typed from
memory.

---

## 8. Endpoint inclusion

All 22 endpoints **pass** the frozen Phase 6A inclusion criteria (≥ 100
usable molecules; ≥ 20 minority-class molecules for classification). None
was excluded, and none was excluded for being difficult.

Full per-endpoint counts — raw rows, invalid SMILES, duplicates collapsed,
conflicting records dropped, usable molecules, class balance or label
spread, scaffold profile, checksums, and split identity hashes — live in the
tracked manifest, `backend/benchmark_manifests/tdc_admet_group.json`.

---

## 9. ChEMBL exposure

`smiles_tfidf_4096` carries a vocabulary and IDF fitted on the frozen ChEMBL
37 corpus, so some benchmark molecules contributed to it. This is
**unsupervised exposure, not label leakage** — no benchmark label was read,
and the overlap computation reads none either — but it is an asymmetry the
other six representations do not have.

Measured over all 22 endpoints, overlap with ChEMBL 37 ranges from
**23.4%** (`vdss_lombardo`) to **99.6%** (`clearance_hepatocyte_az`), with
most endpoints between 45% and 90%. Four endpoints exceed 99%
(`clearance_hepatocyte_az`, `clearance_microsome_az`, `ppbr_az`,
`lipophilicity_astrazeneca`), which is unsurprising — they are
ChEMBL-derived assays — and is exactly the kind of thing that has to be
stated beside a TF-IDF result rather than discovered by a reader.

Per-endpoint counts are in the manifest. **No overlapping molecule
is removed.** Changing the benchmark to suit one representation would be a
worse distortion than the exposure it corrects; the honest treatment is to
measure it and report it beside the result.

---

## 10. Release identity

The frozen release is identified by content, never by when it was made:

```
identity = SHA-256 over
    release name │ protocol version │ serialization id │
    per endpoint: train_val SHA-256, test SHA-256,
                  test-set identity, per-seed set identities
```

Timestamps, absolute paths, directory metadata and acquisition order are
deliberately excluded. Re-freezing the same data tomorrow must produce the
same identity, or the identity records when the work happened rather than
what the data is.

---

## 11. Reproducing this

```bash
# 1. isolated acquisition environment (never the MolFusion venv)
uv venv --python 3.11 <somewhere>/tdcenv
uv pip install --python <somewhere>/tdcenv/Scripts/python.exe --no-deps "PyTDC==1.0.0"
uv pip install --python <somewhere>/tdcenv/Scripts/python.exe \
    pandas numpy requests tqdm scikit-learn fuzzywuzzy packaging \
    "setuptools<81" biopython mygene huggingface_hub "rdkit==2026.3.5"

# 2. download + freeze into backend/benchmark_data/ (git-ignored)
# 3. audit and manifest, in the MolFusion venv:
.\backend\.venv\Scripts\python.exe -m molfusion_backend.benchmark.manifest_cli
```

Verification needs only the MolFusion venv and the frozen directory. If the
checksums match the manifest, the data is the data this protocol was frozen
against.

---

## 11a. Representation availability

All seven Track A representations were computed over every endpoint's
cleaned molecule universe, using the production FeatureAgents unchanged. No
agent was modified because a benchmark molecule failed.

**There were zero representation failures.** For all 22 endpoints the common
evaluation set equals the full cleaned molecule count, so the intersection
rule costs nothing here and the 1% loss alert is not triggered anywhere.
79,712 molecules × 7 representations, no exclusions.

This is worth stating because the intersection rule exists to stop one
representation looking better by having failed on the hard compounds. On
this suite it never gets the chance.

## 11b. Estimated execution cost

Measured from real fits on two size anchors (n≈1,375 and n≈9,191) and two
dimension anchors (1,024 and 4,096), then extrapolated.

Track A1 work:

| | |
| --- | --- |
| 22 endpoints × 7 reps × 2 probes × 5 seeds × 4 candidates | **6,160** hyperparameter fits |
| final fits | **1,540** |
| total model fits | **7,700** |
| test evaluations | **1,540** |
| feature computations, cached | **154** (vs 1,540 uncached) |

Estimated single-core time:

| Component | Track A1 |
| --- | --- |
| linear probe fits | 0.8 h |
| **nonlinear probe fits** | **26.5 h** |
| feature computation (once, cached) | 1.6 h |
| **Track A1 total** | **~29 h** |
| Track A2 (features reused) | +27 h |
| **both tracks** | **~56 h** |

The nonlinear probe is 97% of the cost, and its cost tracks representation
*dimension* almost independently of dataset size: HistGradientBoosting fit
time barely moved between n=1,375 and n=9,191 at fixed width, but nearly
quadrupled from 1,024 to 4,096 columns. `smiles_tfidf_4096` is therefore the
single most expensive cell in the matrix.

Two consequences for Phase 6A.2: the feature cache is worth having (10× on
feature work), and the run should be parallelised across endpoints, which is
trivially safe because endpoints share no state.


## 12. What Phase 6A.1 did not do

No model was trained. No representation was added or modified. No production
FeatureAgent was touched. No artifact was rebuilt. The full
22 × 7 × 2 × 5 matrix remains unrun, and nothing in this phase produced a
predictive score.
