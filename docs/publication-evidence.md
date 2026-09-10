# Phase 6B — Publication evidence package

The frozen benchmark, organised for a manuscript. This document is the map:
what may be claimed, what may not, where each number comes from, and what a
reproducibility section can honestly say.

Nothing here is a new result. Every value traces to a Phase 6A.3 or 6A.4
table. Phase 6B decides what is *publishable*, not what is true.

Generate with:

```powershell
.\backend\.venv\Scripts\python.exe -m molfusion_backend.benchmark.publication_cli
```

Outputs land in `backend/benchmark_runs/publication/` and are git-ignored
(`/benchmark_runs/`), like every other generated benchmark output. The code
that produces them is tracked; the bytes are reproducible from it.

---

## 1. The goal is the smallest defensible set of claims

The risk at this stage is not arithmetic. Every number is already computed
and checked. The risk is a true number restated one degree too strongly —
"best mean rank" becoming "best representation", "unsupervised corpus
overlap" becoming "leakage". That kind of drift is cheapest to prevent
before any prose exists, which is what the claim registry is for.

Each of the 11 claims in `evidence/claim_registry.csv` carries:

| Field | Purpose |
| --- | --- |
| `claim_type` | PRIMARY / ROBUSTNESS / SECONDARY / NEGATIVE / CAVEAT / EXPLORATORY |
| `supported_by` | the frozen tables behind it |
| `statistical_basis` | test, effect size, interval, unit of inference |
| `limitations` | what bounds it — never empty |
| `allowed_in_abstract` / `allowed_in_conclusion` | placement gate |
| `recommended_wording` | the sentence the evidence licenses |
| `prohibited_wording` | sentences it does **not** license |

`validate_registry()` refuses a registry with a claim lacking a limitation
or a prohibition, with an exploratory claim marked abstract-safe, or with
no primary claim. Tests enforce all of it.

### The claims

| ID | Type | Confidence | Abstract | Claim |
| --- | --- | --- | --- | --- |
| C1 | PRIMARY | high | yes | Physicochemical descriptors lead the nonlinear probe; CI upper bound below every competitor's mean rank |
| C2 | ROBUSTNESS | high | yes | That lead survived repartitioning and stricter cleaning |
| C3 | NEGATIVE | high | yes | No representation separated clearly under the linear probe |
| C4 | NEGATIVE | moderate | no | The TF-IDF regression advantage weakened under repartitioning |
| C5 | ROBUSTNESS | high | no | 9/11 Holm-significant contrasts reproduced; direction preserved 11/11 |
| C6 | SECONDARY | moderate | no | The best nonlinear representation was also among the cheapest |
| C7 | CAVEAT | high | no | Six endpoints cannot support per-endpoint claims |
| C8 | CAVEAT | high | no | Two endpoints are cleaning-sensitive; conclusions unchanged without them |
| C9 | EXPLORATORY | low | **no** | ChEMBL 37 corpus exposure — reported, never tested |
| C10 | SECONDARY | moderate | no | Representation performance is probe-dependent |
| C11 | ROBUSTNESS | high | no | Fragment counts bottom-two in all four probe × track combinations |

---

## 2. The primary claim, and the check behind it

> The 217-dimensional physicochemical descriptor representation achieved
> the best nonlinear-probe mean rank, and its bootstrap 95% confidence
> interval upper bound remained below the mean rank of every competing
> fixed-vector representation.

Phase 6B was told not to assert interval separation without checking it, so
`ci_separation()` computes the comparison and `evidence/ci_separation_check.csv`
records every pair. The answer:

| Track | Probe | Leader | CI | Clear of all competitors |
| --- | --- | --- | --- | --- |
| A1 | nonlinear | `rdkit_physchem_descriptors` | [1.45, 2.41] | **yes** |
| A2 | nonlinear | `rdkit_physchem_descriptors` | [1.32, 2.27] | **yes** |
| A1 | linear | `smiles_tfidf_4096` | [1.82, 3.36] | no (3 overlap) |
| A2 | linear | `morgan_ecfp4_1024` | [2.14, 3.50] | no (4 overlap) |

So the stronger statement is available, and it holds in both tracks. It is
still stated in the weaker form above, for a reason worth keeping in the
manuscript: these are **marginal per-representation intervals, not a
simultaneous band**, so non-overlap is supporting evidence for a
difference, not a test of one. The test is the Holm-corrected Wilcoxon,
which is reported alongside.

**Predictive accessibility, not information content.** 217 descriptors put
more signal within reach of this frozen probe than 4096 TF-IDF dimensions
did. That is not a claim about what the representations *contain*.

---

## 3. Terminology

The six competitors are not all fingerprints, so the collective term
`structural fingerprints` is prohibited and tested against. Use **"six
other fixed-vector representations"** collectively, and the accurate
category individually:

| Representation | Category | Dim |
| --- | --- | --- |
| `rdkit_physchem_descriptors` | physicochemical descriptors | 217 |
| `smiles_tfidf_4096` | SMILES n-gram TF-IDF | 4096 |
| `maccs_keys_167` | substructure key fingerprint | 167 |
| `avalon_1024` | substructure fingerprint | 1024 |
| `morgan_ecfp4_1024` | circular fingerprint | 1024 |
| `erg_reduced_graph_315` | reduced-graph features | 315 |
| `rdkit_fragment_descriptors` | fragment counts | 85 |

---

## 4. The linear result is a finding, not a failure

A1's linear leader was `smiles_tfidf_4096`; A2's was `morgan_ecfp4_1024`;
their intervals overlap substantially, as do those of two or three others.
The correct statement is:

> No single representation clearly separated from the field under the
> linear probe.

Prohibited: that the representations are *equivalent* under a linear probe,
or that either track's leader is "the best linear representation".

---

## 5. TF-IDF: mixed, and the negative half is not buried

Competitive under the linear probe, competitive under nonlinear modelling,
**weaker in regression robustness**, and substantially more expensive.

| | A1 | A2 |
| --- | --- | --- |
| regression mean rank (linear probe) | 1.33 | 2.33 |
| regression wins | 7/9 | 4/9 |
| regression top-3 | 9/9 | 7/9 |

The movement exceeds the pre-registered 0.5-rank tolerance, so H3's verdict
is `weakened`. This is C4, and it belongs in the Results and the
Discussion. Prohibited: "TF-IDF fails at regression", or that it was shown
equivalent to other representations — n = 9 and no regression-only contrast
survives Holm correction in A2.

---

## 6. ChEMBL exposure stays exploratory

Six of 22 endpoints have ≥ 90% molecule overlap with the ChEMBL 37 corpus
that fitted the TF-IDF vocabulary. This is **external unsupervised corpus
exposure**: no benchmark label was read during fitting. No molecule was
removed retrospectively.

The frozen analysis plan contains **no test of exposure as a factor**, and
none was run in Phase 6B. The observed difference in TF-IDF rank movement
between high- and low-exposure endpoints is confounded with task type (five
of the six high-exposure endpoints are regression) and rests on 6 versus
16 endpoints.

C9 is therefore `EXPLORATORY`, barred from both abstract and conclusion by
schema. Prohibited throughout: *leakage*, *label leakage*, *contamination*,
*exposure explains TF-IDF performance*, *corrected for corpus overlap*.

---

## 7. Endpoint stability

Kendall's W across the five A2 partitions, taken on each endpoint's
**weaker** probe so a strong probe cannot mask an unstable one. Below
W = 0.35, per-endpoint interpretation is **NOT RECOMMENDED**:

| Endpoint | W linear | W nonlinear | min |
| --- | --- | --- | --- |
| `herg` | 0.157 | 0.514 | 0.157 |
| `cyp2c9_substrate_carbonmangels` | 0.171 | 0.363 | 0.171 |
| `clearance_hepatocyte_az` | 0.209 | 0.286 | 0.209 |
| `cyp3a4_substrate_carbonmangels` | 0.420 | 0.217 | 0.217 |
| `cyp2d6_substrate_carbonmangels` | 0.312 | 0.389 | 0.312 |
| `bioavailability_ma` | 0.562 | 0.331 | 0.331 |

All six are pre-registered. **All six stay in the 22-endpoint benchmark** —
they are flagged, never dropped.

One honest disagreement: the same rule also flags `vdss_lombardo`
(nonlinear W = 0.343), which was *not* pre-registered. It is recorded as
`BORDERLINE` and surfaced in `publication_report.json` rather than silently
added — quietly widening a pre-registered exclusion after seeing the data
turns a caveat list into a post-hoc filter.

---

## 8. The 19-endpoint subset

19 of 22 endpoints were genuinely repartitioned (mean pairwise test Jaccard
≤ 0.50). `ames`, `ld50_zhu` and `solubility_aqsoldb` were not;
`solubility_aqsoldb` produced one identical test set at all five seeds.

`table7_22_vs_19_endpoint_subset.csv` reports **both, side by side**. One is
never silently substituted for the other. The nonlinear leader is unchanged
in both.

---

## 9. Cost and performance are separate axes

No composite efficiency score is defined, and a test asserts the cost table
contains no such column.

| Representation | Dim | Feature s | Linear s | Nonlinear s | Share | A2 rank |
| --- | --- | --- | --- | --- | --- | --- |
| `rdkit_physchem_descriptors` | 217 | 4051 | 124 | 4999 | 9.5% | **1.77** |
| `smiles_tfidf_4096` | 4096 | 160 | 1333 | 18539 | **35.4%** | 3.05 |
| `maccs_keys_167` | 167 | 430 | 81 | 4261 | 8.1% | 4.14 |
| `avalon_1024` | 1024 | 449 | 385 | 6268 | 12.0% | 4.18 |
| `morgan_ecfp4_1024` | 1024 | 106 | 298 | 11580 | 22.1% | 4.68 |
| `erg_reduced_graph_315` | 315 | 186 | 117 | 3450 | 6.6% | 4.77 |
| `rdkit_fragment_descriptors` | 85 | 342 | 32 | 3327 | 6.3% | 5.41 |

Share is of Track A1 nonlinear model compute; the Track A2 figure is 30%.
Both are quoted in C6 so the two tracks cannot look inconsistent.

Wall-clock on one 2-core host with a fixed grid: relative costs are
indicative, not hardware-independent.

---

## 10. Tables and figures

**Tables** (`publication/tables/`)

| File | Content |
| --- | --- |
| `table1_representation_characteristics.csv` | category, dimension, value type, sparsity, artifact dependency, feature cost |
| `table2_a1_primary_performance.csv` | A1 mean ranks by probe and task family, wins, top-3 |
| `table3_a2_robustness.csv` | A1 vs A2 rank, displacement, verdict, per task family |
| `table4_key_statistical_contrasts.csv` | main-text contrasts only — Holm p, rank-biserial, direction |
| `table5_computational_cost.csv` | cost by representation and probe, share, relative cost |
| `table6_endpoint_stability.csv` | Kendall's W and the stability flag |
| `table7_22_vs_19_endpoint_subset.csv` | both endpoint sets side by side |

**Figures** (`publication/figures/`), each with `figure_data/figure_NN_data.csv`

| Figure | Content |
| --- | --- |
| 1 | endpoint × representation rank heatmap, separate linear/nonlinear panels |
| 2 | mean rank with bootstrap CI, one shared scale across panels |
| 3 | A1 → A2 slopegraph per probe |
| 4 | nonlinear rank versus nonlinear compute, two axes |
| 5 | Kendall's W per endpoint — **supplementary** if the main paper is crowded |

Figures are emitted as **deterministic SVG with no plotting dependency**.
The project adds dependencies only after individual approval, and
hand-emitted SVG is byte-identical run to run — no font, backend or library
version can shift it. Every figure is a rendering of its exported CSV, never
a separate calculation, so any tool can redraw it.

Direction-normalised ranks throughout. MAE and AUROC are never averaged.

---

## 11. Methods evidence map

| Methods element | Source of truth |
| --- | --- |
| Dataset freeze | `benchmark/release.py` (`molfusion_frozen_csv_v1`), `benchmark_manifests/tdc_admet_group.json`, `docs/benchmark-data.md` |
| Official split semantics | `benchmark/tdc.py` (`OFFICIAL_TRAIN_VAL_FRACTIONS`, `OFFICIAL_SEEDS`), `docs/benchmark-data.md` |
| A1 splits | `benchmark/a1.py::official_splits`, replayed from frozen membership |
| A2 splits | `benchmark/a2.py::build_splits`, `split_audits.json` |
| Representations | `agents/` registry; `table1_representation_characteristics.csv` |
| Probes | `benchmark/pipelines.py`; `protocol.PROBES` |
| Hyperparameter selection | `pipelines.hyperparameter_grid`; `hyperparameter_audit.csv` |
| Metrics and direction | `benchmark/metrics.py::orient`; `protocol` metric map |
| Statistics | `benchmark/analysis.py` (Friedman, Holm, Wilcoxon, rank-biserial, bootstrap) |
| A1/A2 roles | `protocol.TRACK_A1`, `protocol.TRACK_A2`, `protocol.split_id` |
| Cleaning | `benchmark/a2.py::load_cleaned_endpoint`; `cleaning_effects.csv` |
| Provenance | `benchmark/provenance.py`; `provenance_audit.json`; `docs/benchmark-execution.md` §11 |
| Leakage guards | `a1.verify_leakage_guards`; `split_audits.json` (0 overlaps, 110 splits) |

---

## 12. Reproducibility evidence

Two things must stay distinct, and the package keeps them apart:

**Historical execution** — what produced the published numbers.

| | Commits |
| --- | --- |
| A1 execution | `459653b`, `ddabb42`, `2bcb467` |
| A2 execution | `e6ae297` |
| Analysis | `fe4bc60` (A1), `15b78a2` (A2) |

**Post-hoc infrastructure** — `89335dc`, the Phase 6A.5 provenance
hardening, which **did not produce these results**. It was written
afterwards and must never be presented as the pipeline behind them.

The disclosure the manuscript should carry:

> Historical A1/A2 scientific results remain valid, but 338 of 616 result
> shards lack a recorded Git commit because of a pre-6A.5 worker-local
> provenance implementation in which each worker resolved the commit
> independently and silently recorded a null on failure. The defect was
> fixed after execution; the audit in `provenance_audit.json` separates
> what each shard recorded from what the run provenance demonstrably was,
> and no shard was retrospectively modified.

Raw identities, verified before every use:

| | Scientific identity |
| --- | --- |
| A1 | `d40ef09b398f47914aa51f99fd6a4f5893f7778b50c0cca04404b575632de868` |
| A2 | `9dd5dfa6067c8a760b0bb8fb39648f71f662f2fa1bbf4cc5d7cb0cd495a69f14` |
| A2 analysis | `bda6bd23db77c08a49f8529db609dbc02ecc2136982b8153c6a57cb60c100217` |

---

## 13. Proposed Results outline

Prose is **not** written in Phase 6B. This is the structure the evidence
supports, with each section's governing claims.

| § | Section | Claims |
| --- | --- | --- |
| 3.1 | Benchmark design and representation coverage | Table 1, Table 7 |
| 3.2 | Linear-probe representation performance | **C3** |
| 3.3 | Nonlinear-probe representation performance | **C1**, C10 |
| 3.4 | Robustness under independent scaffold repartitioning | **C2**, C11 |
| 3.5 | Statistical comparison and effect sizes | **C5**, Table 4 |
| 3.6 | Representation stability across endpoints | **C7**, Figure 5 |
| 3.7 | Computational efficiency | **C6**, Table 5 |
| 3.8 | TF-IDF robustness and external corpus exposure | **C4**, **C9** |
| 3.9 | Practical implications | C1, C3, C6 |

One adjustment to the suggested order is worth considering: §3.8 currently
carries both a solid negative result (C4) and an exploratory observation
(C9). Splitting them, or moving C9 to the Limitations, would keep the
exploratory material further from the results — but the given structure
works if C9 stays explicitly framed as untested.

---

## 14. Known manuscript risks

1. **C1 will be read as "descriptors are best".** The prohibited-wording
   list exists for this claim above all.
2. **Non-overlapping intervals will be read as a significance test.** They
   are not; cite the Holm-corrected Wilcoxon for that.
3. **A2 will be read as external validation.** It is the same molecules,
   repartitioned — supplementary, not confirmatory.
4. **C9 will attract causal language.** It is the single most likely
   overclaim in the paper.
5. **Regression conclusions rest on n = 9.**
6. **A2 confounds repartitioning with cleaning.** H6 bounds it; it does not
   separate them.
