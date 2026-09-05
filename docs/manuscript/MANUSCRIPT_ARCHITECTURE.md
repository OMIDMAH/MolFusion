# Manuscript architecture — frozen (Phase 6C.1)

Structure only. No Methods, Results or Discussion prose is written in this
phase.

Evidence source: Phase 6B publication package, identity
`5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`.

---

## 1. Three frozen editorial decisions

### Decision 1 — `vdss_lombardo` stays BORDERLINE

It is **not** added to the six pre-registered low-stability endpoints. Its
threshold crossing (nonlinear Kendall's W = 0.343, against a 0.35
threshold) was identified *after* observing results, and widening a
pre-registered exclusion on that basis converts a caveat list into a
post-hoc filter.

It is reported transparently in §3.6 and in Supplementary Figure S1 with
the `BORDERLINE` label, and it remains in every cross-endpoint analysis.

### Decision 2 — TF-IDF robustness and corpus exposure are separate sections

The solid negative result (C4, TF-IDF regression robustness) and the
exploratory observation (C9, external corpus exposure) are never combined.
They occupy **§3.8** and **§3.9** respectively. No sentence may imply that
corpus exposure caused any TF-IDF performance difference; the frozen
analysis plan contains no test of it, and none was run.

### Decision 3 — Endpoint stability is Supplementary Figure S1

Main text carries Figures 1–4. The Kendall's W endpoint-stability figure
becomes **Supplementary Figure S1**, promoted only if a journal-specific
layout strongly justifies it.

---

## 2. Section structure

```
1.   Introduction

2.   Materials and Methods
2.1  MolFusion framework
2.2  Molecular representations
2.3  TDC ADMET benchmark
2.4  Dataset curation and identity
2.5  Track A1 — official evaluation
2.6  Track A2 — independent scaffold robustness evaluation
2.7  Predictive probes
2.8  Hyperparameter selection
2.9  Evaluation metrics
2.10 Statistical analysis
2.11 Computational cost measurement
2.12 Reproducibility and provenance

3.   Results
3.1  Benchmark coverage and representation characteristics
3.2  Linear-probe representation performance
3.3  Nonlinear-probe representation performance
3.4  Robustness under independent scaffold repartitioning
3.5  Statistical contrasts and effect sizes
3.6  Representation stability across endpoints
3.7  Computational cost
3.8  TF-IDF predictive robustness
3.9  External unsupervised corpus exposure
3.10 Practical implications

4.   Discussion
5.   Limitations
6.   Conclusion
```

### Introduction — argument sequence

1. Representation choice is a primary design decision in molecular property
   prediction.
2. Published comparisons are hard to compare: splits, curation and probe
   families vary, and protocol decisions are often made after seeing
   results.
3. Representation and downstream model are rarely varied together, so a
   reported ranking may be a property of the probe rather than the
   representation.
4. Gap: a frozen-protocol, provenance-tracked comparison of heterogeneous
   representations evaluated under more than one model family and more than
   one partitioning scheme.
5. This work: MolFusion, seven fixed-vector representations, 22 ADMET
   endpoints, two probes, two evaluation tracks.
6. Contributions (four, as frozen in `TITLE_AND_ABSTRACT.md`).

### Methods — what each subsection must establish

| § | Establishes | Source of truth |
| --- | --- | --- |
| 2.1 | framework, agent registry, feature caching, shard/resume design | `docs/benchmark-execution.md` |
| 2.2 | the seven representations and their accurate categories | Table 1 |
| 2.3 | TDC ADMET group, 22 endpoints, inclusion rule | benchmark manifest |
| 2.4 | frozen serialization contract, content-derived identities, curation policy | `docs/benchmark-data.md` |
| 2.5 | official partition semantics: one shipped test set, five train/validation realizations, seeds 1–5 | `benchmark/tdc.py` |
| 2.6 | Bemis–Murcko repartitioning, seeds 0–4, stricter curation, distinctness audit | `benchmark/a2.py` |
| 2.7 | regularized linear probe; gradient-boosting probe; leakage prevented structurally by pipeline | `benchmark/pipelines.py` |
| 2.8 | fixed grid, validation-selected, identical budget across representations | `hyperparameter_audit.csv` |
| 2.9 | per-endpoint primary metric and its direction; why metrics are never averaged | `benchmark/metrics.py` |
| 2.10 | endpoint as unit of inference; Friedman → Holm-Wilcoxon → rank-biserial → endpoint bootstrap | `benchmark/analysis.py` |
| 2.11 | wall-clock, single host, fixed grid — explicitly not hardware-portable | Table 5 |
| 2.12 | identities, execution commits, and the historical null-shard defect | `docs/benchmark-execution.md` §11 |

**§2.5 must state plainly** that the five official seeds re-split only
train/validation against one fixed test set, so they are not five
independent test sets. Every later statistical statement depends on it.

**§2.12 must disclose** that 338 of 616 shards lack a recorded commit
because of a pre-6A.5 worker-local provenance defect fixed *after*
execution, and that the hardened code did not produce these results.

### Results — subsection contents

| § | Content | Claims | Figures/Tables |
| --- | --- | --- | --- |
| 3.1 | 22 endpoints × 7 representations × 2 probes × 5 seeds; matrix completeness; representation characteristics; 22 vs 19 endpoint subsets | — | Table 1, Table 7 |
| 3.2 | linear ranks; leaders differ across tracks; intervals overlap; **no clear separation** | **C3** | Fig 1 (linear panel), Fig 2 (linear panel), Table 2 |
| 3.3 | nonlinear ranks; physchem leads (1.91 A1, 1.77 A2); Friedman + Holm-Wilcoxon + rank-biserial | **C1**, C10 | Fig 1 (nonlinear panel), Fig 2 (nonlinear panel), Table 2 |
| 3.4 | A1 → A2 displacement; nonlinear leader unchanged; bottom-rank stability | **C2**, C11 | Fig 3, Table 3 |
| 3.5 | omnibus results; Holm-corrected contrasts with effect sizes; 9/11 reproduced, direction 11/11 | **C5** | Table 4, Supp. Tables S2–S3 |
| 3.6 | Kendall's W; six pre-registered low-stability endpoints; `vdss_lombardo` BORDERLINE; all retained | **C7** | **Fig S1**, Table 6 |
| 3.7 | cost by representation and probe; separate axis, no composite score | **C6** | Fig 4, Table 5 |
| 3.8 | TF-IDF mixed profile; regression rank 1.33 → 2.33, wins 7/9 → 4/9, top-3 9/9 → 7/9 | **C4** | Table 3, Supp. Table S4 |
| 3.9 | external unsupervised corpus exposure; six endpoints ≥ 90%; explicitly untested | **C9** | Supp. Table S5 |
| 3.10 | what a practitioner should take from this | C1, C3, C6 | — |

### Discussion — five threads

1. **Probe dependence as the central result.** A representation ranking is
   a joint property of representation and probe; reporting one without the
   other is incomplete. *(C1, C3, C10)*
2. **Why a compact descriptor set can lead a nonlinear probe.** Framed as
   *predictive accessibility* under a fixed probe and budget — never as
   information content. *(C1)*
3. **What robustness under repartitioning does and does not show.** Same
   molecules, repartitioned — supplementary, not external validation.
   *(C2, C5)*
4. **Cost as an independent decision axis.** *(C6)*
5. **Protocol freezing and provenance as a methodological argument.** Where
   the discipline changed what could be claimed — including the two
   contrasts that lost significance and the endpoints excluded from
   per-endpoint claims. *(C5, C7)*

### Limitations — eight, all pre-identified

1. Seven fixed-vector representations only; no learned or pretrained
   representations, and SELFIES was not part of Track A.
2. Two probe families with a fixed grid; a different budget could reorder
   results.
3. Regression rests on n = 9 endpoints; **the A1 nonlinear regression-only
   Friedman does not reject (p = 0.079)**, so no regression-specific
   nonlinear claim is made.
4. Six endpoints do not support per-endpoint claims; `vdss_lombardo` is
   borderline. *(C7)*
5. Track A2 varies repartitioning and curation together and does not
   separate them. *(C2, C8)*
6. External corpus exposure is reported but untested and confounded with
   task type. *(C9)*
7. Costs are single-host wall-clock, not hardware-portable. *(C6)*
8. 338 of 616 shards lack recorded commit provenance; reconstructed and
   audited, never backfilled.

### Conclusion

One paragraph, drawn only from conclusion-permitted claims. Must state the
probe-dependent framing, the nonlinear finding with its robustness, and the
linear negative result. **C9 must not appear.**

---

## 3. Figure and table placement

### Main figures

| Figure | Content | First appears | Source |
| --- | --- | --- | --- |
| 1 | Representation rank heatmap, endpoint × representation, separate probe panels | §3.2 | `figure_01_rank_heatmap.svg` / `figure_01_data.csv` |
| 2 | Mean rank with bootstrap CI, shared scale across panels | §3.2 | `figure_02_mean_rank_ci.svg` / `figure_02_data.csv` |
| 3 | A1 → A2 rank robustness slopegraph | §3.4 | `figure_03_rank_robustness.svg` / `figure_03_data.csv` |
| 4 | Predictive rank versus computational cost | §3.7 | `figure_04_rank_vs_cost.svg` / `figure_04_data.csv` |

Figures 1 and 2 both first appear in §3.2 and are referenced again in §3.3;
each is *placed* once, at first appearance.

### Supplementary figures

| Figure | Content | Referenced | Source |
| --- | --- | --- | --- |
| S1 | Endpoint stability, Kendall's W per endpoint | §3.6 | `figure_05_endpoint_stability.svg` / `figure_05_data.csv` |

### Main tables

| Table | Content | First appears |
| --- | --- | --- |
| 1 | Representation characteristics | §3.1 |
| 2 | Track A1 primary performance | §3.2 |
| 3 | A1/A2 robustness | §3.4 |
| 4 | Key statistical contrasts (12 of 126) | §3.5 |
| 5 | Computational cost | §3.7 |
| 6 | Endpoint stability | §3.6 |
| 7 | 22- versus 19-endpoint subsets | §3.1 |

Tables 6 and 7 are candidates for demotion to supplementary if the main
text is crowded; 1–5 are main-text.

### Supplementary tables

| Table | Content |
| --- | --- |
| S1 | All 126 A2 pairwise contrasts |
| S2 | All 105 A1 pairwise contrasts |
| S3 | A1 → A2 contrast reproduction (11 rows) |
| S4 | Endpoint × representation ranks, both tracks |
| S5 | ChEMBL 37 exposure per endpoint |
| S6 | Split stability detail |
| S7 | Cleaning effects and split distinctness |
| S8 | Friedman omnibus, both tracks |

---

## 4. Statistical attribution rule

Statistical support is attributed **only** to:

```
Friedman omnibus  →  Holm-corrected paired Wilcoxon  →  rank-biserial effect size
```

Bootstrap confidence intervals appear in Figure 2 and §3.3 as **uncertainty
visualization**. Interval non-overlap is never presented as a significance
test, even though the nonlinear leader's interval is in fact clear of every
competitor's in both tracks. That separation may be *described*; it may not
be used as the inferential basis.

The endpoint is the unit of inference throughout. The five seeds are never
treated as independent replicates, and A1 and A2 rows are never pooled.
