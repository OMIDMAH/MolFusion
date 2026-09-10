# Results evidence map (Phase 6C.3)

Paragraph-level mapping for [`RESULTS_DRAFT.md`](RESULTS_DRAFT.md). Every
scientific paragraph maps to at least one registered claim; descriptive
setup paragraphs are marked `DESCRIPTIVE` and carry no interpretation.

Paragraph IDs are tooling identifiers and do not appear in manuscript
prose.

Evidence identity: `5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`.

---

## 3.1 Benchmark coverage and representation characteristics

| ID | Claim | Source | Fields | Figure/Table | Test | Prohibited wording checked |
| --- | --- | --- | --- | --- | --- | --- |
| R3.1-P1 | DESCRIPTIVE | benchmark manifest | endpoint count, task types | — | — | no performance statement |
| R3.1-P2 | DESCRIPTIVE | `table1_representation_characteristics.csv` | `dimension`, `value_type`, `mean_sparsity`, `artifact_dependency` | Table 1 | — | dimension not equated with information content |
| R3.1-P3 | DESCRIPTIVE | `split_distinctness.csv`, `table7_22_vs_19_endpoint_subset.csv` | `mean_pairwise_test_jaccard`, `genuinely_repartitioned` | Table 7 | — | no claim that all 22 gave five distinct partitions |

## 3.2 Linear-probe representation performance

| ID | Claim | Source | Fields | Figure/Table | Test | Prohibited wording checked |
| --- | --- | --- | --- | --- | --- | --- |
| R3.2-P1 | **C3** | — | topic sentence from C3 `recommended_wording` | — | — | "all representations equivalent"; "definitive linear winner" |
| R3.2-P2 | **C3** | `table2_…csv`, `table3_a2_robustness.csv` | `linear_mean_rank`, `a1/a2_mean_rank`, `a1/a2_position` | Tables 2, 3 | — | "Morgan outperformed TF-IDF"; "TF-IDF won" |
| R3.2-P3 | **C3** | `publication_report.json` → `ci_separation.a1/a2.linear`; `bootstrap_mean_rank.csv` | `leader_ci_lower/upper`, `overlapping_competitors` | Figure 2 | bootstrap (visualisation only) | "not significant because intervals overlapped" |
| R3.2-P4 | **C3** | `friedman_a1.csv`, `friedman_a2.csv` | `statistic`, `p_value`, `n_endpoints` | — | Friedman | no leader identified from omnibus |
| R3.2-P5 | DESCRIPTIVE | figure data | — | Figures 1, 2 | — | shared axis noted; no visual magnification |

## 3.3 Nonlinear-probe representation performance

| ID | Claim | Source | Fields | Figure/Table | Test | Prohibited wording checked |
| --- | --- | --- | --- | --- | --- | --- |
| R3.3-P1 | **C1** | `table2_…csv`, `table3_…csv`, `table7_…csv` | `nonlinear_mean_rank` (1.91, 1.77), `nonlinear_wins_22`, `nonlinear_top3_22` | Tables 2, 3, 7 | — | "best representation"; "contains more information" |
| R3.3-P2 | **C1** | `friedman_a1.csv`, `friedman_a2.csv` | nonlinear/all rows | — | Friedman | inference attributed to §3.5, not to ranks |
| R3.3-P3 | **C1** | `publication_report.json` → `ci_separation.*.nonlinear`; `bootstrap_mean_rank.csv` | CI [1.45, 2.41] / [1.32, 2.27]; competitor lower bounds 2.86 / 2.36 | Figure 2 | bootstrap (visualisation only) | "significant because intervals did not overlap" |
| R3.3-P4 | **C1** (restriction) | `friedman_a1.csv` | nonlinear/regression: χ² = 11.33, p = 0.079, reject = False | — | Friedman | **hard prohibition**: no nonlinear regression superiority claim |
| R3.3-P5 | **C10** | `table2_…csv`, `table3_…csv` | probe-wise positions per representation | Figure 1 | — | "TF-IDF for linear, descriptors for nonlinear" |

## 3.4 Robustness under independent scaffold repartitioning

| ID | Claim | Source | Fields | Figure/Table | Test | Prohibited wording checked |
| --- | --- | --- | --- | --- | --- | --- |
| R3.4-P1 | **C2** | manuscript architecture; track roles | — | — | — | "external validation"; "validation cohort" |
| R3.4-P2 | **C2** | `table3_…csv`, `table7_…csv` | `rank_displacement` −0.14; `a2_repartitioned_only_mean_rank` 1.84; linear subset 2.79 | Table 3, Table 7, Figure 3 | — | "confirmed"; "proved"; "validated externally" |
| R3.4-P3 | **C11** | `table3_…csv` | `a1_position`, `a2_position` per probe; bottom-two counts 4/4 and 3/4 | Figure 3 | — | "the bottom two were the same throughout" |
| R3.4-P4 | **C2** (caveat) | Methods §2.6; Amendment C | — | — | — | single clear statement of the confound |
| R3.4-P5 | **C8** | `cleaning_effects.csv` | `raw_rows`, `usable`, `conflicting_molecule_count` — 54.3% / 30.6% | Supp. Table S7 | — | no endpoint claim for `clearance_hepatocyte_az` |

## 3.5 Statistical comparisons and effect sizes

| ID | Claim | Source | Fields | Figure/Table | Test | Prohibited wording checked |
| --- | --- | --- | --- | --- | --- | --- |
| R3.5-P1 | **C5** | Methods §2.10 | procedure only | — | Friedman → Wilcoxon → Holm → rank-biserial | — |
| R3.5-P2 | **C5** | `all_pairwise_contrasts_a1/a2.csv`, `a1_vs_a2_contrasts.csv` | 11/105, 18/126; `reproduced` 9; `effect_direction_preserved` 11 | Table 4, Supp. S1–S2 | Holm-corrected Wilcoxon | "all findings replicated" |
| R3.5-P3 | **C5** | `a1_vs_a2_contrasts.csv` | the two non-reproduced rows, `a1_p_holm` / `a2_p_holm` | Supp. S3 | — | "the difference disappeared"; "became equivalent" |
| R3.5-P4 | **C5** | `table4_…csv`, `all_pairwise_contrasts_a2.csv` | `rank_biserial` ranges 0.71–0.98 and 0.79–0.98; superior-arm counts 10/2; 5 classification, 7 all-endpoint | Table 4 | — | "large effect" without a stated range |

## 3.6 Representation stability across endpoints

| ID | Claim | Source | Fields | Figure/Table | Test | Prohibited wording checked |
| --- | --- | --- | --- | --- | --- | --- |
| R3.6-P1 | **C7** | `table6_endpoint_stability.csv` | `kendall_w_linear`, `kendall_w_nonlinear` range 0.157–1.000 | Supp. Fig S1, Table 6 | Kendall's W | — |
| R3.6-P2 | **C7** | `table6_…csv` | six `LOW` endpoints with `kendall_w_min` | Table 6 | — | "unreliable datasets"; "excluded from the benchmark" |
| R3.6-P3 | **C7** | `table6_…csv` | `vdss_lombardo` W = 0.343, flag `BORDERLINE` | Table 6 | — | must not be listed as LOW |
| R3.6-P4 | **C7** | `table6_…csv` | remaining 15 endpoints, `kendall_w_min` 0.371–0.966 | Table 6 | — | no endpoint-specific superlative |

## 3.7 Computational cost

| ID | Claim | Source | Fields | Figure/Table | Test | Prohibited wording checked |
| --- | --- | --- | --- | --- | --- | --- |
| R3.7-P1 | **C6** | Methods §2.11 | — | — | — | "efficiency score"; "cost-adjusted performance" |
| R3.7-P2 | **C6** | `table5_computational_cost.csv`; A2 `timings.json` | `nonlinear_model_seconds`, `share_of_nonlinear_model_seconds`, `relative_nonlinear_cost` | Table 5, Figure 4 | — | track-specific shares on a common denominator |
| R3.7-P3 | **C6** | `table5_…csv` | `feature_seconds` vs `nonlinear_model_seconds` | Table 5 | — | — |
| R3.7-P4 | **C6** | `table5_…csv` | `nonlinear_mean_rank` 1.77 / 3.05 | Figure 4 | — | "most efficient representation" |

## 3.8 TF-IDF predictive robustness

| ID | Claim | Source | Fields | Figure/Table | Test | Prohibited wording checked |
| --- | --- | --- | --- | --- | --- | --- |
| R3.8-P1 | **C4** | `table2_…csv`, `table3_…csv` | linear 2.55 / 3.00; nonlinear 4.14 → 3.05 | Tables 2, 3 | — | "TF-IDF is inferior" |
| R3.8-P2 | **C4** | `representation_ranks.csv` (both tracks, `subset = regression`) | `mean_rank` 1.33 → 2.33; `wins` 7/9 → 4/9; `top3` 9/9 → 7/9 | Table 3, Supp. S4 | — | "TF-IDF failed"; "does not generalize" |
| R3.8-P3 | **C4** (limits) | `friedman_a2.csv`; `all_pairwise_contrasts_a2.csv` | n = 9; no regression-only contrast significant after Holm | — | Holm-corrected Wilcoxon | "shown to be no better than other representations" |

## 3.9 External unsupervised corpus exposure

| ID | Claim | Source | Fields | Figure/Table | Test | Prohibited wording checked |
| --- | --- | --- | --- | --- | --- | --- |
| R3.9-P1 | **C9** | Methods §2.2; TF-IDF artifact contract | fitted before evaluation, no labels | — | — | "leakage"; "contamination" |
| R3.9-P2 | **C9** | `chembl_exposure.csv` | `chembl37_overlap_fraction` range 23.4%–99.6%, median 80.4%; six endpoints ≥ 90% | Supp. Table S5 | — | — |
| R3.9-P3 | **C9** | frozen analysis plan | no exposure test exists | — | **none run** | **hard prohibition**: no causal statement |

## 3.10 Practical implications

| ID | Claim | Source | Fields | Figure/Table | Test | Prohibited wording checked |
| --- | --- | --- | --- | --- | --- | --- |
| R3.10-P1 | DESCRIPTIVE | — | scope statement | — | — | no mechanistic claim |
| R3.10-P2 | **C10**, C1, C3 | §3.2, §3.3 | — | Figure 1 | — | "universally best" |
| R3.10-P3 | **C1**, C11 | `table1_…csv`, `table2_…csv`, `table3_…csv` | dimension vs nonlinear rank | Tables 1–3 | — | dimension not causal |
| R3.10-P4 | **C2**, C5, C4 | §3.4, §3.5, §3.8 | — | — | — | "confirmed"; "replicated" |
| R3.10-P5 | **C6**, C10 | §3.7 | — | Figure 4 | — | no composite metric; no biological claim |

---

## Claim coverage

| Claim | Primary home | Also referenced | Registry section (frozen) |
| --- | --- | --- | --- |
| C1 | §3.3 | §3.10 | 3.3 ✓ |
| C2 | §3.4 | §3.10 | 3.4 ✓ |
| C3 | §3.2 | §3.10 | 3.2 ✓ |
| C4 | **§3.8 only** | §3.10 (synthesis) | 3.8 ✓ |
| C5 | §3.5 | §3.10 | 3.5 ✓ |
| C6 | §3.7 | §3.10 | 3.7 ✓ |
| C7 | §3.6 | — | 3.6 ✓ |
| C8 | §3.4 | — | 3.4 ✓ |
| C9 | **§3.9 only** | — | 3.9 ✓ |
| C10 | §3.3 | §3.10 | 3.3 ✓ |
| C11 | §3.4 | §3.10 | 3.4 ✓ |

All eleven registered claims appear; no claim beyond C1–C11 is introduced.
C4 and C9 each have exactly one primary home, matching the frozen
claim-to-section map.

## Numerical provenance

| Value | Source |
| --- | --- |
| 22 / 13 / 9 / 7 / 2 | benchmark manifest; `protocol.TRACK_A_REPRESENTATIONS`, `protocol.PROBES` |
| 85 / 167 / 217 / 315 / 1024 / 1024 / 4096 | `table1_representation_characteristics.csv` |
| sparsity 0.496, 0.984 | `table1_…csv` |
| 19 of 22 repartitioned | `table7_…csv`, `split_distinctness.csv` |
| linear A1 2.55, 3.14, 3.73, 3.82 | `table2_a1_primary_performance.csv` |
| linear A2 2.77, 3.00, 3.77, 2.79 | `table3_…csv`, `table7_…csv` |
| linear CIs [1.82, 3.36], [2.14, 3.50]; 3 and 4 overlaps | `publication_report.json` `ci_separation` |
| Friedman linear all: 28.23 / 8.5e-05; 31.87 / 1.7e-05 | `friedman_a1.csv`, `friedman_a2.csv` |
| nonlinear physchem 1.91, 1.77, 1.84 | `table2_…`, `table3_…`, `table7_…` |
| wins 12/22, 13/22; top-3 20/22, 19/22 | `table2_…csv`, `table7_…csv` |
| Friedman nonlinear all: 29.92 / 4.1e-05; 42.29 / 1.6e-07 | `friedman_a1/a2.csv` |
| nonlinear CIs [1.45, 2.41], [1.32, 2.27]; bounds 2.86, 2.36 | `bootstrap_mean_rank.csv` both tracks |
| **A1 nonlinear regression Friedman 11.33 / p = 0.079 / no rejection** | `friedman_a1.csv` |
| displacement −0.14; five of seven ≤ 0.5 | `table3_…csv` `rank_displacement` |
| bottom-two 4/4 and 3/4 | `table3_…csv` positions (derived, matches C11 basis) |
| cleaning 54.3%, 30.6%; 521, 178 conflicts | `cleaning_effects.csv` |
| 11/105 and 18/126 significant | `all_pairwise_contrasts_a1/a2.csv` |
| 9 reproduced, 11 direction preserved | `a1_vs_a2_contrasts.csv` |
| Holm p 0.010 → 0.100; 0.005 → 0.116 | `a1_vs_a2_contrasts.csv` |
| \|r\| 0.71–0.98 and 0.79–0.98; 10/2 superior arms; 5 classification, 7 all | `all_pairwise_contrasts_a2.csv`, `table4_…csv` |
| Kendall's W 0.157–1.000; six LOW values; `vdss_lombardo` 0.343; 15 others 0.371–0.966 | `table6_endpoint_stability.csv` |
| A1 cost 18,539 s / 35.4%; 4,999 s / 9.5%; 1.00×–5.57× | `table5_computational_cost.csv` |
| **A2 cost shares 30.7% and 9.0%** | A2 `timings.json`, nonlinear-only denominator (see note) |
| feature 4,051 s and 160 s | `table5_…csv` |
| TF-IDF regression 1.33 → 2.33; 7/9 → 4/9; 9/9 → 7/9 | `representation_ranks.csv`, `subset = regression` |
| exposure 23.4%–99.6%, median 80.4%; six ≥ 90% | `chembl_exposure.csv` |

### Note on the Track A2 compute share

The Phase 6C.3 brief quotes an A2 TF-IDF nonlinear compute share of 29.8%
alongside the A1 figure of 35.4%. These two are computed on **different
denominators** and are not directly comparable:

| Quantity | A1 | A2 |
| --- | --- | --- |
| TF-IDF nonlinear ÷ **nonlinear-only** model compute | **35.4%** | **30.7%** |
| TF-IDF nonlinear ÷ **all** model compute (linear + nonlinear) | 33.8% | 29.8% |

`table5_computational_cost.csv` defines
`share_of_nonlinear_model_seconds` on the nonlinear-only denominator, which
gives the A1 value of 35.4%. The brief's 29.8% is the A2 value on the
all-model denominator; the like-for-like A2 figure is **30.7%**.

The Results draft uses the nonlinear-only denominator for both tracks —
35.4% and 30.7% — so the two are comparable. Both denominators are recorded
here so the 29.8% figure is reconciled rather than contradicted. No frozen
table was modified.
