# 3. Results — draft (Phase 6C.3)

Evidence source: Phase 6B publication package, identity
`5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`
(verified before and after drafting). Every numerical statement traces to a
frozen Phase 6B table; paragraph-level mapping is in
[`RESULTS_EVIDENCE_MAP.md`](RESULTS_EVIDENCE_MAP.md).

This document contains the Results section only.

---

## 3.1 Benchmark coverage and representation characteristics

The benchmark comprised **22 ADMET endpoints** from the Therapeutics Data
Commons — **13 classification** and **9 regression** tasks — evaluated
across **seven fixed-vector representations** and **two probe families**,
with five partition realisations per cell. All 22 endpoints were retained
throughout; none was added, removed, or reweighted after results were
observed.

The seven representations differ substantially in family, dimensionality
and value type (Table 1). Dimensions span nearly two orders of magnitude:
`rdkit_fragment_descriptors` (85 count-valued dimensions),
`maccs_keys_167` (167 binary), `rdkit_physchem_descriptors` (217
continuous), `erg_reduced_graph_315` (315 continuous), `morgan_ecfp4_1024`
and `avalon_1024` (1024 binary each), and `smiles_tfidf_4096` (4096
continuous). Mean sparsity likewise varies, from 0.496 for the
physicochemical descriptors to 0.984 for the TF-IDF representation. Only
`smiles_tfidf_4096` depends on a fitted artifact; the remaining six are
stateless functions of molecular structure. Dimensionality is reported here
as a structural property of each encoding and is not interpreted as a
measure of information content.

Track A2 repartitioned each endpoint independently by Bemis–Murcko
scaffold. Because the frozen splitter permutes only within equal-size
scaffold groups, the five seeds did not produce five distinct test
partitions at every endpoint: **19 of the 22 endpoints** met the
pre-specified distinctness criterion (mean pairwise test-set Jaccard
similarity ≤ 0.50) and are designated *genuinely repartitioned*. All 22
endpoints remain in the headline Track A2 analysis, and the 19-endpoint set
is reported alongside it as a sensitivity subset (Table 7).

---

## 3.2 Linear-probe representation performance

**No single representation separated clearly from the field under the
linear probe.** The evidence for this is of two kinds: the identity of the
top-ranked representation was not stable across evaluation tracks, and the
bootstrap intervals of the leading representations overlapped one another
substantially.

Under Track A1, `smiles_tfidf_4096` attained the lowest linear-probe mean
rank (2.55), ahead of `rdkit_physchem_descriptors` (3.14), `avalon_1024`
(3.73) and `morgan_ecfp4_1024` (3.82) (Table 2). Under Track A2 the
ordering of the leading group changed: `morgan_ecfp4_1024` attained the
lowest linear-probe mean rank (2.77), with `smiles_tfidf_4096` second
(3.00) and `rdkit_physchem_descriptors` fourth (3.77) (Table 3). Neither
representation is identified here as the leading linear representation.

Bootstrap intervals overlapped substantially among the leading linear
representations in both tracks (Figure 2). In Track A1 the interval for the
top-ranked representation, [1.82, 3.36], overlapped those of three
competing representations; in Track A2 the interval for the top-ranked
representation, [2.14, 3.50], overlapped those of four. These intervals
characterise uncertainty in mean rank and are not used here as a test of
difference.

Friedman tests did indicate differences among representations under the
linear probe in both tracks — across all 22 endpoints, χ²(6) = 28.23,
p = 8.5 × 10⁻⁵ in Track A1 and χ²(6) = 31.87, p = 1.7 × 10⁻⁵ in Track A2 —
so the field as a whole is not exchangeable. What the evidence does not
support is the identification of a single leading linear representation:
the differences that reach significance after correction (§3.5) involve the
lowest-ranked representations rather than separating the leading group.

The endpoint-level rank pattern underlying these summaries is shown in
Figure 1 (linear panel), and the corresponding mean ranks with bootstrap
intervals in Figure 2 (linear panel). Both figure panels share a common
rank axis, so the compressed spread of the linear results is displayed at
the same scale as the nonlinear results.

---

## 3.3 Nonlinear-probe representation performance

**Under the nonlinear probe, the 217-dimensional RDKit physicochemical
descriptor representation achieved the lowest cross-endpoint mean rank.**
This was the case in both evaluation tracks: mean rank **1.91** in Track A1
and **1.77** in Track A2 (Tables 2 and 3). In Track A1 it attained the
best rank at 12 of 22 endpoints and placed in the top three at 20; in Track
A2, at 13 and 19 endpoints respectively (Tables 2 and 7).

Friedman tests indicated differences among representations under the
nonlinear probe across all 22 endpoints in both tracks: χ²(6) = 29.92,
p = 4.1 × 10⁻⁵ in Track A1, and χ²(6) = 42.29, p = 1.6 × 10⁻⁷ in Track A2.
Inferential support for the pairwise ordering comes from the Holm-corrected
paired Wilcoxon analysis reported in §3.5, in which the physicochemical
representation was the superior arm of several corrected contrasts with
matched-pairs rank-biserial correlations of |r| ≥ 0.79.

The bootstrap interval for the nonlinear physicochemical representation was
separated from those of all competing representations in both tracks:
[1.45, 2.41] in Track A1 and [1.32, 2.27] in Track A2, against a lowest
competing interval bound of 2.86 and 2.36 respectively (Figure 2,
nonlinear panel). This separation is reported as uncertainty evidence.
Statistical inference rests on the Friedman, Holm-corrected Wilcoxon and
rank-biserial analysis, not on interval separation.

This nonlinear result does not extend to a regression-specific claim. The
Track A1 nonlinear Friedman test restricted to the nine regression
endpoints did not reject the null hypothesis (χ²(6) = 11.33, p = 0.079), so
no claim of nonlinear superiority on regression endpoints is made, in
either track.

**Representation ranking depended on the probe.** The representation with
the lowest mean rank under the linear probe was not the representation with
the lowest mean rank under the nonlinear probe, in either track. The
contrast is visible across the two panels of Figure 1: `smiles_tfidf_4096`
ranked first under the linear probe and fourth under the nonlinear probe in
Track A1, while `rdkit_physchem_descriptors` ranked second and first
respectively. Representation comparisons obtained under one probe family
therefore do not transfer to the other.

---

## 3.4 Robustness under independent scaffold repartitioning

Track A1 remains the primary, TDC-comparable evaluation; Track A2 is a
supplementary robustness evaluation applying independent MolFusion scaffold
repartitioning together with stricter curation to the same underlying
benchmark datasets.

**The nonlinear leadership of the physicochemical descriptor representation
was reproduced under Track A2.** Its nonlinear mean rank moved from 1.91 to
1.77, a displacement of −0.14, within the pre-registered tolerance of 0.5
rank positions, and it held the first position in both tracks (Table 3,
Figure 3). Restricting the Track A2 analysis to the 19 genuinely
repartitioned endpoints gives a nonlinear mean rank of 1.84 and leaves the
ordering of the leading representations unchanged (Table 7). Under the
linear probe, by contrast, the leading position changed between tracks
(§3.2), and that instability persists in the 19-endpoint subset, where
`morgan_ecfp4_1024` again holds the lowest linear mean rank (2.79).

Rank displacement between tracks was modest for most representations under
the nonlinear probe: five of seven moved by 0.5 rank positions or less, and
the ordering of the lowest-ranked representations was largely preserved
(Figure 3). **`rdkit_fragment_descriptors` occupied a bottom-two position
in all four probe × track combinations**, and `erg_reduced_graph_315` did
so in three of four; the exception is the Track A1 nonlinear probe, where
`erg_reduced_graph_315` ranked fifth. No representation held a bottom-two
position in every combination other than `rdkit_fragment_descriptors`.

Track A2 alters both the partitioning scheme and the curation policy
relative to Track A1. The separate causal contributions of these two
changes cannot be identified from this design, and Track A2 results are
therefore reported as joint robustness evidence rather than as an isolated
test of repartitioning.

Two endpoints are affected most by the stricter curation: `ppbr_az`, for
which 54.3% of official records are removed (521 conflicting molecules),
and `clearance_hepatocyte_az`, for which 30.6% are removed (178 conflicting
molecules). Excluding these two endpoints from the cross-endpoint analysis
left the nonlinear leading representation unchanged. No endpoint-specific
performance claim is made for `clearance_hepatocyte_az`, which is also
among the low-stability endpoints identified in §3.6.

---

## 3.5 Statistical comparisons and effect sizes

Pairwise comparisons were performed only within probe × task families whose
Friedman test rejected at α = 0.05, using paired Wilcoxon signed-rank tests
over endpoint-level scores with Holm correction applied within each family,
and matched-pairs rank-biserial correlation reported as effect size.

Under Track A1, 11 of 105 pairwise contrasts were significant after Holm
correction; under Track A2, 18 of 126. **Of the 11 contrasts that were
significant after Holm correction in Track A1, 9 remained significant in
Track A2, and effect direction was preserved in all 11.** The 12 contrasts
of principal interest are given in Table 4; the complete sets for both
tracks are provided in Supplementary Tables S1 and S2.

The two contrasts that did not remain significant were
`maccs_keys_167` versus `smiles_tfidf_4096` under the linear probe across
all endpoints (Holm-adjusted p = 0.010 in Track A1, p = 0.100 in Track A2)
and `erg_reduced_graph_315` versus `rdkit_physchem_descriptors` under the
linear probe on classification endpoints (p = 0.005 and p = 0.116). In both
cases the contrast did not remain statistically significant after Holm
correction under Track A2, although its effect direction was preserved.
Failure to reject is not evidence of equivalence between the
representations concerned.

Effect sizes for the corrected contrasts were large throughout: among the
Track A2 contrasts significant after Holm correction, the matched-pairs
rank-biserial correlation ranged from **|r| = 0.71 to 0.98**, and among the
12 contrasts in Table 4 from |r| = 0.79 to 0.98.
`rdkit_physchem_descriptors` is the superior arm in 10 of those 12
contrasts, nine of them under the nonlinear probe; the remaining two have
`smiles_tfidf_4096` as the superior arm under the linear probe, in each
case against one of the two lowest-ranked representations. Five of the 12
are classification-family contrasts and seven are computed across all 22
endpoints.

---

## 3.6 Representation stability across endpoints

Because the five Track A2 partitions are different evaluation sets, the
stability of a representation ordering can be quantified at the endpoint
level. Kendall's coefficient of concordance across the five partitions
varied widely between endpoints (Supplementary Figure S1, Table 6),
from W = 0.157 to W = 1.000 depending on endpoint and probe.

Six endpoints were pre-registered as low-stability under the criterion that
the weaker of their two probe values falls below W = 0.35: **`herg`
(W = 0.157), `cyp2c9_substrate_carbonmangels` (0.171),
`clearance_hepatocyte_az` (0.209), `cyp3a4_substrate_carbonmangels`
(0.217), `cyp2d6_substrate_carbonmangels` (0.312) and `bioavailability_ma`
(0.331)**. Their endpoint-specific representation rankings were
insufficiently stable across partitions for strong individual
interpretation, and no endpoint-specific claim is drawn from them. All six
remain in the 22-endpoint cross-endpoint analyses reported throughout this
section; none was excluded from the benchmark.

One further endpoint, `vdss_lombardo`, falls marginally below the same
threshold on its weaker probe (nonlinear W = 0.343, against the predefined
threshold of 0.35). It is reported as **BORDERLINE** and was retained,
because modifying the pre-specified exclusion set after observing results
would be post-hoc.

The remaining 15 endpoints show substantially more stable orderings, with
minimum Kendall's W between 0.371 and 0.966. Endpoint stability is not
uniform across the benchmark, and cross-endpoint summaries in this section
aggregate over both stable and unstable endpoints.

---

## 3.7 Computational cost

Computational cost is reported as an evidence axis separate from predictive
rank. No cost-adjusted or composite performance measure was computed, and
no predictive rank reported in this section was influenced by timing.
All timings are single-host measurements collected under a fixed
hyperparameter grid and are interpretable only as relative costs within
this study.

Nonlinear model-fitting cost varied by more than a factor of five across
representations (Table 5, Figure 4). In Track A1, `smiles_tfidf_4096`
required 18,539 s of nonlinear model time, **35.4% of all nonlinear model
compute**, while `rdkit_physchem_descriptors` required 4,999 s, **9.5%**.
The same pattern held in Track A2, where `smiles_tfidf_4096` accounted for
**30.7%** of nonlinear model compute and `rdkit_physchem_descriptors` for
9.0%. Relative to the least expensive representation
(`rdkit_fragment_descriptors`), nonlinear cost ranged from 1.00× to 5.57×.

Feature-generation cost follows a different ordering from model-fitting
cost. `rdkit_physchem_descriptors` was the most expensive representation to
compute (4,051 s across all endpoints) but among the least expensive to fit
under the nonlinear probe, whereas `smiles_tfidf_4096` was inexpensive to
compute (160 s, using its pre-fitted artifact) and the most expensive to
fit. Feature matrices were computed once per endpoint and representation
and reused across probes and seeds, so feature cost is incurred once
whereas model-fitting cost scales with the number of probes, seeds and
hyperparameter candidates.

The compact physicochemical representation combined the strongest nonlinear
mean rank (1.77 in Track A2) with substantially lower within-study
model-fitting cost than the 4096-dimensional TF-IDF representation
(3.05). Predictive rank and computational cost are shown on separate axes
in Figure 4; no efficiency metric combining them is defined.

---

## 3.8 TF-IDF predictive robustness

The SMILES TF-IDF representation produced a mixed profile across probes,
task families and tracks. It attained the lowest linear-probe mean rank in
Track A1 (2.55) and the second lowest in Track A2 (3.00), and improved its
nonlinear-probe position between tracks, from fourth (4.14) to second
(3.05) (Tables 2 and 3).

**The linear regression advantage observed for TF-IDF in Track A1 weakened
under the Track A2 robustness evaluation.** Restricted to the nine
regression endpoints under the linear probe, its mean rank moved from
**1.33 in Track A1 to 2.33 in Track A2**; it attained the best rank at
**7 of 9** regression endpoints in Track A1 and **4 of 9** in Track A2, and
placed in the top three at **9 of 9** and **7 of 9** endpoints
respectively. The displacement of 1.00 rank positions exceeds the
pre-registered tolerance of 0.5.

Two limits on the strength of this observation should be read alongside it.
The regression family contains **n = 9 endpoints**, the smallest of the
task families analysed. And no regression-only pairwise contrast survived
Holm correction in Track A2, so the weakening is a rank-level observation
rather than a statistically corrected pairwise result. The evidence
supports a weakening of the advantage originally observed under the
official partition; it does not establish that the representation performs
poorly on regression endpoints.

---

## 3.9 External unsupervised corpus exposure

The TF-IDF vocabulary and IDF weights were fitted on a frozen ChEMBL 37
reference corpus before any endpoint evaluation, without access to
benchmark labels. Because that corpus is a general-purpose compound
library, a proportion of benchmark molecules also occur in it. This is
**external unsupervised corpus exposure**, and it is quantified here for
transparency.

Exposure varied substantially across endpoints, from 23.4%
(`vdss_lombardo`) to 99.6% (`clearance_hepatocyte_az`,
`clearance_microsome_az`), with a median of 80.4% across the 22 endpoints
(Supplementary Table S5). Six endpoints show at least 90% overlap:
`clearance_hepatocyte_az`, `clearance_microsome_az` (99.6% each),
`ppbr_az` (99.5%), `lipophilicity_astrazeneca` (99.4%),
`half_life_obach` (97.1%) and `bioavailability_ma` (93.0%).

No molecule was removed from any endpoint on the basis of corpus overlap,
and **no analysis in this study tests exposure as a factor influencing
representation performance**. The frozen statistical plan contains no such
test and none was performed. The relationship between corpus exposure and
TF-IDF performance is untested here, is confounded with task family — five
of the six high-exposure endpoints are regression endpoints — and remains
an open question. This exposure audit is reported as an observational
characteristic of the benchmark, not as an explanation of any result in
§3.2, §3.3 or §3.8.

---

## 3.10 Practical implications

The results above bear on how representation comparisons should be
conducted and reported, rather than on the chemistry of any individual
endpoint.

Representation ranking was contingent on the predictive probe. The
representation attaining the lowest mean rank under the linear probe was
not the one attaining the lowest mean rank under the nonlinear probe in
either track (§3.3), and a comparison performed with one probe family
therefore does not determine the ordering that another would produce. A
representation ranking reported without specifying the probe is
underdetermined.

Larger representation dimensionality did not correspond to a stronger
cross-endpoint ranking under the frozen nonlinear probe. The
217-dimensional physicochemical representation attained the lowest
nonlinear mean rank in both tracks, while the 4096-dimensional TF-IDF
representation ranked second in Track A2 and fourth in Track A1, and the
two 1024-dimensional fingerprints ranked fourth and fifth in Track A2
(Tables 2 and 3). The 85-dimensional fragment-count representation
consistently occupied a bottom-two position (§3.4). Dimensionality alone
did not order the representations in either direction.

Conclusions differed in how well they survived a change of partitioning
scheme. The nonlinear ranking was reproduced under independent scaffold
repartitioning, and 9 of 11 corrected contrasts remained significant with
direction preserved in all 11 (§3.4, §3.5); the linear-probe leading
position and the TF-IDF regression advantage did not reproduce in the same
way (§3.2, §3.8). Robustness under repartitioning is therefore an
informative property of a benchmark result and is not implied by
significance under a single partition.

Computational cost separated the representations by more than a factor of
five under the nonlinear probe and did not follow predictive rank (§3.7).
Taken together, these results indicate that molecular representation
comparisons should report both the representation and the predictive probe,
while treating robustness under repartitioning and computational cost as
separate evaluation dimensions.
