# Table captions (Phase 6C.5)

Each caption defines its abbreviations, names the track and probe where
relevant, states the rank direction, and identifies the unit of inference
for any statistic. No caption introduces an interpretation absent from
Results.

---

## Main table captions

**Table 1. Characteristics of the seven fixed-vector representations.**
For each representation: its family, output dimension, value type (binary,
count or continuous), mean sparsity across the benchmark molecule set (the
fraction of vector entries equal to zero), whether it depends on a fitted
artifact, and the total wall-clock time to compute its feature matrices
across all 22 endpoints. Feature matrices are computed once per endpoint and
representation and reused across probes, seeds and hyperparameter
candidates, so the reported feature cost is incurred once. Only
`smiles_tfidf_4096` depends on a fitted artifact — a vocabulary and inverse
document frequency weights frozen on an external ChEMBL 37 corpus
[CITATION: ChEMBL] before any endpoint evaluation and without access to
benchmark labels; the other six are stateless functions of molecular
structure. Dimension is reported as a structural property of each encoding
and is not a measure of information content. Representations are ordered by
their Track A2 nonlinear mean rank, the ordering used in every table and
figure.

**Table 2. Track A1 cross-endpoint performance summary.** Track A1 is the
primary, TDC-comparable evaluation. For each representation: mean rank
under the linear and nonlinear probes across all 22 endpoints, and
separately within the 13 classification and 9 regression endpoints;
together with win count (endpoints at which the representation attained
rank 1) and top-3 frequency out of 22. Ranks are computed within each
endpoint on its primary metric after applying the metric's direction, then
averaged over the five train/validation realisations and across endpoints;
**a lower mean rank indicates stronger cross-endpoint performance.** Raw
AUROC and MAE values are never averaged across endpoints. Classification
and regression summaries are computed over disjoint endpoint sets and are
not combined.

**Table 3. Track A1 versus Track A2 robustness summary.** For each
representation and probe: cross-endpoint mean rank and position under
Track A1 and Track A2, the rank displacement between them, the
corresponding value restricted to the 19 genuinely repartitioned endpoints,
and mean ranks within the classification and regression endpoint sets.
**Track A1** consumes the official partition unmodified; **Track A2**
independently repartitions the same molecules by Bemis–Murcko scaffold
[CITATION: Bemis–Murcko scaffold] and applies stricter curation. The two
tracks are analysed under their own sampling interpretations and their rows
are never pooled. A robustness verdict of *reproduced* denotes a
displacement within the pre-registered tolerance of 0.5 rank positions;
*weakened* and *strengthened* denote displacement beyond that tolerance
away from and toward rank 1 respectively. Because Track A2 varies
partitioning and curation together, these verdicts describe persistence
under a combined intervention and do not attribute change to either factor
alone.

**Table 4. Pairwise contrasts of principal interest.** A subset of the
pairwise comparisons; the complete sets for both tracks are given in
Supplementary Tables S1 and S2. Comparisons were performed only within
probe × task families whose Friedman omnibus test [CITATION: Friedman test]
rejected at α = 0.05, using **paired Wilcoxon signed-rank tests over
endpoint-level scores** [CITATION: Wilcoxon signed-rank test] with **Holm
correction applied within each family** [CITATION: Holm correction]. **The
endpoint is the unit of statistical inference throughout; the five
partition realisations within an endpoint are not treated as independent
observations.** For each contrast: probe, task family, the two
representations, the Holm-adjusted p-value, the matched-pairs rank-biserial
correlation as an effect size [CITATION: rank-biserial effect size], the
representation
ranking better, and whether the contrast was also significant after Holm
correction under Track A1. Values shown are from Track A2. Effect sizes are
reported for every contrast; significance is not reported without
magnitude and direction.

**Table 5. Computational cost by representation.** For each
representation: output dimension; total feature-generation time across all
22 endpoints; total model time under the linear and nonlinear probes,
comprising hyperparameter selection over four candidates plus the final
fit; the representation's share of total **nonlinear-probe** model time;
its nonlinear cost relative to the least expensive representation; the
equivalent Track A2 nonlinear model time; and its Track A2 nonlinear mean
rank. All times are wall-clock seconds summed over 22 endpoints and five
partition realisations. **Shares are computed on a nonlinear-only
denominator**, that is, as a fraction of nonlinear-probe model time rather
than of all model time; the two are not interchangeable and are not mixed
within or between tables. All measurements were collected on a single
execution host with a fixed number of concurrent worker processes and a
fixed hyperparameter grid, and are intended for relative comparison within
this study rather than as portable hardware-independent figures. Cost is
reported as an axis separate from predictive rank; no composite measure
combining them is defined.

**Table 6. Stability of representation ordering across partitions.** For
each of the 22 endpoints: Kendall's coefficient of concordance (W)
[CITATION: Kendall's W] computed across the five Track A2 partitions under
the linear and the nonlinear probe, the minimum of the two, and the
resulting stability flag. **W = 1 indicates identical orderings across all
five partitions.** An endpoint is flagged LOW when its weaker probe falls
below the pre-registered threshold of W = 0.35; the minimum across probes
is used so that a stable ordering under one probe cannot mask an unstable
one under the other. LOW endpoints are excluded from endpoint-specific
interpretation only and **remain in every cross-endpoint analysis**.
`vdss_lombardo` is flagged BORDERLINE: it falls marginally below the
threshold but was not pre-registered and was retained rather than added to
the exclusion set after the fact.

**Table 7. All-endpoint and repartitioned-subset summaries side by side.**
Cross-endpoint mean rank, win count and top-3 frequency for each
representation under each probe, computed over all 22 Track A2 endpoints
and separately over the 19 endpoints that met the pre-registered
repartitioning criterion (mean pairwise Jaccard similarity between the five
test molecule sets at most 0.50). The 19-endpoint set is a sensitivity
subset reported alongside the full set, **not a replacement for it**; no
endpoint was removed from the benchmark on this basis. The three endpoints
outside the subset are those for which the frozen scaffold splitter, which
orders scaffold groups by size and permutes only within groups of equal
size, produced test partitions that changed little across seeds.

---

## Supplementary table captions

**Supplementary Table S1. Complete pairwise contrasts, Track A2.** All 126
pairwise comparisons between representations under both probes and all task
families, with Wilcoxon signed-rank statistic, raw p-value, Holm-adjusted
p-value, significance decision at α = 0.05, matched-pairs rank-biserial
effect size, median difference, and the number of endpoints entering each
comparison. Endpoint-level inference throughout.

**Supplementary Table S2. Complete pairwise contrasts, Track A1.** As
Supplementary Table S1, for the 105 pairwise comparisons conducted under
Track A1.

**Supplementary Table S3. Reproduction of Track A1 contrasts under Track
A2.** For each of the eleven contrasts significant after Holm correction in
Track A1: the Track A1 and Track A2 Holm-adjusted p-values and effect
sizes, whether the contrast remained significant under Track A2, and
whether the direction of effect was preserved. A contrast that did not
remain significant is reported as such; failure to reject is not evidence
that the representations concerned are equivalent.

**Supplementary Table S4. Endpoint-level ranks for both tracks.** Rank of
every representation within every endpoint, by probe, for Track A1 and
Track A2. Ranks are direction-aware and averaged over the five partition
realisations, with ties receiving the average of the positions they span.

**Supplementary Table S5. ChEMBL 37 corpus exposure by endpoint.** For each
endpoint: the number of benchmark molecules, the number also present in the
frozen ChEMBL 37 corpus used to fit the SMILES TF-IDF vocabulary
[CITATION: ChEMBL], the resulting overlap fraction, and the Track A1 and
Track A2 endpoint ranks of the TF-IDF representation under each probe. The
vocabulary and inverse document frequency weights were fitted before any
endpoint evaluation and without access to benchmark labels; this overlap is
therefore **external unsupervised corpus exposure**, reported for
transparency. **No analysis in this study tests exposure as a factor
influencing representation performance**, the frozen analysis plan contains
no such test, and the observation is confounded with task family. No causal
relationship is claimed in either direction.

**Supplementary Table S6. Per-partition stability detail.** For every
endpoint, probe and representation: the mean, standard deviation, minimum,
maximum and range of that representation's rank across the five Track A2
partitions. Reported for description only; these per-partition values are
never used as independent observations in any statistical test.

**Supplementary Table S7. Curation effects and partition distinctness by
endpoint.** For each endpoint: the number of official records, the number
of usable molecules after MolFusion curation, records removed as
unparseable, duplicate canonical structures collapsed, molecules carrying
conflicting labels, and the resulting count dropped; together with the mean
and maximum pairwise Jaccard similarity between the five Track A2 test
molecule sets and whether the endpoint met the repartitioning criterion.
Curation was applied in Track A2 only; Track A1 consumes the official
records exactly as shipped, so that comparability with the published
benchmark is preserved.

**Supplementary Table S8. Friedman omnibus results for both tracks.** Test
statistic, degrees of freedom, p-value and rejection decision at α = 0.05
for every probe × task family in Track A1 and Track A2, computed over
within-endpoint ranks with endpoints as blocks
[CITATION: Friedman test]. Pairwise comparisons were conducted only within
families whose omnibus test rejected.

**Supplementary Table S9. Execution provenance audit.** Per-track shard
counts by recorded commit identifier, distinguishing what each result shard
literally recorded from the execution provenance reconstructed from
run-level metadata, content-derived scientific identities, immutable
checksums and source-code history. Some shards lack an embedded commit
identifier owing to a logging defect in the benchmark runner as it existed
at execution time; the numerical results and their scientific identities
were unaffected, no value was reconstructed into a shard, and the provenance
mechanism was hardened only after these runs completed.
