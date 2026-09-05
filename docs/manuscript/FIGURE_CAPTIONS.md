# Figure captions (Phase 6C.5)

Each caption is self-contained: it defines its abbreviations, names the
track and probe, states the rank direction, and identifies the unit of
inference where a statistic is shown. No caption introduces an
interpretation that is not already in Results.

---

## Figure captions

**Figure 1. Representation rank by endpoint under each predictive probe.**
Rows are the 22 ADMET endpoints; columns are the seven fixed-vector
representations. Each cell gives the rank of that representation within
that endpoint, computed on the endpoint's primary metric after applying the
metric's direction (AUROC higher-is-better for classification, MAE
lower-is-better for regression) and averaged over the five partition
realisations; ties receive the average of the positions they span. **Rank 1
denotes the strongest endpoint-relative performance** and rank 7 the
weakest; darker cells indicate stronger ranks. Ranks are computed
*within* each endpoint, so raw AUROC and MAE values are never compared or
averaged across endpoints. The left panel shows the linear probe
(regularised logistic regression for classification, ridge regression for
regression) and the right panel the nonlinear probe (histogram-based
gradient boosting); both panels use the same rank scale. Values shown are
from Track A2 (MolFusion scaffold repartitioning). Endpoints marked with an
asterisk are the six pre-registered low-stability endpoints, whose
representation ordering did not persist across the five partitions
(Kendall's W below 0.35 on the weaker probe); they are retained in all
cross-endpoint analyses, but no endpoint-specific conclusion is drawn from
these rows.

**Figure 2. Cross-endpoint mean rank with bootstrap confidence intervals.**
Points give the mean rank of each representation across the 22 ADMET
endpoints; horizontal bars give the 95 % confidence interval from 10,000
nonparametric bootstrap resamples in which **the endpoint is the resampling
unit**, matching the unit of statistical inference used throughout. Ranks
are direction-aware and computed within each endpoint before aggregation,
so **a lower mean rank indicates stronger cross-endpoint performance**. The
left panel shows the linear probe and the right panel the nonlinear probe;
both panels share a single rank axis spanning positions 1 to 7, so
differences of equal magnitude are displayed at equal visual size in both.
Values shown are from Track A2. Under the nonlinear probe the interval for
the physicochemical descriptor representation, [1.32, 2.27], lies below the
intervals of all six other representations; the corresponding Track A1
interval is [1.45, 2.41]. These intervals are marginal per-representation
intervals rather than a simultaneous band and are shown to convey
uncertainty in the mean rank; **they are not a test of difference between
representations.** Inferential comparisons are the Friedman omnibus test
followed by Holm-corrected paired Wilcoxon signed-rank tests with
matched-pairs rank-biserial effect sizes, reported in Table 4.

**Figure 3. Change in representation rank between the two evaluation
tracks.** Each line connects a representation's position under Track A1
(left) to its position under Track A2 (right), separately for the linear
probe and the nonlinear probe. Positions are ordered by cross-endpoint mean
rank, with **position 1 denoting the strongest** and position 7 the
weakest; the numeric change is given beside each Track A2 endpoint, and
dashed lines indicate representations whose position was unchanged. **Track
A1** is the primary, TDC-comparable evaluation, which consumes the shipped
official test partition unmodified and varies only the train/validation
split across its five seeds. **Track A2** is a supplementary robustness
evaluation that independently repartitions the same molecules by
Bemis–Murcko scaffold across five seeds and additionally applies stricter
curation, collapsing duplicate canonical structures and removing molecules
carrying conflicting labels. Track A2 is not an external validation: the
underlying molecules are the same, and it varies partitioning and curation
together, so the separate contribution of each cannot be identified from
this comparison.

**Figure 4. Cross-endpoint nonlinear rank against nonlinear model-fitting
cost.** The horizontal axis gives the total wall-clock time spent on
hyperparameter selection and final model fitting under the nonlinear probe,
summed over 22 endpoints and five partition realisations; the vertical axis
gives the cross-endpoint mean rank under the same probe, where **a lower
rank is stronger**. The physicochemical descriptor and SMILES TF-IDF
representations are labelled. **The two axes are independent dimensions of
evidence and are not combined:** no composite efficiency or cost-adjusted
performance measure is defined anywhere in this study, and no timing
measurement influenced any rank, statistical test or model-selection
decision. Timings were collected on a single execution host under a fixed
hyperparameter grid with a fixed number of concurrent worker processes, and
support relative comparison within this study only; they are not portable
hardware-independent benchmarks. No trend line is fitted.

**Supplementary Figure S1. Stability of representation ordering across
partitions, by endpoint.** Bars give Kendall's coefficient of concordance
(W) computed across the five Track A2 partitions for each endpoint, with a
light bar for the linear probe and a dark bar for the nonlinear probe;
**W = 1 indicates that all five partitions produced an identical ordering
of the seven representations, and values near zero indicate that the
ordering was effectively resampled at each partition.** Endpoints are
sorted by their weaker probe value, which is also the value shown at the
right of each row and the quantity used for classification: the minimum is
taken across probes so that a stable ordering under one probe cannot mask
an unstable one under the other. The dashed line marks the pre-registered
threshold of W = 0.35. The six endpoints below it (red labels) are the
pre-registered low-stability endpoints and are not used for
endpoint-specific conclusions; **all six were retained in the benchmark and
in every cross-endpoint analysis.** One further endpoint,
`vdss_lombardo`, falls marginally below the threshold on its weaker probe
(nonlinear W = 0.343) and is reported as BORDERLINE; it was **not**
excluded, because widening a pre-specified exclusion set after observing
results would be post-hoc.
