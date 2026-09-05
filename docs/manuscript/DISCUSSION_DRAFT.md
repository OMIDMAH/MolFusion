# 4. Discussion · 5. Limitations · 6. Conclusion — draft (Phase 6C.4)

Evidence source: Phase 6B publication package, identity
`5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`
(verified before and after drafting). Every interpretive paragraph maps to
a registered claim; the mapping is in
[`DISCUSSION_EVIDENCE_MAP.md`](DISCUSSION_EVIDENCE_MAP.md).

No new evidence is generated here. Numbers appear only where they already
appear in `RESULTS_DRAFT.md` or the frozen publication package.

---

# 4. Discussion

## 4.1 Representation utility is probe-dependent

The clearest result of this benchmark is not that one representation is
preferable, but that the answer to "which representation performs best"
changed with the model consuming it. Under the nonlinear probe, the
217-dimensional physicochemical descriptor representation attained the
lowest cross-endpoint mean rank in both evaluation tracks. Under the linear
probe, no representation separated clearly from the field, and the
top-ranked representation differed between tracks.

This contrast requires careful statement, because two different findings
are easily conflated. The Friedman omnibus test **rejected** in the linear
families of both tracks: differences among the seven representations were
detectable under the linear probe. What the evidence does not support is
the identification of a leading linear representation. The corrected
pairwise contrasts that reached significance under the linear probe
separated the lowest-ranked representations from the rest rather than
distinguishing among the leading group, whose bootstrap intervals
overlapped substantially. The linear result is therefore *unresolved
leadership*, not *absence of difference*, and these should not be reported
interchangeably.

A representation and a probe jointly determine measured performance. A
fixed-length encoding makes certain relationships available to a model in a
particular functional form; whether a given model can exploit them depends
on the function class it can express. One possible interpretation of the
divergence observed here is that some structure encoded in these
representations becomes accessible only once nonlinear interactions and
thresholding are modelled, and is not recoverable by a linear decision
function under the tuning budget applied. This is an interpretation of the
observed rank pattern, not a demonstration: the study measured accessible
predictive performance under two specified probes and did not decompose
what either probe failed to extract.

The practical consequence is that a representation ranking reported without
its probe is underdetermined. A comparison conducted with a single model
family may support a conclusion that a different family would reverse, as
occurred here between the linear and nonlinear probes in both tracks.

## 4.2 Compact physicochemical descriptors under nonlinear probing

Under the frozen nonlinear probe, the compact physicochemical descriptor
representation ranked above the other six fixed-vector representations
across the cross-endpoint analysis, in both the official and the
repartitioned evaluation. It attained the best rank at more endpoints than
any other representation and placed in the top three at the large majority
of them, and the corrected pairwise contrasts in which it was the superior
arm carried large matched-pairs rank-biserial effect sizes.

The correct framing of this result is **predictive accessibility**, not
information content. The benchmark measured how much predictive performance
each encoding made available to two specified probes under a fixed tuning
budget. It did not measure how much chemical information each encoding
contains, and a representation that performed less well here may encode
information that these probes could not reach. Nothing in this design
supports a claim that physicochemical descriptors carry more molecular
information than circular fingerprints, substructure keys, reduced-graph
features or SMILES n-gram statistics.

Why a compact continuous descriptor set should be well matched to a
histogram-based gradient-boosting probe is not something this study tested,
and any explanation offered here is a hypothesis rather than a finding. One
plausible interpretation is that these descriptors present properties such
as molecular size, lipophilicity, polarity and hydrogen-bonding capacity
[CITATION: molecular descriptors] as directly interpretable continuous
axes, which a tree-based model can threshold individually, whereas
substructure-presence encodings [CITATION: molecular fingerprints] express
related information through combinations of sparse indicator bits that the
same model must reconstruct. This pattern is also consistent with the fact
that many ADMET endpoints are physicochemically governed properties. These
considerations are offered as candidate explanations; no
descriptor-importance or ablation analysis was performed, and none should
be inferred.

A practical aspect of this result is that the representation concerned is
also among the smaller and, under the nonlinear probe, among the less
expensive to fit. The strongest cross-endpoint nonlinear ranking was
therefore not obtained by scaling representation size or model-fitting
effort. That relationship is developed in Section 4.4, where cost is
treated as an axis in its own right rather than as a property of the
ranking.

One boundary on this result must be carried explicitly. The cross-endpoint
nonlinear finding should not be interpreted as evidence of a
regression-specific superiority: the Track A1 nonlinear omnibus comparison
restricted to the nine regression endpoints did not reject. The nonlinear
result reported here is a cross-endpoint result, and it is qualified again
in Section 5.

## 4.3 Robustness across official and repartitioned evaluation

Track A1 is the primary, TDC-comparable evaluation, and Track A2 is a
supplementary robustness evaluation applying independent MolFusion scaffold
repartitioning and stricter curation to the same underlying benchmark
datasets. Track A2 is not an external validation: the molecules are the
same, and no independent cohort was introduced.

Within that scope, the principal nonlinear finding persisted. The leading
representation under the nonlinear probe was unchanged between tracks, its
mean rank moved within the pre-registered tolerance, and restricting Track
A2 to the genuinely repartitioned endpoints left the leading order intact.
By contrast, the leading position under the linear probe changed between
tracks and did not stabilise in the repartitioned subset. Robustness under
repartitioning therefore differentiated between the two probe-level
findings rather than uniformly supporting both, which is what makes it
informative.

The robustness analysis demonstrates persistence under the combined change
in partitioning and curation; it does not identify which component caused
individual rank changes. Track A2 alters both simultaneously, and the
design cannot separate them. A sensitivity analysis excluding the two
endpoints most affected by curation left the leading nonlinear
representation unchanged, which bounds the contribution of curation without
isolating it.

At the level of individual statistical contrasts, reproduction was
substantial but incomplete: of the eleven contrasts significant after Holm
correction in Track A1, nine remained significant in Track A2, and effect
direction was preserved in all eleven. The two contrasts that did not
remain significant should not be described as differences that disappeared
or as evidence that the representations concerned are equivalent. Failure
to reject is not evidence of equivalence, and both contrasts retained their
original direction under an independent partitioning with its own noise.

The same asymmetry appears within a single representation. The SMILES
TF-IDF representation held the lowest regression mean rank under the linear
probe in Track A1, and that advantage was less stable under Track A2, where
both its regression mean rank and its win count declined. This is a
rank-level observation over nine endpoints, and no regression-only pairwise
contrast survived Holm correction in Track A2, so it should be read as a
weakening of an advantage observed under one particular partition rather
than as poor regression performance. The same representation improved its
nonlinear-probe position between tracks, which is why its overall profile
is better described as mixed than as directional.

The ordering of the lowest-ranked representations was more stable across
tracks than the ordering of the leading ones. The fragment-count
representation occupied a bottom-two position in every probe and track
combination examined, and the reduced-graph representation in all but one.
That asymmetry — stable at the bottom, unstable at the top under the linear
probe — is worth noting, since benchmark conclusions are usually drawn from
the top of a ranking, which is where this study found the least stability.

Stability also varied markedly across endpoints. Six endpoints showed
representation orderings that did not persist across the five repartitions
and were pre-registered as unsuitable for endpoint-specific interpretation,
while remaining in all cross-endpoint analyses. These six are among the
smaller endpoints in the benchmark, and this pattern is compatible with
greater rank sensitivity in smaller datasets; the study did not formally
test sample size as a determinant of stability, and the relationship is not
simple, since the two smallest endpoints in the benchmark were not among
those flagged. No causal account of endpoint stability is offered.

## 4.4 Representation complexity and computational cost

Representation dimensionality did not order the representations by
cross-endpoint rank in either direction. The strongest nonlinear ranking
was obtained by a 217-dimensional representation, while the
4096-dimensional and the two 1024-dimensional representations ranked below
it under that probe, and the 85-dimensional representation ranked at the
bottom. Larger encodings were neither systematically better nor
systematically worse.

Computational cost is reported as an independent axis. Nonlinear
model-fitting cost differed substantially between representations: the
SMILES TF-IDF representation accounted for the largest share of nonlinear
model compute in both tracks — 35.4% in Track A1 and 30.7% in Track A2 on
the same nonlinear-only basis — while the physicochemical representation
accounted for well under half that share. The strongest nonlinear
cross-endpoint ranking was therefore obtained without the largest
representation and without the greatest model-fitting cost.

That statement should not be compressed into a claim of efficiency. No
composite efficiency or cost-adjusted performance score was defined in this
study, and the exchange rate between compute and predictive performance is
application-specific. Nor is the physicochemical representation cheaper in
every respect: it was the most expensive representation to *compute*, while
being among the least expensive to *fit* under the nonlinear probe, and the
TF-IDF representation showed the opposite profile, being inexpensive to
compute from its pre-fitted artifact and the most expensive to fit. Because
feature matrices are computed once per endpoint and reused across probes,
seeds and hyperparameter candidates whereas fitting cost scales with all of
them, the relative importance of these two components depends on the size
of the evaluation. Cost is a multi-component quantity, and reducing it to a
single ordering would misrepresent it.

All timings were collected on a single execution host under a fixed
hyperparameter grid and support relative comparison within this study only.

## 4.5 Implications for molecular representation benchmarking

Several methodological implications follow from the results, and they
concern how such comparisons are conducted rather than which encoding a
practitioner should adopt.

First, a representation benchmark should specify the downstream probe as
part of its result. A ranking obtained under one model family did not
transfer to another here, in either evaluation track, so a single model
family gives an incomplete picture of comparative representation utility
[CITATION: representation learning].

Second, comparability with an official benchmark partition and robustness
under repartitioning answer different questions, and both are useful. The
official partition supports comparison with other work; independent
repartitioning tests whether a conclusion depends on one particular
division of the data. In this study the two led to the same conclusion
under the nonlinear probe and to different leading representations under
the linear probe, which is precisely the situation in which reporting only
one of them would mislead.

Third, heterogeneous endpoint metrics should not be averaged directly.
AUROC and MAE differ in scale and direction, and any cross-endpoint summary
that averages them is not interpretable. Ranking within each endpoint
before aggregation avoids this while preserving the comparison of interest.

Fourth, rankings should be accompanied by effect sizes and by an explicit
multiple-comparison correction. In this benchmark, 126 pairwise comparisons
were available in a single track, and reporting uncorrected results would
have substantially overstated the number of distinguishable pairs.

Fifth, computational cost should be reported separately from predictive
performance rather than folded into a combined score, so that readers with
different compute constraints can weigh the two themselves.

Finally, the pre-registration of analysis decisions had a visible effect on
what could be claimed. The endpoints excluded from endpoint-specific
interpretation, the tolerance used to judge rank reproduction, and the
statistical plan were all fixed before results were observed, and each of
them constrained a conclusion after the fact — including one endpoint that
fell marginally outside a pre-registered set and was retained rather than
added to it.

These implications are drawn from a benchmark of seven fixed-vector
representations on 22 ADMET endpoints under two probes, and their scope is
bounded accordingly.

---

# 5. Limitations

**Probe scope.** Two downstream probe families were evaluated: regularised
linear models and histogram-based gradient-boosting models. The conclusions
apply to these families under the fixed hyperparameter budget described in
Methods. They do not establish how the same representations would order
under neural networks, kernel methods, graph neural networks, transformer
architectures, or heads trained on pretrained molecular foundation models.
Because the central finding of this study is that representation ranking
depends on the probe, this limitation bears directly on the interpretation
of every result reported.

**Representation scope.** Track A comprises seven fixed-vector
representations. No learned or pretrained molecular representation was
benchmarked. A SELFIES sequence representation is implemented within the
framework but was excluded from Track A because it produces a
variable-length categorical sequence and would require an additional
learned encoder, which would confound the representation with the capacity
and training of that encoder. This is a scope decision rather than a
negative result: no SELFIES model was trained or evaluated, and the study
makes no claim about its utility. Consequently this work does not compare
fixed-vector encodings against graph neural network embeddings,
language-model embeddings, learned sequence encoders, or three-dimensional
learned representations.

**Domain scope.** The benchmark covers 22 ADMET endpoints. Conclusions
should not be generalised without new evaluation to binding affinity
prediction, reaction prediction, protein–ligand interaction modelling,
quantum-chemical property prediction, or generative molecular design, which
differ in label semantics, data scale and the structural features that
matter.

**Regression evidence.** The regression family contains nine endpoints, the
smallest of the task families analysed. The Track A1 nonlinear omnibus
comparison restricted to these endpoints did not reject (p = 0.079), and no
regression-only pairwise contrast survived Holm correction in Track A2.
Accordingly, no regression-specific nonlinear superiority claim is
supported by this study, and the nonlinear finding reported in Results and
interpreted in Section 4.2 should be read as a cross-endpoint result.

**Confounded robustness interventions.** Track A2 changes both the
partitioning scheme and the curation policy relative to Track A1. Their
individual contributions cannot be separated within this design; isolating
them would require a third track varying one factor at a time, which was
not undertaken. Track A2 results should therefore be read as evidence of
persistence under a combined intervention.

**Endpoint stability.** Six endpoints — `herg`,
`cyp2c9_substrate_carbonmangels`, `clearance_hepatocyte_az`,
`cyp2d6_substrate_carbonmangels`, `cyp3a4_substrate_carbonmangels` and
`bioavailability_ma` — showed representation orderings that were
insufficiently stable across the five repartitions to support
endpoint-specific interpretation. They were pre-registered as such and
retained in all cross-endpoint analyses; none was removed from the
benchmark. A seventh endpoint, `vdss_lombardo`, falls marginally below the
same threshold on its weaker probe and is reported as BORDERLINE. It was
retained rather than excluded, because widening a pre-specified exclusion
set after observing results would be post-hoc. Cross-endpoint summaries in
this study aggregate over both stable and unstable endpoints.

**External unsupervised corpus exposure.** The SMILES TF-IDF vocabulary and
IDF weights were fitted on a frozen ChEMBL 37 reference corpus before any
endpoint evaluation and without access to benchmark labels. Because that
corpus is a general-purpose compound library, a substantial and
endpoint-dependent proportion of benchmark molecules also occur in it, and
this overlap is reported per endpoint in Results for transparency. It is
described as external unsupervised corpus exposure rather than as leakage
or contamination, because no benchmark label was visible during fitting.

This exposure is a genuine limitation on the interpretation of the TF-IDF
results, and it remains **exploratory and untested**. The frozen analysis
plan contains no test of exposure as a factor influencing representation
performance, and none was performed. The observation is additionally
confounded with task family, since most of the high-exposure endpoints are
regression endpoints. No causal relationship between corpus exposure and
TF-IDF performance is claimed, in either direction, and none can be drawn
from the present evidence. Establishing whether such a relationship exists
would require a pre-registered analysis that this study does not contain.

**Robustness of the TF-IDF regression result.** The regression advantage
observed for the TF-IDF representation under the linear probe in Track A1
was less stable under Track A2, where its regression mean rank and win
count both fell. This is a rank-level observation over nine endpoints and
is not supported by a corrected pairwise contrast; it indicates a weakening
of the advantage originally observed under the official partition rather
than poor regression performance in general.

**Computational timing.** All timings were measured on a single execution
host, with a fixed number of concurrent workers and a fixed hyperparameter
grid. Absolute times are not portable hardware-independent benchmarks, and
only within-study relative comparisons are intended. Costs would change
with different hardware, parallelism, library versions, or tuning budgets.

**Execution provenance.** Some result shards from the completed benchmark
runs lack an embedded Git commit identifier, the consequence of a logging
defect in the benchmark runner as it existed at execution time, in which
each worker resolved the commit independently and recorded a null value on
failure. The numerical results and their content-derived scientific
identities were unaffected. Attribution was verified after the fact using
run-level metadata, content-derived scientific identities, immutable
dataset and result checksums, and the source-code history of the runner.
The provenance mechanism was subsequently hardened; that hardened
mechanism applies to subsequent executions and did not produce the results
reported here. Per-shard counts are given in Supplementary Material.

---

# 6. Conclusion

Across 22 ADMET endpoints, seven fixed-vector molecular representations and
two predictive probes, representation performance was probe-dependent: the
representation attaining the lowest cross-endpoint mean rank under one
probe was not the representation attaining it under the other. Under the
nonlinear probe, a compact 217-dimensional physicochemical descriptor
representation achieved the strongest cross-endpoint ranking, supported by
a Friedman omnibus test, Holm-corrected paired Wilcoxon comparisons and
large rank-biserial effect sizes, and this ranking was reproduced under
independently generated Bemis–Murcko scaffold partitions with stricter
curation. Under the linear probe, differences among representations were
detectable but no single representation separated clearly from the field,
and the highest-ranked representation differed between evaluation tracks.
The strongest nonlinear ranking was obtained without the largest
representation and without the greatest model-fitting cost, although cost
and predictive performance are reported here as separate dimensions rather
than combined.

MolFusion is presented as a reproducible framework for systematic
comparison of heterogeneous molecular representations, not as a new ADMET
predictor. The results indicate that molecular representation benchmarks
should report the representation jointly with the predictive probe, and
should evaluate robustness under repartitioning and computational cost as
distinct dimensions rather than as a single measure of quality.
