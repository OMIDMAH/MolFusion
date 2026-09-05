# Cover letter — draft

**Not for submission until the author metadata below is completed.**

---

**To:** The Editors, *Journal of Cheminformatics*

**Re:** Submission of a Research article — *MolFusion: Probe-Dependent
Performance of Molecular Representations Across 22 ADMET Endpoints*

Dear Editors,

We submit for your consideration a Research article describing MolFusion, a
reproducible framework for the systematic comparison of heterogeneous
molecular representations, together with the benchmark it was built to
support.

The manuscript addresses a methodological question in molecular machine
learning: whether conclusions about molecular representations remain stable
when downstream probe capacity, benchmark semantics, scaffold
repartitioning, data curation, statistical inference and computational cost
are controlled. Representation choice is among the earliest decisions in a
molecular modelling pipeline, yet reported comparisons are difficult to
interpret because they differ in splits, curation, tuning budget and — most
consequentially for our argument — in the model family used to evaluate
them. A ranking obtained with one model class may not describe the same
representations under another.

We evaluated seven fixed-vector representations across 22 ADMET endpoints
from the Therapeutics Data Commons under two deliberately different
predictive probes, a regularised linear model and a histogram-based
gradient-boosting model, with an identical hyperparameter budget for every
representation. Evaluation was conducted in two tracks that answer
different questions: a primary track consuming the official benchmark
partition unmodified, preserving comparability with other work, and a
supplementary robustness track that independently repartitions the same
molecules by Bemis–Murcko scaffold under stricter curation. The two tracks
are analysed separately and never pooled.

Our central finding is that representation performance was probe-dependent:
the representation attaining the lowest cross-endpoint mean rank under one
probe was not the representation attaining it under the other. Under the
nonlinear probe, a compact 217-dimensional physicochemical descriptor
representation gave the strongest cross-endpoint ranking, and that ranking
was reproduced under independent scaffold repartitioning. Under the linear
probe, differences among representations were detectable but no single
representation separated clearly from the field, and the highest-ranked
representation differed between tracks. We report computational cost as an
independent axis rather than folded into a performance score.

We would highlight three features that we believe fit the journal's scope
particularly well.

First, the study is a methodological contribution rather than a claim of
improved prediction. We do not present MolFusion as a new ADMET predictor,
and we make no comparison against published ADMET models; the benchmark
measures relative ranking among representations under specified probes.

Second, the work is built around reproducibility as a first-class
requirement. Datasets are frozen by content, split semantics are fixed in
advance, molecular identity is canonical and explicit, every representation
arm receives an identical tuning budget, preprocessing is fitted on
training data only, and every result set carries a content-derived
scientific identity that is invariant to the machine that produced it. The
framework, the benchmark protocol, the analysis code and the complete
result tables are openly available.

Third, we report the limits of our own evidence explicitly. The manuscript
states where a conclusion did not reproduce, where an omnibus test did not
reject, which endpoints cannot support endpoint-specific interpretation,
where two robustness interventions are confounded, and where a logging
defect left part of the historical execution provenance incomplete —
together with the audit that established attribution after the fact. We
took the view that a benchmark paper is more useful when it is candid about
what it cannot show.

We believe the *Journal of Cheminformatics* is the appropriate venue: the
work is squarely cheminformatic, it concerns molecular representation and
benchmark methodology rather than a single application, and the journal's
requirement that published research be reproducible by third parties is the
standard this study was designed around.

The manuscript is original, is not under consideration elsewhere, and all
authors have approved the submission.

**[AUTHOR INPUT REQUIRED]** — the following must be completed before this
letter is sent:

- author list and order
- corresponding author name, affiliation and email
- statement of any competing interests
- funding statement
- suggested and excluded reviewers, if the journal invites them

Yours sincerely,

**[AUTHOR INPUT REQUIRED]**
