# 1. Introduction — draft (Phase 6C.5)

Evidence source: Phase 6B publication package, identity
`5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`.
Written last, against the frozen Abstract, Methods, Results, Discussion and
Conclusion, so that the question it poses is the question the study
answers. It contains no result.

---

# 1. Introduction

Machine learning models for molecular property prediction do not consume
molecules; they consume representations of molecules. Every model
therefore inherits the assumptions of whatever encoding precedes it, and
the choice of representation is one of the earliest and least reversible
design decisions in a molecular modelling pipeline
[CITATION: representation learning].

The representations in common use encode different things and embed
different inductive biases. Circular fingerprints enumerate the
substructural environments around each atom out to a fixed radius
[CITATION: Morgan fingerprint; ECFP]. Substructure key fingerprints record
the presence or absence of a predefined list of chemical patterns
[CITATION: MACCS keys], and related substructure fingerprints enumerate
paths and features over the molecular graph [CITATION: Avalon].
Physicochemical descriptors instead compute continuous properties — size,
lipophilicity, polarity, hydrogen-bonding capacity, topological indices —
that summarise a molecule as a vector of interpretable quantities
[CITATION: molecular descriptors]. Fragment-count descriptors count
occurrences of defined functional groups. Reduced-graph representations
abstract a molecule to pharmacophoric node types and the distances between
them, deliberately discarding substructural detail to emphasise
scaffold-level similarity [CITATION: ErG]. Representations derived from
the SMILES line notation [CITATION: SMILES] treat the molecule as a string
and encode statistics over its tokens; string encodings such as SELFIES
extend this idea with syntactic robustness guarantees
[CITATION: SELFIES]. A further class of learned representations —
graph neural networks, molecular language models and three-dimensional
representation learning — derives the encoding from data rather than
specifying it in advance [CITATION: representation learning].

These encodings are not interchangeable, and which one is preferable is an
empirical question. It is also a question that has been difficult to
answer cleanly, for reasons that are largely methodological rather than
chemical.

The first difficulty is that reported representation performance depends
on much more than the representation. It depends on the capacity of the
downstream model, on how the data were split into training and evaluation
sets, on how molecules were curated and de-duplicated, on how much
hyperparameter search each arm received, and on which metric was chosen for
a heterogeneous set of endpoints [CITATION: molecular fingerprints]. Two
studies can reach opposite conclusions about the same pair of
representations without either being wrong, because they measured
different things. Benchmark efforts that fix datasets, metrics and splits
were introduced precisely to reduce this variance
[CITATION: molecular benchmarking][CITATION: TDC ADMET],
but a benchmark constrains only what it specifies, and representation
comparisons frequently vary the model family alongside the encoding.

The second difficulty follows from the first and is the specific concern of
this study. A representation makes information available to a model in a
particular functional form, and whether a model can use that information
depends on the function class it can express. A regularised linear model
measures the predictive structure that is accessible through a linear
decision function of the representation. A model that can express
interactions and thresholds may extract structure that no linear function
of the same vector can reach. If a comparison is conducted with a single
model family, the resulting ranking confounds two things — how much a
representation encodes, and how accessible that encoding is to that
particular model — and the conclusion may not transfer to a different
family. This is not a claim that one probe is more informative than the
other; it is an argument that a representation ranking is incomplete
without the model that produced it.

The third difficulty is reproducibility in the narrow, mechanical sense.
Representation comparisons involve many small decisions — which molecules
are excluded as unparseable, whether duplicate structures are collapsed,
whether conflicting labels are removed, whether a scaler is fitted before
or after the split, how ties are broken in a ranking — and each can move a
result without appearing in a methods section. When such decisions are
made after results are visible, a benchmark becomes difficult to interpret
even when every individual step is defensible. Controls that address this
are well understood in isolation: freezing datasets by content, fixing
split semantics in advance, defining a canonical molecular identity, giving
every arm an identical tuning budget, fitting preprocessing on training
data only, correcting for multiple comparisons, reporting effect sizes
alongside significance, and testing whether a conclusion survives a
different partitioning of the same data
[CITATION: Friedman test]. Applying them together, and recording enough
provenance that a reported number can be traced to the exact inputs and
code that produced it, is less common.

ADMET property prediction is a useful setting in which to examine these
issues. Absorption, distribution, metabolism, excretion and toxicity
endpoints are central to early-stage compound assessment; they span both
classification and regression tasks; their molecular determinants differ
substantially, from bulk physicochemical behaviour to specific metabolic
recognition; and the available datasets vary by more than an order of
magnitude in size. A representation that suits one endpoint need not suit
another, which makes ADMET a demanding test of whether any general
statement about representations can be made at all. Public collections
such as the Therapeutics Data Commons provide standardised ADMET datasets
and partitions [CITATION: TDC; TDC ADMET], which makes comparability with
other work possible while leaving the representation-versus-probe question
open.

The question this study addresses is therefore:

> How does the apparent utility of heterogeneous molecular representations
> change when they are evaluated under controlled linear and nonlinear
> predictive probes using a frozen, reproducible ADMET benchmark?

Here we present MolFusion, a reproducible framework for the systematic
comparison of heterogeneous molecular representations. MolFusion is not a
predictive model and is not offered as an improved ADMET predictor; it is
the infrastructure under which different encodings can be compared on equal
terms, together with the benchmark protocol that makes such a comparison
interpretable. Representations are supplied by versioned components
registered in a central registry, molecular identity is canonical and
explicit, feature matrices are cached under content-derived keys, and every
result carries an identity derived from its own scientific content rather
than from its filename or the time it was produced.

Using this framework, we compare seven fixed-vector molecular
representations — a circular fingerprint, two substructure fingerprints,
physicochemical descriptors, fragment counts, a reduced-graph encoding and
a SMILES token n-gram TF-IDF — across 22 ADMET endpoints comprising 13
classification and 9 regression tasks. Each representation is evaluated
through two deliberately different predictive probes, a regularised linear
model and a histogram-based gradient-boosting model
[CITATION: scikit-learn], under an identical hyperparameter budget, so that
any difference between representations is attributable to the encoding and
its required preprocessing rather than to model capacity or tuning effort.
Evaluation is conducted in two tracks that answer different questions: a
primary track that consumes the official benchmark partition unmodified,
preserving comparability with other work, and a supplementary robustness
track that independently repartitions the same molecules by Bemis–Murcko
scaffold [CITATION: Bemis–Murcko scaffold] under stricter curation. The two
tracks are analysed separately and never pooled. Statistical evidence is
reported at the level of the endpoint, with multiple-comparison correction
and effect sizes, and computational cost is measured and reported as an
independent dimension rather than folded into a performance score.

The contributions of this work are fourfold: a reproducible,
frozen-protocol framework for representation comparison in which every
reported number carries a verifiable identity; a systematic comparison of
seven heterogeneous fixed-vector representations across 22 ADMET endpoints
under matched probes; a dual-track design that separates official
benchmark comparability from robustness under independent repartitioning;
and a probe-dependent analysis in which statistical evidence and
computational cost are quantified as distinct axes. Learned molecular
representations are discussed here as context but were not part of the
benchmark and are not evaluated in this study.
