# Title, contributions, abstract, keywords — frozen (Phase 6C.1)

Evidence source: Phase 6B publication package, identity
`5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`
(verified before drafting).

Every claim below maps to a registered claim ID. Nothing here was drafted
from memory.

---

## 1. Framing

MolFusion is presented as **a reproducible framework for systematic
comparison of heterogeneous molecular representations**, with ADMET
prediction as the evaluation domain.

It is *not* presented as a new state-of-the-art ADMET predictor. Nothing in
the evidence supports that: no absolute performance comparison against
published ADMET models was made, and the benchmark measures relative
ranking among seven representations under two frozen probes.

The scientific story is deliberately narrow:

> Representation utility depends strongly on the downstream probe. Under a
> nonlinear probe, a compact 217-dimensional physicochemical descriptor
> representation achieved the strongest and most reproducible
> cross-endpoint ranking. Under a linear probe, no representation clearly
> separated from the field.

---

## 2. Title candidates

Twelve candidates in three styles. Superlatives (*best*, *state-of-the-art*,
*universal*, *optimal*) are avoided throughout.

### Framework-focused

| # | Title |
| --- | --- |
| T1 | MolFusion: A Reproducible Framework for Systematic Comparison of Heterogeneous Molecular Representations |
| T2 | A Frozen-Protocol Framework for Reproducible Molecular Representation Benchmarking on ADMET Endpoints |
| T3 | MolFusion: Reproducible, Provenance-Tracked Evaluation of Molecular Representations for ADMET Prediction |
| T4 | Systematic Evaluation of Heterogeneous Molecular Representations Under a Frozen Benchmark Protocol |

### Finding-focused

| # | Title |
| --- | --- |
| T5 | Representation Utility Depends on the Downstream Probe: A Systematic ADMET Benchmark |
| T6 | Probe-Dependent Performance of Molecular Representations in ADMET Prediction |
| T7 | Compact Physicochemical Descriptors Rank Highest Under Nonlinear Probing Across 22 ADMET Endpoints |
| T8 | Probe Choice Reorders Molecular Representations: Evidence from 22 ADMET Endpoints |

### Balanced (framework + finding)

| # | Title |
| --- | --- |
| **T9** | **MolFusion: Probe-Dependent Performance of Molecular Representations Across 22 ADMET Endpoints** |
| T10 | Probe-Dependent Performance of Molecular Representations: A Reproducible Benchmark of Seven Fixed-Vector Encodings Across 22 ADMET Endpoints |
| T11 | Benchmarking Seven Molecular Representations Across 22 ADMET Endpoints: Probe Dependence and Robustness Under Scaffold Repartitioning |
| T12 | MolFusion: A Reproducible Benchmark of Molecular Representations Under Official and Independently Repartitioned ADMET Splits |

### Ranking

| Rank | Title | Why |
| --- | --- | --- |
| 1 | **T9** | Names the framework, states the finding, no superlative, fits a title line |
| 2 | T10 | Most precise; better for a methods-oriented venue, but long |
| 3 | T11 | Strong and specific; omits the framework name |
| 4 | T6 | Clean finding-focused; loses the framework contribution |
| 5 | T1 | Accurate framework framing; buries the scientific finding |
| 6 | T12 | Precise on design; "official and independently repartitioned" is jargon-heavy for a title |
| 7 | T5 | Good but vague on scale |
| 8 | T3 | Provenance emphasis is a Methods strength, not a title one |
| 9 | T8 | "Reorders" slightly overstates a partly-overlapping linear result |
| 10 | T7 | Finding-true but reads as a superiority claim out of context |
| 11 | T2 | Framework-only, no domain finding |
| 12 | T4 | Least specific |

**Recommended: T9 — *MolFusion: Probe-Dependent Performance of Molecular
Representations Across 22 ADMET Endpoints***

T10 is the recommended alternative if the venue prefers a fully specified
title over a concise one.

A note on a rejected phrasing: "No Universal Winner" was considered and
dropped. It is accurate, but it leads with a negation and uses a prohibited
superlative even in denial, which invites exactly the framing the evidence
does not support.

---

## 3. Contribution statement

Four contributions. Each maps to registered evidence.

1. **A reproducible, frozen-protocol framework for molecular representation
   comparison.** Content-derived dataset identities, a deterministic
   serialization contract, matrix-level feature caching keyed on molecule
   ordering, atomic result shards with checkpoint/resume, and
   execution-provenance capture — such that every reported number carries a
   verifiable identity. *(Methods §2.1, §2.4, §2.12)*

2. **A systematic comparison of seven heterogeneous fixed-vector
   representations** — circular, substructure and substructure-key
   fingerprints, physicochemical and fragment-count descriptors, a
   reduced-graph encoding, and a SMILES n-gram TF-IDF — across 22 TDC ADMET
   endpoints under two frozen probes and five seeds. *(Results §3.1)*

3. **A dual-track evaluation separating official comparability from
   robustness.** Track A1 consumes the official TDC partition unmodified;
   Track A2 independently repartitions by Bemis–Murcko scaffold under
   stricter curation. The tracks are never pooled, and the endpoint is the
   unit of inference throughout. *(Methods §2.5–2.6; Results §3.4)*
   *(C2, C5)*

4. **A probe-dependent analysis of representation performance**, with
   endpoint-level statistics (Friedman → Holm-corrected paired Wilcoxon →
   rank-biserial effect sizes → endpoint bootstrap) and computational cost
   reported as an independent axis rather than folded into a composite
   score. *(Results §3.2–3.3, §3.5, §3.7)* *(C1, C3, C6, C10)*

**Explicitly not claimed:** SELFIES was implemented in the wider MolFusion
framework but was **not** part of Track A and is not benchmarked here. No
learned or pretrained representation was benchmarked. No comparison against
published ADMET state-of-the-art was performed.

---

## 4. Abstract (220 words)

> Molecular representation choice is a central design decision in
> property prediction, yet comparisons are often confounded by
> inconsistent splits, undocumented curation, and evaluation against a
> single model family. We present MolFusion, a reproducible framework for
> systematic comparison of heterogeneous molecular representations under a
> frozen protocol, in which datasets, splits, hyperparameter budgets and
> statistical procedures are fixed in advance and every result carries a
> content-derived identity. We benchmarked seven fixed-vector
> representations — circular, substructure and substructure-key
> fingerprints, physicochemical and fragment-count descriptors, a
> reduced-graph encoding, and a SMILES n-gram TF-IDF — across 22 ADMET
> endpoints from the Therapeutics Data Commons, using a regularized linear
> probe and a gradient-boosting probe with the endpoint as the unit of
> statistical inference. Under the nonlinear probe, a compact
> 217-dimensional physicochemical descriptor representation achieved the
> strongest cross-endpoint mean ranking (1.91 on the official partition),
> supported by a Friedman omnibus test, Holm-corrected paired Wilcoxon
> comparisons, and large matched-pairs rank-biserial effect sizes. Under
> the linear probe, no representation separated clearly from the field, and
> the highest-ranked representation differed between evaluation tracks. The
> nonlinear ranking was reproduced under independently generated
> Bemis–Murcko scaffold partitions with stricter curation (mean rank 1.77),
> whereas the linear result remained unresolved. These findings indicate
> that representation comparisons are contingent on the downstream model
> family, and that reporting a representation ranking without specifying
> the probe is insufficient.

### Numerical claims and their sources

| Number | Source | Claim |
| --- | --- | --- |
| 22 ADMET endpoints | benchmark manifest; Table 7 | — |
| seven fixed-vector representations | Table 1 | — |
| 217 dimensions | Table 1 | C1 |
| mean rank 1.91 (A1 nonlinear) | `representation_ranks.csv` (A1) | C1 |
| mean rank 1.77 (A2 nonlinear) | `representation_ranks.csv` (A2) | C2 |
| Friedman, Holm-Wilcoxon, rank-biserial | `friedman.csv`, `statistical_tests.csv` | C1 |

No average performance improvement is stated; no incompatible metrics are
averaged; no absolute predictive performance is claimed.

**Bootstrap intervals are deliberately absent from the abstract.** They
belong to Figure 2 as uncertainty visualization. Statistical support is
attributed only to the Friedman omnibus, Holm-corrected Wilcoxon, and
rank-biserial effect sizes — never to interval non-overlap.

**Deliberately excluded from the abstract**, per the registry and the phase
brief: the TF-IDF regression weakening (C4), external corpus exposure (C9),
low-stability endpoint detail (C7), computational cost (C6), and the
historical provenance defect. All appear later in the paper.

---

## 5. Keywords

Eight, ordered from domain to method:

```
molecular representation
ADMET prediction
molecular fingerprints
physicochemical descriptors
scaffold split
benchmarking
reproducibility
cheminformatics
```

`SMILES`, `TF-IDF`, `machine learning` and `drug discovery` were considered
and dropped: the first two over-weight one of seven representations, and
the last two are too generic to aid indexing.
