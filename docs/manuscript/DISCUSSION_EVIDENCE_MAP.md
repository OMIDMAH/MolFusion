# Discussion / Limitations / Conclusion evidence map (Phase 6C.4)

Paragraph-level mapping for [`DISCUSSION_DRAFT.md`](DISCUSSION_DRAFT.md).
Every interpretive paragraph maps to at least one registered claim;
paragraphs marked `DESCRIPTIVE` carry no interpretation. Paragraph IDs are
tooling identifiers and do not appear in manuscript prose.

Interpretation types:

| Type | Meaning |
| --- | --- |
| `EVIDENCE` | restates a registered finding |
| `INFERENCE` | reasoning that follows from registered findings |
| `HYPOTHESIS` | explicitly marked as a candidate explanation, not a finding |
| `METHODOLOGICAL` | implication for how such studies should be run |
| `LIMITATION` | bound on interpretation |

Evidence identity: `5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`.

---

## 4.1 Representation utility is probe-dependent

| ID | Claim | Source | Type | Allowed inference | Prohibited inference | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| D4.1-P1 | C1, C3, C10 | R3.2-P1/P2, R3.3-P1/P5 | EVIDENCE | ranking changed with probe | "X is universally best" | — |
| D4.1-P2 | **C3** | R3.2-P4, R3.5-P2 | INFERENCE | differences existed; leadership unresolved | **"the linear probe found no differences"** | — |
| D4.1-P3 | C10 | R3.3-P5 | HYPOTHESIS | structure may be accessible only after nonlinear modelling | "nonlinear probes recover all information"; "linear probes underestimate quality" | — |
| D4.1-P4 | C10, C3 | R3.10-P2 | METHODOLOGICAL | a ranking without its probe is underdetermined | prescription beyond the two probes tested | — |

## 4.2 Compact physicochemical descriptors under nonlinear probing

| ID | Claim | Source | Type | Allowed inference | Prohibited inference | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| D4.2-P1 | **C1**, C2 | R3.3-P1, R3.5-P4 | EVIDENCE | ranked above the other six under this probe | "best molecular representation" | — |
| D4.2-P2 | **C1** | R3.3-P1 | INFERENCE | predictive accessibility under specified probes | **"contains more molecular information"** | — |
| D4.2-P3 | C1 | R3.3-P1 | **HYPOTHESIS** | candidate explanations, explicitly marked | presenting descriptor mechanisms as tested | `[CITATION: molecular descriptors]`, `[CITATION: molecular fingerprints]` |
| D4.2-P4 | C1, **C6** | R3.7-P2/P4 | EVIDENCE | strongest ranking without scaling size or fitting cost | "most efficient representation" | — |
| D4.2-P5 | C1 (restriction) | R3.3-P4 | LIMITATION | cross-endpoint only | **regression-specific superiority** | — |

## 4.3 Robustness across official and repartitioned evaluation

| ID | Claim | Source | Type | Allowed inference | Prohibited inference | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| D4.3-P1 | C2 | R3.4-P1 | EVIDENCE | A1 primary, A2 supplementary | **"external validation"**, "independent cohort" | — |
| D4.3-P2 | **C2** | R3.4-P2, R3.2-P2 | EVIDENCE | nonlinear finding persisted; linear did not | "confirmed"; "proved"; "A2 validated A1" | — |
| D4.3-P3 | C2, **C8** | R3.4-P4/P5 | LIMITATION | persistence under a combined intervention | attributing rank changes to repartitioning alone | — |
| D4.3-P4 | **C5** | R3.5-P2/P3 | EVIDENCE | substantial but incomplete reproduction | "full replication"; "differences disappeared"; "became equivalent" | — |
| D4.3-P5 | **C4** | R3.8-P1/P2/P3 | EVIDENCE | TF-IDF regression advantage less stable; profile mixed | "TF-IDF failed"; "does not generalize"; "overfits ChEMBL" | — |
| D4.3-P6 | **C11** | R3.4-P3 | EVIDENCE | bottom ordering more stable than top | "the bottom two were the same throughout" | — |
| D4.3-P7 | **C7** | R3.6-P2 | HYPOTHESIS | compatible with size sensitivity, untested; counterexample noted | causal sample-size claim | — |

## 4.4 Representation complexity and computational cost

| ID | Claim | Source | Type | Allowed inference | Prohibited inference | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| D4.4-P1 | C1, C11 | R3.10-P3 | EVIDENCE | dimensionality did not order the ranking | "smaller representations are generally better" | — |
| D4.4-P2 | **C6** | R3.7-P2 | EVIDENCE | 35.4% / 30.7%, nonlinear-only denominator | pairing 35.4% with 29.8% | — |
| D4.4-P3 | **C6** | R3.7-P3 | LIMITATION | cost is multi-component; profiles differ | "physchem is cheaper in every respect"; "efficiency score" | — |
| D4.4-P4 | C6 | R3.7-P1 | LIMITATION | single-host, relative only | hardware-independent cost | — |

## 4.5 Implications for molecular representation benchmarking

| ID | Claim | Source | Type | Allowed inference | Prohibited inference | Citation |
| --- | --- | --- | --- | --- | --- | --- |
| D4.5-P1 | DESCRIPTIVE | — | METHODOLOGICAL | scope statement | — | — |
| D4.5-P2 | C10, C3 | R3.10-P2 | METHODOLOGICAL | specify the probe | universal prescription | `[CITATION: representation learning]` |
| D4.5-P3 | C2, C3 | R3.4-P2 | METHODOLOGICAL | comparability and robustness are different questions | "A2 is validation" | — |
| D4.5-P4 | DESCRIPTIVE | Methods §2.9 | METHODOLOGICAL | heterogeneous metrics not averaged | — | — |
| D4.5-P5 | C5 | R3.5-P1/P2 | METHODOLOGICAL | report effect sizes and correction | — | — |
| D4.5-P6 | C6 | R3.7-P1 | METHODOLOGICAL | report cost separately | composite score | — |
| D4.5-P7 | C7 | R3.6-P3 | METHODOLOGICAL | pre-registration constrained conclusions | — | — |
| D4.5-P8 | DESCRIPTIVE | — | LIMITATION | scope bound | — | — |

---

## 5. Limitations

| ID | Claim / basis | Source | Type | Notes |
| --- | --- | --- | --- | --- |
| L5-P1 | C1, C3, C10 | R3.2, R3.3 | LIMITATION | probe scope: two families only |
| L5-P2 | methodological (Methods §2.1) | Methods §2.1 | LIMITATION | representation scope; SELFIES exclusion as scope, not failure |
| L5-P3 | methodological | Methods §2.3 | LIMITATION | ADMET domain scope |
| L5-P4 | C1 (restriction) | R3.3-P4, R3.8-P3 | LIMITATION | **n = 9; A1 nonlinear regression Friedman p = 0.079; no regression-only A2 contrast survived Holm** |
| L5-P5 | **C2**, C8 | R3.4-P4 | LIMITATION | A2 confound; a third track would be needed |
| L5-P6 | **C7** | R3.6-P2/P3 | LIMITATION | six LOW endpoints retained; `vdss_lombardo` BORDERLINE, not excluded |
| L5-P7 | **C9** | R3.9-P1/P2 | LIMITATION | exposure described, non-causal |
| L5-P8 | **C9** | R3.9-P3 | LIMITATION | **explicitly exploratory, untested, confounded; no causal claim in either direction** |
| L5-P9 | **C4** | R3.8-P2/P3 | LIMITATION | TF-IDF regression weakening, rank-level only |
| L5-P10 | C6 | R3.7-P1 | LIMITATION | single-host timing |
| L5-P11 | methodological (Methods §2.12) | Methods §2.12 | LIMITATION | provenance defect; hardened mechanism did **not** produce A1/A2 |

## 6. Conclusion

| ID | Claims | Source | Type | Notes |
| --- | --- | --- | --- | --- |
| C6-P1 | **C1, C2, C3**, C6, C10 | R3.2, R3.3, R3.4, R3.7 | EVIDENCE | all conclusion-permitted in the registry |
| C6-P2 | C1, C3, C6, C10 | R3.10 | METHODOLOGICAL | framework framing; report probe jointly; cost and robustness as distinct axes |

**C9 does not appear in the Conclusion**, in accordance with its registry
permission (`allowed_in_conclusion = False`). It appears in Limitations only
(L5-P7, L5-P8), where it is explicitly labelled exploratory, untested and
non-causal, and it appears nowhere in the Discussion as an explanation for
TF-IDF behaviour.

---

## Claim coverage

| Claim | Discussion | Limitations | Conclusion | Registry `allowed_in_conclusion` | Consistent |
| --- | --- | --- | --- | --- | --- |
| C1 | 4.1, 4.2, 4.4 | L5-P1, L5-P4 | yes | True | ✓ |
| C2 | 4.2, 4.3, 4.5 | L5-P5 | yes | True | ✓ |
| C3 | 4.1, 4.5 | L5-P1 | yes | True | ✓ |
| C4 | 4.3 | L5-P9 | no | True (not needed) | ✓ |
| C5 | 4.3, 4.5 | — | no | True (not needed) | ✓ |
| C6 | 4.2, 4.4, 4.5 | L5-P10 | yes | True | ✓ |
| C7 | 4.3, 4.5 | L5-P6 | no | True (not needed) | ✓ |
| C8 | 4.3 | L5-P5 | no | True (not needed) | ✓ |
| **C9** | **absent** | L5-P7, L5-P8 | **absent** | **False** | ✓ |
| C10 | 4.1, 4.5 | L5-P1 | yes | True | ✓ |
| C11 | 4.3, 4.4 | — | no | True (not needed) | ✓ |

All eleven registered claims are accounted for. No claim outside C1–C11 is
introduced, and no new hypothesis, subgroup or mechanism is presented as
fact.

## Numerical provenance

Discussion introduces no new numbers. Every figure that appears is already
in `RESULTS_DRAFT.md` or the frozen publication package:

| Value | Appears in Results | Source |
| --- | --- | --- |
| 217 dimensions | §3.1, §3.3 | `table1_representation_characteristics.csv` |
| 35.4% (A1), 30.7% (A2) | §3.7 | `table5_computational_cost.csv`; A2 `timings.json`, nonlinear-only denominator |
| p = 0.079 | §3.3 | `friedman_a1.csv`, nonlinear/regression row |
| nine regression endpoints | §3.8 | `representation_ranks.csv`, `subset = regression` |
| eleven / nine contrasts | §3.5 | `a1_vs_a2_contrasts.csv` |
| 22 endpoints, seven representations, two probes | §3.1 | benchmark manifest |

No new ratio, percentage or subgroup count was computed. Bootstrap
confidence intervals are **not** restated in Discussion or Conclusion, so
no opportunity arises to present interval separation as inference.

### Compute-share denominator

Discussion quotes **35.4% (A1)** and **30.7% (A2)**, both on the
nonlinear-only denominator, so the two are comparable. The all-model
figures (33.8% and 29.8%) are recorded in
`RESULTS_EVIDENCE_MAP.md` and are deliberately not used here; pairing 35.4%
with 29.8% is blocked by test.

## Confidence-interval verification (Phase 6C.4 pre-drafting check)

The frozen `bootstrap_mean_rank.csv` for the nonlinear probe and
`rdkit_physchem_descriptors` gives:

| Track | Raw | Reported (2 dp) |
| --- | --- | --- |
| A1 | [1.454545, 2.409091] | **[1.45, 2.41]** |
| A2 | [1.318182, 2.272727] | **[1.32, 2.27]** |

`RESULTS_DRAFT.md` (line 105) and `RESULTS_EVIDENCE_MAP.md` (lines 38, 142)
both carry these values, and the value `2.47` does not occur anywhere in
`docs/manuscript/`. **No correction to the Results draft was required.** The
`[1.45, 2.47]` variant existed only in the Phase 6C.3 conversational report
and never entered a committed artifact. The values are now pinned by
regression test against the frozen table.
