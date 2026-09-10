# Claim-to-section map — frozen (Phase 6C.1)

Machine-readable form: [`claim_to_section.csv`](claim_to_section.csv),
validated by `backend/tests/test_manuscript_plan.py`.

Every manuscript statement about representation superiority, robustness,
cost, TF-IDF, stability, or corpus exposure must map to a registered claim
in the Phase 6B registry
(`backend/benchmark_runs/publication/evidence/claim_registry.csv`).

Evidence identity: `5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`.

---

## Map

| Claim | Type | Section | Abstract | Results | Discussion | Conclusion | Figures | Tables |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | PRIMARY | 3.3 | **yes** | yes | yes | yes | 1, 2 | 2, 4 |
| C2 | ROBUSTNESS | 3.4 | **yes** | yes | yes | yes | 3 | 3 |
| C3 | NEGATIVE | 3.2 | **yes** | yes | yes | yes | 1, 2 | 2 |
| C4 | NEGATIVE | 3.8 | no | yes | yes | yes | — | 3, S4 |
| C5 | ROBUSTNESS | 3.5 | no | yes | yes | yes | — | 4, S3 |
| C6 | SECONDARY | 3.7 | no | yes | yes | yes | 4 | 5 |
| C7 | CAVEAT | 3.6 | no | yes | yes | yes | S1 | 6 |
| C8 | CAVEAT | 3.4 | no | yes | yes | yes | — | S7 |
| **C9** | **EXPLORATORY** | 3.9 | **no** | yes | **no** | **no** | — | S5 |
| C10 | SECONDARY | 3.3 | no | yes | yes | yes | 1 | 2, 3 |
| C11 | ROBUSTNESS | 3.4 | no | yes | yes | yes | 3 | 3 |

Only **C1, C2, C3** are abstract-permitted, matching the registry's
`allowed_in_abstract` column exactly.

**C9 is confined to Results §3.9 and the Limitations section.** It may not
appear in the Abstract, the Discussion argument, or the Conclusion. It is
the single most likely source of overclaiming in this manuscript, because
corpus overlap invites causal language that the evidence does not support.

---

## Prohibited-wording audit

The Abstract and contribution statement were checked against every
`prohibited_wording` entry in the registry.

| Prohibited pattern | Present? | Note |
| --- | --- | --- |
| universal superiority (*best representation*, *universally outperform*) | **no** | Abstract says "strongest cross-endpoint mean ranking", scoped to the nonlinear probe |
| *contains more molecular information* | **no** | framed as ranking under a probe, not information content |
| *structural fingerprints* as a collective term | **no** | categories enumerated individually |
| state-of-the-art ADMET prediction | **no** | no absolute performance or literature comparison claimed |
| external validation / *A2 confirms A1* | **no** | Abstract says "reproduced under independently generated … partitions" |
| significance from CI non-overlap | **no** | intervals absent from the Abstract; support attributed to Friedman, Holm-Wilcoxon, rank-biserial |
| *TF-IDF fails* / *is inferior* | **no** | C4 is not in the Abstract at all |
| leakage / contamination / causal exposure | **no** | C9 is not in the Abstract, Discussion or Conclusion |
| *all representations are equivalent under a linear probe* | **no** | Abstract says "no representation separated clearly from the field" |
| *these endpoints were excluded from the benchmark* | **no** | all 22 endpoints retained; C7 not in the Abstract |
| efficiency score / cost-adjusted performance | **no** | cost absent from the Abstract; kept as a separate axis in §3.7 |

**Audit result: PASS** — no prohibited construction appears in the frozen
Abstract or contribution statement.

Two phrasings were changed during drafting as a direct result of this
audit:

- An earlier abstract sentence read "*and this advantage was confirmed
  under independent scaffold partitions*". "Confirmed" edges toward C2's
  prohibited "A2 confirms A1", so it became "**reproduced under
  independently generated Bemis–Murcko scaffold partitions**".
- Bootstrap intervals were removed from the abstract entirely. They were
  accurate and favourable — the leader's interval is clear of every
  competitor's in both tracks — but placing them beside the statistical
  attribution risked reading as inference from non-overlap.

---

## Drafting rule for Phase 6C.2 onward

For each Results subsection:

1. Take the governing claim's `recommended_wording` from the registry as
   the topic sentence.
2. Draft supporting sentences only from the tables named above.
3. Check the finished paragraph against that claim's `prohibited_wording`.
4. Any statement that maps to no claim must either be removed or raised as
   a proposed registry addition — the registry is authoritative and is not
   edited during drafting.
