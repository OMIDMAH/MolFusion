# MolFusion v2 Documentation

- [`reproducibility.md`](reproducibility.md) — artifact infrastructure:
  directory convention, metadata schema, checksum verification,
  `agent_version` vs `artifact_version`; the SMILES normalization and
  tokenization contracts every text-fitted representation depends on;
  the ChEMBL 37 reference corpus builder and its `fit_corpus_sha256`; and
  the Phase 5F-C token n-gram vocabulary study -- corpus identity gate,
  the deterministic hash holdout, DF/TF accounting, and the feature
  ranking rule MolFusion freezes instead of delegating to a vectorizer
- [`tfidf-vocabulary-decision.md`](tfidf-vocabulary-decision.md) — the
  measured Phase 5F-C recommendation for the SMILES TF-IDF vocabulary:
  n-gram range, `min_df`, dimension, DF-vs-TF ranking, and whether
  unigrams need protecting, with the evidence for each. Recommendation
  only — nothing is fitted or frozen until Phase 5F-D
- [`tfidf-weighting-decision.md`](tfidf-weighting-decision.md) — the
  measured Phase 5F-C.1 recommendation for how a retained feature becomes
  a number: term-frequency rule, exact IDF formula, normalization,
  precision policy, vector index ordering, lossless vocabulary encoding,
  and the zero-vector and out-of-vocabulary semantics. Recommendation
  only — Phase 5F-D packages it

The frozen SMILES TF-IDF artifact itself (Phase 5F-D — identity, payload
schemas, the checksum DAG, immutability and rebuild rules, and the two
validation layers) and the production `smiles_tfidf_4096` FeatureAgent that
consumes it (Phase 5G — runtime pipeline, artifact lifecycle, OOV and
zero-vector semantics, feature-name format) are documented in the final
sections of [`reproducibility.md`](reproducibility.md), which also records
the Phase 5H per-agent error-isolation contract: how an unparseable
molecule, a failed representation, and a legitimately empty result are kept
distinguishable from one another, and the Phase 5I availability and
preflight contract: which agents exist, which can currently run, and
why a systemic prerequisite failure is rejected once per request rather
than reported once per molecule.

- [`benchmark-protocol.md`](benchmark-protocol.md) — the frozen Phase 6A
  scientific benchmark protocol: datasets and inclusion rules, the
  fixed-vector and sequence tracks, scaffold splits and seeds, the shared
  probes and tuning budget, metrics and cross-endpoint ranking, the
  statistical plan, failure accounting, and the reproducibility metadata
  every run must record. Protocol only — no benchmark has been executed
- [`benchmark-data.md`](benchmark-data.md) — the Phase 6A.1 TDC ADMET
  acquisition and dataset freeze: how PyTDC is kept out of the MolFusion
  dependency surface, the frozen serialization contract behind every
  checksum, what TDC's official split actually does (a fixed held-out test
  set, re-split only for train/validation), the amendment that split the
  benchmark into Track A1 and Track A2, and why Track A1 applies no
  cleaning to the official rows
- [`benchmark-execution.md`](benchmark-execution.md) — the Phase 6A.2 Track
  A1 execution contract: what "five seeds on one fixed test set" means for
  later statistics, the matrix feature cache and its key, checkpoint/resume
  semantics, the two result digests (file versus scientific identity), how
  the worker count was measured rather than guessed, and the leakage guards
  re-verified before every endpoint. Its final section records the Phase
  6A.5 execution-provenance contract: why worker-local git discovery wrote
  `null` into 338 of 616 historical shards, the once-per-run parent capture
  that replaced it, why tracked and untracked cleanliness are now separate
  fields, the tracked-diff identity, the fail-at-startup policy, and the
  historical audit that reports the gap without backfilling it
- [`benchmark-analysis.md`](benchmark-analysis.md) — the Phase 6A.3 Track
  A1 analysis contract: why the endpoint is the statistical unit and the
  five seeds are not five observations, the formal review of the two 6A.2
  amendments, direction-aware ranking with average ties, the Friedman →
  Holm-corrected Wilcoxon → rank-biserial → endpoint-bootstrap plan, and
  the regularization-saturation caveat that qualifies the linear results
- [`publication-evidence.md`](publication-evidence.md) — the Phase 6B
  publication evidence package: the claim registry and its
  prohibited-wording gate, the confidence-interval separation check behind
  the primary claim, why the linear result is a finding rather than a
  failure, the TF-IDF regression weakening, why ChEMBL exposure stays
  exploratory and untested, endpoint stability flagging and the 22-versus-19
  endpoint subsets, cost and performance as separate axes, the table and
  figure plan with exported figure data, the Methods evidence map, and the
  reproducibility statement that separates historical execution commits from
  the post-hoc provenance hardening
- [`manuscript/`](manuscript/) — the Phase 6C.1 manuscript freeze:
  [`MANUSCRIPT_ARCHITECTURE.md`](manuscript/MANUSCRIPT_ARCHITECTURE.md)
  (section structure, the three frozen editorial decisions, figure and table
  placement, and the rule that statistical support is attributed to the
  Friedman/Holm/rank-biserial chain rather than to interval non-overlap),
  [`TITLE_AND_ABSTRACT.md`](manuscript/TITLE_AND_ABSTRACT.md) (framing,
  twelve ranked title candidates, the contribution statement, the frozen
  abstract and its numerical sources, keywords), and
  [`CLAIM_TO_SECTION_MAP.md`](manuscript/CLAIM_TO_SECTION_MAP.md) with its
  machine-readable companion — which claim may appear where, and the
  prohibited-wording audit of the abstract. Phase 6C.2 adds
  [`METHODS_DRAFT.md`](manuscript/METHODS_DRAFT.md), the complete Materials
  and Methods section, and
  [`METHODS_EVIDENCE_MAP.md`](manuscript/METHODS_EVIDENCE_MAP.md), which
  traces every technical and numerical Methods statement to committed
  source or frozen output and lists the proposed Supplementary Methods.
  Phase 6C.3 adds [`RESULTS_DRAFT.md`](manuscript/RESULTS_DRAFT.md), the
  Results section §3.1-3.10, and
  [`RESULTS_EVIDENCE_MAP.md`](manuscript/RESULTS_EVIDENCE_MAP.md), which
  maps every Results paragraph to a registered claim and records the
  denominator reconciliation behind the reported compute shares. Phase
  6C.4 adds [`DISCUSSION_DRAFT.md`](manuscript/DISCUSSION_DRAFT.md) —
  Discussion §4.1-4.5, Limitations and Conclusion — and
  [`DISCUSSION_EVIDENCE_MAP.md`](manuscript/DISCUSSION_EVIDENCE_MAP.md),
  which classifies each interpretive paragraph as evidence, inference,
  hypothesis, methodological or limitation, and records the
  confidence-interval verification that confirmed the Results draft needed
  no correction. Phase 6C.5 completes the manuscript:
  [`INTRODUCTION_DRAFT.md`](manuscript/INTRODUCTION_DRAFT.md),
  [`FIGURE_CAPTIONS.md`](manuscript/FIGURE_CAPTIONS.md),
  [`TABLE_CAPTIONS.md`](manuscript/TABLE_CAPTIONS.md), a verified
  bibliography ([`references.json`](manuscript/references.json) as the
  structured source of truth, rendered to
  [`REFERENCES.md`](manuscript/REFERENCES.md), with per-reference
  verification recorded in
  [`REFERENCE_EVIDENCE_MAP.md`](manuscript/REFERENCE_EVIDENCE_MAP.md)), the
  assembled [`MANUSCRIPT_DRAFT.md`](manuscript/MANUSCRIPT_DRAFT.md) built
  deterministically from the section drafts by
  `molfusion_backend.benchmark.manuscript_cli`, and the whole-manuscript
  [`MANUSCRIPT_EVIDENCE_MAP.md`](manuscript/MANUSCRIPT_EVIDENCE_MAP.md)

Planned documents (not yet written):

- `architecture.md` — backend/frontend split, data flow
- `agents.md` — feature-agent registry: implemented vs. planned, with
  references for each algorithm
- `api.md` — generated from the FastAPI OpenAPI schema
