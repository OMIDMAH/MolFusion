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

Planned documents (not yet written):

- `architecture.md` — backend/frontend split, data flow
- `agents.md` — feature-agent registry: implemented vs. planned, with
  references for each algorithm
- `api.md` — generated from the FastAPI OpenAPI schema
