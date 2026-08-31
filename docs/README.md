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

Planned documents (not yet written):

- `architecture.md` — backend/frontend split, data flow
- `agents.md` — feature-agent registry: implemented vs. planned, with
  references for each algorithm
- `api.md` — generated from the FastAPI OpenAPI schema
