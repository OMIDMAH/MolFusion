# Methods evidence map (Phase 6C.2)

Every technical and numerical statement in
[`METHODS_DRAFT.md`](METHODS_DRAFT.md) traces to a committed source or a
frozen output. Configuration values were read from source at drafting time,
not recalled — several (the Avalon bit-flag, the ErG parameters, the
absence of class weighting, the representation-specific scaling policy)
would have been wrong if inferred from names or defaults.

Evidence identity: `5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`.

---

## 2.1 MolFusion framework

| Statement | Source | Value |
| --- | --- | --- |
| FeatureAgent registry, declared identity/version/value type/dimension | `agents/registry.py`, `agents/base.py` | 8 registered agents, all v1.0.0 |
| Agent identity participates in the feature-cache key | `benchmark/feature_store.py::matrix_cache_key` | includes `agent_id`, `agent_version`, `output_dim` |
| Protocol frozen before execution | `benchmark/protocol.py` | `PROTOCOL_VERSION = "6A.1"` |
| Seven Track A representations | `protocol.TRACK_A_REPRESENTATIONS` | 7 identifiers |
| SELFIES registered but excluded from Track A | `agents/selfies_agent.py`; registry listing | `selfies_sequence`, `output_dim = None`, `value_type = categorical` |

Citations: `[CITATION: SELFIES]`.

## 2.2 Molecular representations

| Representation | Source | Frozen values |
| --- | --- | --- |
| `morgan_ecfp4_1024` | `agents/morgan.py` | `MORGAN_RADIUS = 2`, `MORGAN_FP_SIZE = 1024`, `rdFingerprintGenerator.GetMorganGenerator` |
| `maccs_keys_167` | `agents/maccs.py`; registry | dim 167, binary, RDKit MACCS |
| `rdkit_physchem_descriptors` | `agents/descriptors.py`; registry | dim 217, continuous, frozen name order |
| `avalon_1024` | `agents/avalon.py` | `AVALON_BIT_FLAGS = 15761407`, `nBits = 1024`, `pyAvalonTools.GetAvalonFP` |
| `erg_reduced_graph_315` | `agents/erg.py` | `atomTypes = 0`, `fuzzIncrement = 0.3`, `minPath = 1`, `maxPath = 15`, `ERG_OUTPUT_DIM = 315` (derived, not assumed) |
| `rdkit_fragment_descriptors` | `agents/fragments.py`; registry | dim 85, count |
| `smiles_tfidf_4096` | `agents/smiles_tfidf.py`, `tfidf/contract.py` | see below |

TF-IDF contract (`tfidf/contract.py`):

| Field | Value |
| --- | --- |
| `ARTIFACT_ID` | `chembl37_token_ngrams_1_3` |
| `ARTIFACT_VERSION` | `1.0.0` |
| `DIMENSION` / `MAX_FEATURES` | 4096 |
| `NGRAM_ORDERS` | (1, 2, 3) |
| `MIN_DF` | 5 (absolute) |
| `SELECTION_RANKING` | descending full-corpus document frequency |
| `SELECTION_TIE_BREAK` | ascending lexicographic n-gram token tuple |
| `INDEX_ORDER` | lexicographic token tuple after selection |
| `OOV_POLICY` | out-of-vocabulary n-gram contributes nothing; no UNK dimension, no refit, no error |
| `ZERO_VECTOR_POLICY` | valid zero vector, not a failure; L2 leaves it exactly zero |
| weighting | sublinear TF × smoothed IDF, L2-normalised (`agents/smiles_tfidf.py` docstring) |

Normalisation and tokenisation:

| Contract | Source | Value |
| --- | --- | --- |
| Canonicalisation | `chemistry.CANONICAL_SMILES_NORMALIZATION_ID` | `rdkit_canonical_isomeric_smiles_v1` |
| Tokenisation | `smiles_tokenizer.SMILES_TOKENIZER_ID` | `rdkit_smiles_lexer_v1` |
| Losslessness | Phase 5F-A contract, `docs/reproducibility.md` | concatenated tokens reproduce input exactly |

Citations: `[CITATION: RDKit]`, `[CITATION: Morgan fingerprint; ECFP]`,
`[CITATION: MACCS keys]`, `[CITATION: Avalon]`, `[CITATION: ErG]`,
`[CITATION: ChEMBL]`, `[CITATION: SMILES]`.

## 2.3 TDC ADMET benchmark

| Statement | Source | Value |
| --- | --- | --- |
| Release name | `benchmark_manifests/tdc_admet_group.json` | `TDC-ADMET-2026-09` |
| Endpoint count and composition | manifest | 22 endpoints: **13 classification, 9 regression** |
| All retained | manifest `included` flags | all `True` |
| Serialization contract | `benchmark/release.py` | `molfusion_frozen_csv_v1` |
| Release identity | manifest `release_identity_sha256` | `10bda5f0…35e3` (Supplementary) |
| PyTDC not a runtime dependency | `docs/benchmark-data.md`; test asserting no `tdc` import | acquisition/reconciliation only |

Citations: `[CITATION: TDC]`, `[CITATION: TDC ADMET]`.

## 2.4 Standardization, identity, curation

| Statement | Source | Value |
| --- | --- | --- |
| Identity = canonical isomeric SMILES | `protocol.CANONICALIZATION_ID` | `rdkit_canonical_isomeric_smiles_v1` |
| Regression conflict tolerance | `protocol.REGRESSION_CONFLICT_TOLERANCE_FRACTION` | 0.01 (1%) |
| A1 applies no cleaning | `protocol.TRACK_A1_CLEANING` | "none; official rows consumed exactly as shipped" |
| A2 applies full curation | `benchmark/a2.py::load_cleaned_endpoint` | duplicates collapsed, conflicts dropped |
| Invalid structures | manifest `solubility_aqsoldb.ingestion` | `rdkit_invalid = 2` of 9,982; **only endpoint affected** |
| Overall curation effect | manifest, summed | 81,809 official records → 79,712 usable |

## 2.5 Track A1

| Statement | Source | Value |
| --- | --- | --- |
| Split strategy | `protocol.TRACK_A1_SPLIT_STRATEGY` | `tdc_official_fixed_test` |
| Train/validation fractions | `protocol.TRACK_A1_TRAIN_VAL_FRACTIONS` / `tdc.OFFICIAL_TRAIN_VAL_FRACTIONS` | `(0.875, 0.125, 0.0)` |
| Seeds | `protocol.TRACK_A1_SEEDS` / `tdc.OFFICIAL_SEEDS` | (1, 2, 3, 4, 5) |
| Fixed test partition | `benchmark/a1.py::official_splits`; `docs/benchmark-data.md` | shipped `test.csv`, identical across seeds |
| Official scaffold convention | `tdc.OFFICIAL_SCAFFOLD_INCLUDES_CHIRALITY` | `False` |
| ≈70/10/20 derivation | 0.8 × 0.875 = 0.70; 0.8 × 0.125 = 0.10; test 0.20 | arithmetic from the two above |

## 2.6 Track A2

| Statement | Source | Value |
| --- | --- | --- |
| Track identifier | `protocol.TRACK_A2`; `a2.TRACK` | `molfusion_scaffold` |
| Split strategy | `protocol.TRACK_A2_SPLIT_STRATEGY` | `bemis_murcko_scaffold` |
| Seeds | `protocol.TRACK_A2_SEEDS` / `a2.SEEDS` | (0, 1, 2, 3, 4) |
| Fractions | `protocol.TRAIN_FRACTION` / `VALIDATION_FRACTION` / `TEST_FRACTION` | 0.7 / 0.1 / 0.2 |
| Scaffold chirality | `benchmark/splits.py:61`; `tdc.MOLFUSION_SCAFFOLD_INCLUDES_CHIRALITY` | `includeChirality = True` |
| Distinctness threshold | `protocol.A2_PARTITION_VARIABILITY_ALERT` | 0.50 mean pairwise test Jaccard |
| Genuinely repartitioned | `publication/tables/table7_…csv`; `split_distinctness.csv` | **19 of 22** |
| Repartitioning/cleaning confound | Amendment C; `docs/benchmark-analysis.md` | stated in §2.6, not deferred |

Citations: `[CITATION: Bemis–Murcko scaffold]`.

## 2.7 Predictive probes

| Statement | Source | Value |
| --- | --- | --- |
| Probe families | `protocol.PROBES` | (`linear`, `nonlinear`) |
| Linear estimators | `benchmark/pipelines.py` | `LogisticRegression(max_iter=5000)`, `Ridge` |
| Nonlinear estimators | `benchmark/pipelines.py` | `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` |
| Equal budget | `protocol.TRACK_A_REPRESENTATIONS`, single grid function | one grid for all representations |

Citations: `[CITATION: scikit-learn]`.

## 2.8 Hyperparameters and preprocessing

| Statement | Source | Value |
| --- | --- | --- |
| Grid | `pipelines.hyperparameter_grid` | linear-clf `C` ∈ {0.01, 0.1, 1, 10}; linear-reg `alpha` ∈ {0.1, 1, 10, 100}; nonlinear `learning_rate` ∈ {0.05, 0.1} × `max_leaf_nodes` ∈ {15, 31} — 4 candidates each |
| Scaling policy | `pipelines.scaling_for`; `protocol.SCALING_POLICY` | `standard` for physchem, ErG, fragments (linear probe only); `none` for all others and for every nonlinear cell |
| Imputation | `pipelines.build_pipeline` | `SimpleImputer(strategy="median")`, linear probe only, fitted on train |
| Non-finite fold | `pipelines._non_finite_to_nan` | stateless ±∞ → NaN, both probes |
| Leakage prevention | `sklearn.Pipeline` composition | fitted steps see training fold only |
| **Class imbalance** | `pipelines.build_pipeline` `class_weight` default `None`; no caller sets it | **no weighting, no resampling** |
| Test used once | `a1.py` / `a2.py` `run_cell` | selection on validation; single test evaluation |

## 2.9 Metrics

| Statement | Source | Value |
| --- | --- | --- |
| Primary classification | `protocol.PRIMARY_CLASSIFICATION_METRIC` | `auroc` |
| Secondary classification | `protocol.SECONDARY_CLASSIFICATION_METRICS` | `auprc`, `balanced_accuracy`, `mcc` |
| Primary regression | `protocol.PRIMARY_REGRESSION_METRIC` | `mae` |
| Secondary regression | `protocol.SECONDARY_REGRESSION_METRICS` | `rmse`, `r2`, `spearman` |
| Direction | `protocol.LOWER_IS_BETTER`; `metrics.orient` | `mae`, `rmse` lower-is-better |
| Metric rationale | `protocol.METRIC_POLICY` | AUPRC mandatory under imbalance; MAE interpretable, Spearman unit-free |
| Within-endpoint ranking, average ties | `metrics.rank_within_endpoint`; `analysis.rank_endpoint` | ranks, never raw-metric averages |
| Summaries | `analysis.summarise_ranks` | mean rank, median rank, wins, top-3 |

## 2.10 Statistical analysis

| Statement | Source | Value |
| --- | --- | --- |
| Unit of inference | `analysis_a2_cli` configuration; `docs/benchmark-analysis.md` | `statistical_unit = "endpoint"` |
| Seeds not replicates | `analysis.SEED_AGGREGATION` | aggregated within endpoint before inference |
| Friedman | `analysis.friedman` | per probe × task family |
| Pairwise gate | `analysis_cli` / `analysis_a2_cli` | pairwise only where Friedman rejects |
| Wilcoxon + Holm | `analysis.pairwise_tests`, `analysis.holm`; `protocol.MULTIPLE_COMPARISON_CORRECTION` | `holm` |
| Effect size | `protocol.EFFECT_SIZE` | `matched_pairs_rank_biserial_correlation` |
| α | `protocol.ALPHA` | 0.05 |
| Bootstrap | `protocol.BOOTSTRAP_RESAMPLES`, `analysis.BOOTSTRAP_SEED`, `bootstrap_mean_rank.csv` | 10,000 resamples, seed 0, unit = endpoint |
| CI not a test | `publication.ci_separation` docstring; architecture §4 | marginal intervals, visualisation only |
| No cross-track seed pairing | `protocol.TRACK_A1_SEED_POLICY` | disambiguated by track and `split_id`, never seed |
| Kendall's W | `analysis_a2.kendall_w`; `kendall_w.csv` | per endpoint × probe |
| Low-stability rule | `publication.LOW_STABILITY_W_THRESHOLD`; `publication.stability_table` | min across probes < 0.35 |
| Six pre-registered endpoints | `publication.PRE_REGISTERED_LOW_STABILITY` | herg, cyp2c9_substrate, clearance_hepatocyte_az, cyp2d6_substrate, cyp3a4_substrate, bioavailability_ma |
| `vdss_lombardo` BORDERLINE | `table6_endpoint_stability.csv` | flag `BORDERLINE`, `pre_registered = False` |

Citations: `[CITATION: Friedman test]`, `[CITATION: Wilcoxon signed-rank test]`,
`[CITATION: Holm correction]`, `[CITATION: Kendall's W]`, `[CITATION: SciPy]`.

## 2.11 Computational cost

| Statement | Source | Value |
| --- | --- | --- |
| Cost components | `timings.json`; `cost_summary.csv` | `feature_seconds`, `selection_seconds`, `fit_seconds`, `test_predict_seconds`, `validation_predict_seconds` |
| Feature caching | `benchmark/feature_store.py` | one matrix per endpoint × representation, reused |
| Host and concurrency | A2 `run_report.json` `run_summary` | 2 workers, 37.8 h wall-clock |
| Timings excluded from scientific identity | `a1_runner`/`a2_runner` `SCIENTIFIC_COLUMNS` | no duration column present |
| No composite score | `table5_computational_cost.csv`; test asserting no such column | cost and rank on separate axes |

## 2.12 Reproducibility and provenance

| Statement | Source | Value |
| --- | --- | --- |
| Scientific identity columns | `runner.SCIENTIFIC_COLUMNS` | 18 science-only columns |
| A1 scientific identity | A1 `run_report.json` | `d40ef09b…de868` |
| A2 scientific identity | A2 `run_report.json` | `9dd5dfa6…69f14` |
| Publication identity | `publication_report.json` | `5790359b…7acc` |
| Feature-cache key | `feature_store.matrix_cache_key` | release, endpoint, agent id/version, dim, normalisation, ordered row digest, artifact identity |
| Atomic shards, resume | `runner.write_shard`, `read_valid_shard` | staged write + `os.replace` |
| A1 execution commits | `provenance_audit.json` | `459653b`, `ddabb42`, `2bcb467` |
| A2 execution commit | `provenance_audit.json` | `e6ae297` |
| Analysis commit | Phase 6C.1 freeze | `15b78a2` |
| Hardening commit (subsequent) | `docs/benchmark-execution.md` §11 | `89335dc` |
| Shard provenance gap | `provenance_audit.json` | **338 of 616** shards lack an embedded commit → Supplementary |
| Hardened mechanism not retroactive | `benchmark/provenance.py`; audit `backfilled_into_shards: false` | stated explicitly in §2.12 |
| Software versions | A1 and A2 `run_report.json` `environment` | Python 3.11.15, RDKit 2026.03.5, NumPy 2.4.6, scikit-learn 1.9.0 |
| SciPy version | analysis environment (not in run report) | 1.17.1 — disclosed as such |

---

## Proposed Supplementary Methods

Detail deliberately kept out of the main Methods:

| Item | Content |
| --- | --- |
| S-M1 | Per-endpoint SHA-256 checksums and the release identity |
| S-M2 | TF-IDF artifact payload hashes and the checksum DAG |
| S-M3 | Full hyperparameter grid with per-cell selected values |
| S-M4 | Complete computational cost table by representation and probe |
| S-M5 | Per-shard provenance counts: **338 of 616** lacking an embedded commit, by track (A1 127/308, A2 211/308) |
| S-M6 | Per-endpoint curation accounting: duplicates collapsed, conflicts dropped, invalid structures |
| S-M7 | Per-endpoint split distinctness (mean/max pairwise test Jaccard) and the 19-endpoint criterion |

## Citation placeholders used

`[CITATION: TDC]` · `[CITATION: TDC ADMET]` · `[CITATION: RDKit]` ·
`[CITATION: Morgan fingerprint; ECFP]` · `[CITATION: MACCS keys]` ·
`[CITATION: Avalon]` · `[CITATION: ErG]` · `[CITATION: SELFIES]` ·
`[CITATION: SMILES]` · `[CITATION: ChEMBL]` ·
`[CITATION: Bemis–Murcko scaffold]` · `[CITATION: scikit-learn]` ·
`[CITATION: SciPy]` · `[CITATION: Friedman test]` ·
`[CITATION: Wilcoxon signed-rank test]` · `[CITATION: Holm correction]` ·
`[CITATION: Kendall's W]`

No bibliographic details were fabricated; reference assembly is a later
phase.
