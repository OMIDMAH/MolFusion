"""The frozen Phase 6A benchmark protocol, as executable constants.

Every decision a benchmark run must not make for itself lives here, so a
result file can name the protocol version that produced it and a reader can
recover exactly what was done. The prose version is
`docs/benchmark-protocol.md`; this module is the machine-readable half, and
the two are kept in step by tests.

The question the benchmark answers is deliberately narrow:

    given the same downstream model and the same tuning budget, how much
    predictive information does each fixed-length molecular representation
    make accessible, how stable is that across endpoints, and what does it
    cost?

Not "which pipeline wins a leaderboard". A leaderboard result confounds the
representation with the model, the tuning budget and the split; controlling
all three is the entire point.
"""

from typing import Any

PROTOCOL_VERSION = "6A.1"

# ---------------------------------------------------------------------------
# tracks
# ---------------------------------------------------------------------------

# Track A: fixed-length vectors, directly comparable under one shared head.
TRACK_A_REPRESENTATIONS = (
    "morgan_ecfp4_1024",
    "maccs_keys_167",
    "rdkit_physchem_descriptors",
    "avalon_1024",
    "erg_reduced_graph_315",
    "rdkit_fragment_descriptors",
    "smiles_tfidf_4096",
)

# Track B: variable-length token sequences. Deferred, not merged into Track A.
TRACK_B_REPRESENTATIONS = ("selfies_sequence",)

TRACK_B_DECISION = (
    "selfies_sequence is deferred from the primary benchmark. It is a "
    "variable-length categorical token sequence, so entering it into Track A "
    "would first require choosing an encoder -- bag-of-tokens, one-hot, a "
    "learned embedding -- and the resulting score would measure that encoder, "
    "not the representation MolFusion ships. A separate sequence-model track "
    "is future work, and its scores will not be directly comparable with "
    "Track A without accounting for the learned encoder and head."
)

# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
TASK_TYPES = (TASK_CLASSIFICATION, TASK_REGRESSION)

# ---------------------------------------------------------------------------
# dataset inclusion
# ---------------------------------------------------------------------------

# Below this, a scaffold-split test fold is too small for a stable estimate:
# a 20% test fold of 100 molecules is 20 compounds, where one prediction
# moves AUROC by several points. Stated as a rule so a dataset is never
# dropped after the fact for scoring badly.
MINIMUM_MOLECULES = 100
# A classification endpoint needs enough minority examples for AUPRC to mean
# anything; below this the metric is dominated by a handful of compounds.
MINIMUM_MINORITY_CLASS = 20

INCLUSION_CRITERIA = (
    "structure available as a single SMILES string per record",
    "single-molecule property prediction (no reactions, no mixtures)",
    "one clearly defined target label per endpoint",
    f"at least {MINIMUM_MOLECULES} usable molecules after validity filtering",
    f"classification endpoints: at least {MINIMUM_MINORITY_CLASS} minority-class molecules",
    "parseable by the pinned RDKit build",
    "documented provenance and a license permitting research use",
)

EXCLUSION_CRITERIA = (
    "endpoints excluded only by the criteria above, never for scoring poorly",
    "an endpoint failing a criterion is reported as excluded with the reason",
)

# ---------------------------------------------------------------------------
# molecule identity and duplicates
# ---------------------------------------------------------------------------

# The Phase 5F-A contract, reused rather than reinvented, so dataset identity
# and representation input are canonicalized identically.
CANONICALIZATION_ID = "rdkit_canonical_isomeric_smiles_v1"

# No salt stripping, tautomer canonicalization, neutralization,
# largest-fragment selection or stereo removal. Those are modelling choices
# that would change what is being predicted, and applying them here would
# silently benchmark a different dataset than the source defines.
STANDARDIZATION_POLICY = (
    "canonical isomeric SMILES only; no salt stripping, neutralization, "
    "tautomer canonicalization, largest-fragment selection or stereo removal"
)

DUPLICATE_POLICY_AGREEING = "collapse to one record"
DUPLICATE_POLICY_CONFLICTING = "drop all records of that molecule, and record the count"

DUPLICATE_POLICY = (
    "Records are grouped by canonical SMILES. A molecule appearing more than "
    "once with the same label collapses to a single record. A molecule "
    "appearing with conflicting labels is dropped entirely -- both copies -- "
    "and counted. Averaging conflicting labels would invent a value the "
    "source never asserted; keeping one arbitrarily would make the result "
    "depend on row order. Dropping is the only option that adds nothing and "
    "is reproducible, and the count is reported so the cost is visible."
)

# For regression, 'conflicting' needs a tolerance: two measurements of the
# same compound rarely agree to the last float. Relative tolerance against
# the endpoint's own label spread rather than an absolute value, since
# endpoints differ in units by orders of magnitude.
REGRESSION_CONFLICT_TOLERANCE_FRACTION = 0.01

# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

SPLIT_SCAFFOLD = "bemis_murcko_scaffold"
SPLIT_OFFICIAL = "dataset_official"
SPLIT_RANDOM = "random"

PRIMARY_SPLIT = SPLIT_SCAFFOLD

SPLIT_POLICY = (
    "Bemis-Murcko scaffold split is the default. A random split rewards "
    "memorising the scaffolds a dataset happens to contain, which is not the "
    "generalisation a representation is being judged on. Where a dataset "
    "publishes an official split, that split is used instead so results stay "
    "comparable with published numbers -- and the source is recorded per "
    "dataset, never mixed silently."
)

# Acyclic molecules have an empty Bemis-Murcko scaffold. They are grouped
# together under a single explicit key rather than each becoming its own
# scaffold, which is the conventional treatment; the group can be large, so
# its size is reported per dataset.
EMPTY_SCAFFOLD_KEY = "<acyclic>"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.20

# Deterministic, recorded, and never drawn at run time.
SPLIT_SEEDS = (0, 1, 2, 3, 4)
N_SPLITS = len(SPLIT_SEEDS)
# Model-training randomness is held fixed so split variability and training
# variability are not confounded into one spread. A separate seed sweep is a
# different experiment.
MODEL_SEED = 0

SEED_POLICY = (
    f"{N_SPLITS} deterministic scaffold splits, seeds {list(SPLIT_SEEDS)}, with "
    f"model training fixed at seed {MODEL_SEED}. Split variability and training "
    "variability are separated: the reported spread across splits is data "
    "variability alone, not the two mixed together."
)

# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

PROBE_LINEAR = "linear"
PROBE_NONLINEAR = "nonlinear"
PROBES = (PROBE_LINEAR, PROBE_NONLINEAR)

LINEAR_CLASSIFIER = "sklearn.linear_model.LogisticRegression"
LINEAR_REGRESSOR = "sklearn.linear_model.Ridge"
NONLINEAR_CLASSIFIER = "sklearn.ensemble.HistGradientBoostingClassifier"
NONLINEAR_REGRESSOR = "sklearn.ensemble.HistGradientBoostingRegressor"

PROBE_RATIONALE = (
    "Two levels, because they answer different questions. The linear probe "
    "measures how much predictive information is linearly accessible -- a "
    "property of the representation's geometry. The nonlinear probe measures "
    "what a capable tabular model can extract regardless of geometry. A "
    "representation that only the second can use is informative but not "
    "well-shaped, and reporting one number would hide that. "
    "HistGradientBoosting is the nonlinear family because it is already "
    "present via scikit-learn, is scale-invariant, handles the NaNs RDKit "
    "descriptors legitimately produce natively, and copes with 4096 columns "
    "-- adding XGBoost or LightGBM would mean a new heavy dependency for no "
    "argued scientific gain."
)

# Same grid for every representation. The comparison is between
# representations, so the head's budget is held constant; a per-representation
# search would let tuning effort masquerade as representation quality.
LINEAR_CLASSIFIER_GRID: dict[str, list[Any]] = {"C": [0.01, 0.1, 1.0, 10.0]}
LINEAR_REGRESSOR_GRID: dict[str, list[Any]] = {"alpha": [0.1, 1.0, 10.0, 100.0]}
NONLINEAR_GRID: dict[str, list[Any]] = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
}

TUNING_POLICY = (
    "One fixed grid per probe, identical for every representation: 4 linear "
    "candidates and 4 nonlinear candidates. Selection is on the validation "
    "split only, by the endpoint's primary metric. The test partition is "
    "read exactly once, after selection. The grids are deliberately small: "
    "the study compares representations, and an unequal or open-ended search "
    "would let tuning budget substitute for representation quality."
)

# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------

SCALING_NONE = "none"
SCALING_STANDARD = "standard"

# Per representation *and* per probe: the right preprocessing depends on both
# what the values mean and what the model needs.
LINEAR_SCALING = {
    # Bits. Already 0/1 and equally scaled; centering would destroy sparsity
    # and give an "absent" bit a nonzero value, which is not what it means.
    "morgan_ecfp4_1024": SCALING_NONE,
    "maccs_keys_167": SCALING_NONE,
    "avalon_1024": SCALING_NONE,
    # Physicochemical descriptors span many orders of magnitude and must be
    # standardized for a penalized linear model to be meaningful.
    "rdkit_physchem_descriptors": SCALING_STANDARD,
    # Small non-negative counts, but with very unequal variances.
    "rdkit_fragment_descriptors": SCALING_STANDARD,
    # Fuzzy pharmacophore counts on a common small scale.
    "erg_reduced_graph_315": SCALING_STANDARD,
    # Already L2-normalized by the frozen weighting contract; rescaling would
    # undo a deliberate part of the representation.
    "smiles_tfidf_4096": SCALING_NONE,
}

# Trees are invariant to monotone rescaling, so scaling would add a fitted
# step that changes nothing and could only introduce leakage risk.
NONLINEAR_SCALING = {name: SCALING_NONE for name in TRACK_A_REPRESENTATIONS}

SCALING_POLICY = (
    "Preprocessing is chosen per representation and per probe, never applied "
    "uniformly. Any scaler or imputer is fitted on the training split alone "
    "and applied unchanged to validation and test; fitting on anything else "
    "leaks the evaluation distribution into the model."
)

# RDKit descriptors legitimately emit NaN for descriptors that cannot be
# computed (see agents/descriptors.py). Linear models cannot consume NaN, so
# it is imputed with the training-split median; the tree probe consumes NaN
# natively and is left alone, so no value is invented for it.
NAN_POLICY_LINEAR = "median imputation fitted on the training split only"
NAN_POLICY_NONLINEAR = "passed through; HistGradientBoosting handles missing values natively"

# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

PRIMARY_CLASSIFICATION_METRIC = "auroc"
SECONDARY_CLASSIFICATION_METRICS = ("auprc", "balanced_accuracy", "mcc")

PRIMARY_REGRESSION_METRIC = "mae"
SECONDARY_REGRESSION_METRICS = ("rmse", "r2", "spearman")

METRIC_POLICY = (
    "AUROC is the primary classification metric by convention, but AUPRC is "
    "reported for every endpoint and is not optional: ADMET endpoints are "
    "often heavily imbalanced, and AUROC can look respectable on a model that "
    "ranks the minority class poorly. MAE is the primary regression metric "
    "because it is interpretable in the endpoint's own units and is not "
    "dominated by a few large errors; Spearman is always reported alongside "
    "it because it is unit-free and so remains meaningful when comparing "
    "behaviour across endpoints measured on different scales."
)

# Higher-is-better for every metric except these.
LOWER_IS_BETTER = ("mae", "rmse")

# ---------------------------------------------------------------------------
# cross-endpoint aggregation and statistics
# ---------------------------------------------------------------------------

RANKING_POLICY = (
    "Representations are ranked 1..7 within each endpoint on that endpoint's "
    "primary metric, after direction normalization, and only the ranks are "
    "aggregated across endpoints. Raw AUROC and MAE are never averaged "
    "together: they have different units, different directions and different "
    "attainable ranges, and their mean is not a quantity. Reported "
    "aggregates are mean rank, median rank and win count."
)

OMNIBUS_TEST = "friedman"
PAIRWISE_TEST = "wilcoxon_signed_rank"
MULTIPLE_COMPARISON_CORRECTION = "holm"
EFFECT_SIZE = "matched_pairs_rank_biserial_correlation"
CONFIDENCE_INTERVAL = "bootstrap_over_endpoints"
BOOTSTRAP_RESAMPLES = 10_000
ALPHA = 0.05

STATISTICS_POLICY = (
    "Comparisons are paired: every representation is evaluated on the same "
    "endpoints and the same splits. A Friedman omnibus test across all seven "
    "representations is run first; pairwise Wilcoxon signed-rank tests are "
    "performed only if it rejects, and their 21 p-values are Holm-corrected. "
    "Every p-value is reported with a matched-pairs rank-biserial effect size "
    "and a bootstrap confidence interval over endpoints -- a p-value alone "
    "says an ordering is detectable, not that it is large enough to matter."
)

UNCERTAINTY_POLICY = (
    "Split-level scores are retained, never only their mean. Each "
    "endpoint/representation/model cell reports mean, standard deviation and "
    "a 95% interval across splits, and the raw per-split rows stay in the "
    "result table so any later analysis can be recomputed rather than trusted."
)

# ---------------------------------------------------------------------------
# imbalance
# ---------------------------------------------------------------------------

IMBALANCE_POLICY = (
    "No synthetic resampling. SMOTE and its relatives invent molecules that "
    "do not exist, which is indefensible in a chemical benchmark and changes "
    "what the model is trained on in a representation-dependent way. Class "
    "imbalance is first characterised and reported per endpoint; where "
    "handling is needed, `class_weight='balanced'` is a model-side option "
    "applied identically to every representation, training-side only. The "
    "test distribution is never altered."
)

# ---------------------------------------------------------------------------
# representation failures and the evaluation universe
# ---------------------------------------------------------------------------

COMMON_MOLECULE_POLICY = (
    "The evaluation universe for an endpoint is fixed before any model is "
    "fitted: molecules that RDKit parses, deduplicated by canonical SMILES, "
    "and for which *every* Track A representation succeeds. Comparing "
    "representations on different molecule sets would let a representation "
    "look better by having failed on the hard compounds. Per-representation "
    "failure counts are reported, and if any endpoint loses more than "
    "COMMON_SET_LOSS_ALERT of its molecules to the intersection, that is "
    "flagged and a per-representation full-set sensitivity analysis is "
    "reported alongside the primary result."
)
COMMON_SET_LOSS_ALERT = 0.01

FAILURE_ACCOUNTING_FIELDS = (
    "input_records",
    "rdkit_invalid",
    "duplicates_collapsed",
    "duplicates_conflicting_dropped",
    "missing_label_dropped",
    "representation_failures",
    "common_evaluation_set",
)

# ---------------------------------------------------------------------------
# external corpus exposure
# ---------------------------------------------------------------------------

CHEMBL_OVERLAP_AUDIT = (
    "smiles_tfidf_4096 carries a vocabulary and IDF fitted on the frozen "
    "ChEMBL 37 corpus, so some benchmark molecules may have contributed to "
    "it. This is unsupervised exposure, not label leakage -- no benchmark "
    "label was ever read -- but it is an asymmetry the other six "
    "representations do not have, and it is reported rather than left "
    "implicit. The audit canonicalizes benchmark molecules with the frozen "
    "contract and counts membership in the corpus. Overlapping molecules are "
    "never removed: doing so would change the benchmark to suit one "
    "representation."
)
TFIDF_ARTIFACT_IDENTITY = "smiles_tfidf/chembl37_token_ngrams_1_3/1.0.0"
TFIDF_REFIT_POLICY = (
    "The TF-IDF artifact is frozen and is never refitted on benchmark data, "
    "per endpoint or otherwise. Refitting would evaluate a different "
    "representation from the one MolFusion ships, and one fitted on the "
    "evaluation distribution at that."
)


def protocol_summary() -> dict[str, Any]:
    """The frozen protocol as report-ready data."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "track_a_representations": list(TRACK_A_REPRESENTATIONS),
        "track_b_representations": list(TRACK_B_REPRESENTATIONS),
        "track_b_decision": TRACK_B_DECISION,
        "canonicalization_id": CANONICALIZATION_ID,
        "standardization_policy": STANDARDIZATION_POLICY,
        "duplicate_policy": DUPLICATE_POLICY,
        "primary_split": PRIMARY_SPLIT,
        "split_policy": SPLIT_POLICY,
        "split_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "split_seeds": list(SPLIT_SEEDS),
        "model_seed": MODEL_SEED,
        "seed_policy": SEED_POLICY,
        "probes": {
            "linear": {"classification": LINEAR_CLASSIFIER, "regression": LINEAR_REGRESSOR},
            "nonlinear": {
                "classification": NONLINEAR_CLASSIFIER,
                "regression": NONLINEAR_REGRESSOR,
            },
        },
        "probe_rationale": PROBE_RATIONALE,
        "tuning_policy": TUNING_POLICY,
        "scaling_policy": SCALING_POLICY,
        "linear_scaling": dict(LINEAR_SCALING),
        "nonlinear_scaling": dict(NONLINEAR_SCALING),
        "metrics": {
            "classification": {
                "primary": PRIMARY_CLASSIFICATION_METRIC,
                "secondary": list(SECONDARY_CLASSIFICATION_METRICS),
            },
            "regression": {
                "primary": PRIMARY_REGRESSION_METRIC,
                "secondary": list(SECONDARY_REGRESSION_METRICS),
            },
            "lower_is_better": list(LOWER_IS_BETTER),
        },
        "metric_policy": METRIC_POLICY,
        "ranking_policy": RANKING_POLICY,
        "statistics": {
            "omnibus": OMNIBUS_TEST,
            "pairwise": PAIRWISE_TEST,
            "correction": MULTIPLE_COMPARISON_CORRECTION,
            "effect_size": EFFECT_SIZE,
            "confidence_interval": CONFIDENCE_INTERVAL,
            "alpha": ALPHA,
        },
        "statistics_policy": STATISTICS_POLICY,
        "uncertainty_policy": UNCERTAINTY_POLICY,
        "imbalance_policy": IMBALANCE_POLICY,
        "common_molecule_policy": COMMON_MOLECULE_POLICY,
        "failure_accounting_fields": list(FAILURE_ACCOUNTING_FIELDS),
        "chembl_overlap_audit": CHEMBL_OVERLAP_AUDIT,
        "tfidf_artifact_identity": TFIDF_ARTIFACT_IDENTITY,
        "tfidf_refit_policy": TFIDF_REFIT_POLICY,
        "inclusion_criteria": list(INCLUSION_CRITERIA),
        "minimum_molecules": MINIMUM_MOLECULES,
        "minimum_minority_class": MINIMUM_MINORITY_CLASS,
    }


__all__ = [name for name in dir() if not name.startswith("_")]
