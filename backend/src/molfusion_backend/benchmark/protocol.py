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
# Phase 6A.1 reconciliation: official TDC splits vs MolFusion splits
# ---------------------------------------------------------------------------
#
# The constants above are Phase 6A as frozen, and they are left exactly as
# they were written. What follows is an amendment, added after measuring the
# actual PyTDC 1.0.0 implementation rather than reading its documentation.
#
# What Phase 6A assumed: five independent 70/10/20 scaffold splits, each
# drawing its own test partition, so the spread across splits would include
# test-partition variability.
#
# What TDC actually does, established from source and confirmed against the
# data for all 22 endpoints:
#
#   * train_val.csv and test.csv are shipped files. get() and __next__()
#     read them directly and neither accepts a seed.
#   * get_train_valid_split(seed, ...) reads train_val.csv ONLY, and splits
#     it with frac = [0.875, 0.125, 0.0]. The trailing 0.0 is the test
#     fraction: no test set is drawn, because one was already held out.
#   * The seed therefore moves the train/validation boundary and nothing
#     else. Across all 22 endpoints the test molecule set hashed to the
#     same SHA-256 at every seed.
#   * evaluate_many() requires at least 5 runs.
#
# Nesting those fractions reproduces Phase 6A's numbers exactly --
# 0.8 * 0.875 = 0.70 train and 0.8 * 0.125 = 0.10 validation, against a 0.20
# test set -- so the FRACTIONS in Phase 6A were right. What was wrong was the
# assumption that the test partition is re-drawn per seed. It is not.
#
# That difference is not cosmetic: it changes what the spread across runs
# measures. Under TDC's protocol the spread is sensitivity to the
# train/validation boundary on a fixed evaluation set; under Phase 6A's
# original reading it would also have included which molecules are evaluated
# at all. Both are legitimate questions, so both are kept -- as two tracks
# that are never mixed and never share a label.

# --- Phase 6A.2 amendment: non-finite descriptor values ----------------
#
# Phase 6A specified that RDKit descriptors emit NaN where a descriptor
# cannot be computed, and defined handling for it: median imputation for the
# linear probe, native consumption by the trees. It did not anticipate
# +/-inf, which RDKit also emits -- MaxPartialCharge and MaxAbsPartialCharge
# diverge for certain structures. Exactly one molecule in the whole 22
# endpoint suite triggers it (a solubility_aqsoldb training row), producing
# two infinite values out of 152 feature matrices.
#
# scikit-learn tolerates NaN in both probes and rejects inf in both, so those
# two values failed all seven of that endpoint's cells.
#
# The amendment is the smallest one that resolves it: an infinite descriptor
# means what a missing one means -- not meaningfully computable for this
# molecule -- so +/-inf is folded onto NaN and then handled by the machinery
# already frozen. Dropping the molecule was not available, because Track A1
# may not alter the official partitions.
#
# The fold is stateless, applied uniformly to every representation and both
# probes, and is the identity on finite input -- so it changes no result
# computed before it existed.
NON_FINITE_POLICY = "fold +/-inf onto NaN, then apply the frozen NaN policy"
NON_FINITE_POLICY_RATIONALE = (
    "An infinite descriptor value carries the same information as a missing "
    "one: the quantity is not meaningfully computable for that molecule. "
    "Routing it through the existing NaN policy avoids inventing a second "
    "mechanism, and avoids the only alternatives -- dropping a molecule, "
    "which Track A1 forbids, or clipping to an arbitrary finite bound, which "
    "would assert a value the descriptor never produced."
)


TRACK_A1 = "tdc_official"
TRACK_A2 = "molfusion_scaffold"
EVALUATION_TRACKS = (TRACK_A1, TRACK_A2)

# --- Track A1: the official, leaderboard-comparable evaluation ---------
TRACK_A1_SPLIT_STRATEGY = "tdc_official_fixed_test"
TRACK_A1_SEEDS = (1, 2, 3, 4, 5)
TRACK_A1_TRAIN_VAL_FRACTIONS = (0.875, 0.125, 0.0)
TRACK_A1_TEST_IS_FIXED = True

TRACK_A1_DEFINITION = (
    "Track A1 reproduces TDC's official ADMET protocol: the shipped held-out "
    "test set, unchanged and identical at every seed, with train and "
    "validation drawn from train_val.csv at fractions 0.875/0.125 using TDC's "
    "own scaffold splitter, over five runs. Its purpose is comparability with "
    "published TDC leaderboard numbers, so nothing about the official "
    "partitions is modified -- including duplicate molecules that TDC's rows "
    "contain, because removing them would mean scoring on a different test "
    "set than every published number was computed on."
)

# Seed values: PyTDC fixes the run COUNT (evaluate_many enforces >= 5) but
# not the seed values themselves, which the caller passes. 1-5 is the
# convention TDC's own documentation demonstrates, so it is what A1 uses.
# Phase 6A's 0-4 is deliberately NOT reused here: two tracks that differ in
# meaning must not collide in a result file just because both start at a
# small integer.
TRACK_A1_SEED_POLICY = (
    "PyTDC enforces a minimum of five runs but does not fix the seed values; "
    "the caller supplies them. Track A1 uses TDC's documented 1-5 convention. "
    "Track A2 keeps Phase 6A's 0-4. The values differ so that a row's seed "
    "alone can never make the two tracks ambiguous."
)

# --- Track A2: MolFusion's own robustness analysis ---------------------
TRACK_A2_SPLIT_STRATEGY = SPLIT_SCAFFOLD
TRACK_A2_SEEDS = SPLIT_SEEDS
TRACK_A2_TEST_IS_FIXED = False

TRACK_A2_DEFINITION = (
    "Track A2 is MolFusion's own repeated-scaffold-split analysis: five "
    "independent 70/10/20 Bemis-Murcko splits, seeds 0-4, drawn over the "
    "cleaned molecule universe with the full duplicate and conflict policy "
    "applied. It answers a question A1 cannot -- whether a representation's "
    "ranking survives a different scaffold partition, or is an artefact of "
    "the one partition TDC happened to publish. It is NOT comparable with "
    "TDC leaderboard numbers and must never be labelled or reported as "
    "official TDC results."
)

# --- Phase 6A.4 amendment C: partition variability of the A2 splitter ----
#
# Recorded before any A2 score was computed, from the pre-execution audit.
#
# scaffold_split orders groups by (-group_size, hash(seed, scaffold)): size
# dominates, and the seed only permutes within an equal-size tier. How much
# the seed actually moves the test set therefore depends on where the 80%
# train+validation boundary falls relative to the group-size tiers.
#
# Measured over the cleaned universes, as mean pairwise Jaccard overlap
# between the five test sets (lower = the partitions differ more):
#
#   most endpoints          0.17 - 0.35   the boundary falls inside the large
#                                         singleton tier, which the seed
#                                         genuinely permutes
#   ames                    0.85          multi-member groups nearly cover the
#   ld50_zhu                0.88          boundary, so little is left to permute
#   solubility_aqsoldb      1.00          multi-member groups (8,510 molecules)
#                                         already exceed the train+validation
#                                         target (7,984), so the test set is
#                                         IDENTICAL at all five seeds
#
# The splitter is NOT changed. It was frozen in Phase 6A, and altering it
# after observing this property would be exactly the post-hoc adjustment the
# protocol exists to prevent -- and would make A2 incomparable with its own
# specification.
#
# Nor is the affected endpoint dropped: removing it would shrink A2's
# endpoint set relative to A1 and weaken the comparison A2 exists to make.
#
# Instead the flag is recorded per endpoint, and cross-endpoint claims about
# repartitioning are reported both over all 22 endpoints and over the subset
# whose partitions genuinely vary. solubility_aqsoldb contributes exactly
# zero repartitioning information and must not be cited as evidence that a
# finding survived repartitioning.
A2_PARTITION_VARIABILITY_ALERT = 0.50
A2_LOW_VARIABILITY_ENDPOINTS = ("solubility_aqsoldb", "ames", "ld50_zhu")
A2_PARTITION_VARIABILITY_POLICY = (
    "Endpoints whose five A2 test sets have mean pairwise Jaccard overlap "
    f"above {A2_PARTITION_VARIABILITY_ALERT:.2f} are flagged as providing "
    "little repartitioning signal. They are executed and reported, never "
    "dropped, but cross-endpoint repartitioning claims are additionally "
    "reported over the genuinely-repartitioned subset."
)


TRACK_A2_STATUS = "supplementary"
TRACK_A2_STATUS_RATIONALE = (
    "A1 is the headline result because it is the one a reader can check "
    "against published work. A2 is reported as a supplementary robustness "
    "analysis: it is evidence about the stability of the A1 ranking, not a "
    "second opinion competing with it. Promoting A2 to the main manuscript "
    "would invite exactly the confusion this separation exists to prevent."
)


def split_id(track: str, seed: int) -> str:
    """A split identifier that names its track, so results cannot be mixed.

    Every result row carries this string. A bare seed would make
    ``seed=1`` under A1 and ``seed=1`` under A2 indistinguishable in a
    merged table, which is the specific failure this phase exists to
    prevent.
    """
    if track not in EVALUATION_TRACKS:
        raise ValueError(f"unknown evaluation track: {track!r}")
    return f"{track}/seed={seed}"


# --- cleaning policy per track ----------------------------------------
#
# Measured, not assumed. Applying MolFusion's conflicting-label rule to the
# official partitions removes 0% from 8 endpoints and under 6% from 12
# more -- but 30% of clearance_hepatocyte_az and 53-58% of ppbr_az, whose
# rows carry many replicate measurements of the same compound. A test set
# missing 58% of its molecules is not the test set the leaderboard used, so
# for A1 the official rows are consumed as shipped and the duplicate
# structure is reported instead of removed.

TRACK_A1_CLEANING = "none; official rows consumed exactly as shipped"
TRACK_A2_CLEANING = "full Phase 6A policy: canonicalize, collapse agreeing duplicates, drop conflicting groups"

CLEANING_DIVERGENCE_ALERT = 0.05
CLEANING_POLICY_RATIONALE = (
    "MolFusion's duplicate policy is right for MolFusion's own analysis and "
    "wrong for a leaderboard comparison. Track A1 therefore applies no "
    "cleaning: it reports the duplicate and conflict structure of the "
    "official partitions as a caveat on those numbers, rather than silently "
    "evaluating on a smaller set and presenting the result as comparable. "
    f"Any endpoint where cleaning would move more than "
    f"{CLEANING_DIVERGENCE_ALERT:.0%} of either official partition is flagged "
    "explicitly wherever A1 and A2 results appear together."
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
        # Phase 6A.1 amendment: the split fractions above survived contact
        # with the real TDC implementation; the assumption that the test
        # partition is re-drawn per seed did not. Two tracks, never mixed.
        "evaluation_tracks": {
            TRACK_A1: {
                "definition": TRACK_A1_DEFINITION,
                "split_strategy": TRACK_A1_SPLIT_STRATEGY,
                "seeds": list(TRACK_A1_SEEDS),
                "train_val_fractions": list(TRACK_A1_TRAIN_VAL_FRACTIONS),
                "test_is_fixed": TRACK_A1_TEST_IS_FIXED,
                "cleaning": TRACK_A1_CLEANING,
                "status": "primary",
            },
            TRACK_A2: {
                "definition": TRACK_A2_DEFINITION,
                "split_strategy": TRACK_A2_SPLIT_STRATEGY,
                "seeds": list(TRACK_A2_SEEDS),
                "test_is_fixed": TRACK_A2_TEST_IS_FIXED,
                "cleaning": TRACK_A2_CLEANING,
                "status": TRACK_A2_STATUS,
            },
        },
        "track_seed_policy": TRACK_A1_SEED_POLICY,
        "cleaning_policy_rationale": CLEANING_POLICY_RATIONALE,
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
