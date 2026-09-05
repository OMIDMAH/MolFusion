"""Phase 6C.2: the Methods draft must agree with frozen configuration.

Small and targeted. These check the statements that would be wrong if
drafted from memory rather than read from source -- several of them would
have been, without the check: the Avalon bit-flag, the ErG parameters, the
absence of class weighting, and the fact that scaling is applied to only
three of seven representations.

They also enforce two boundaries the phase depends on: Methods must state
the A1 fixed-test semantics explicitly, and Methods must contain no result.
"""

import re
from pathlib import Path

import pytest

from molfusion_backend.agents import registry
from molfusion_backend.benchmark import pipelines, protocol, publication

METHODS = Path("../docs/manuscript/METHODS_DRAFT.md")
EVIDENCE_MAP = Path("../docs/manuscript/METHODS_EVIDENCE_MAP.md")


@pytest.fixture(scope="module")
def methods():
    if not METHODS.exists():
        pytest.skip("methods draft not present")
    return METHODS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plain(methods):
    """Markdown emphasis, ticks and blockquote markers removed.

    Blockquote '> ' prefixes and wrapped lines would otherwise break
    substring matching on sentences the draft spreads across lines.
    """
    body = methods.replace("**", "").replace("`", "")
    body = re.sub(r"^> ?", "", body, flags=re.M)
    return re.sub(r"\s+", " ", body)


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heading", [
    "2.1 MolFusion framework",
    "2.2 Molecular representations",
    "2.3 TDC ADMET benchmark",
    "2.4 Molecular standardization, identity, and data curation",
    "2.5 Track A1 — official TDC evaluation",
    "2.6 Track A2 — independent scaffold robustness evaluation",
    "2.7 Predictive probes",
    "2.8 Hyperparameter selection and preprocessing",
    "2.9 Evaluation metrics",
    "2.10 Statistical analysis",
    "2.11 Computational cost measurement",
    "2.12 Reproducibility, result identity, and provenance",
])
def test_all_twelve_subsections_exist(methods, heading):
    assert f"## {heading}" in methods


def test_subsections_appear_in_order(methods):
    found = [tuple(int(p) for p in m.split("."))
             for m in re.findall(r"^## (2\.\d+)", methods, re.M)]
    assert found == sorted(found)
    assert len(found) == 12


# ---------------------------------------------------------------------------
# values that must match frozen source, not recollection
# ---------------------------------------------------------------------------


def test_every_track_a_representation_is_described(plain):
    for name in protocol.TRACK_A_REPRESENTATIONS:
        assert name in plain, f"{name} missing from Methods"


def test_declared_dimensions_match_the_registry(plain):
    agents = {a["id"]: a for a in registry.list_agents()}
    for name in protocol.TRACK_A_REPRESENTATIONS:
        assert str(agents[name]["output_dim"]) in plain


def test_morgan_configuration_matches_source(plain):
    from molfusion_backend.agents import morgan

    assert f"radius {morgan.MORGAN_RADIUS}" in plain
    assert f"{morgan.MORGAN_FP_SIZE} bits" in plain


def test_avalon_bit_flags_match_source(plain):
    from molfusion_backend.agents import avalon

    assert str(avalon.AVALON_BIT_FLAGS) in plain, (
        "the pinned Avalon bit-flag must be reported, not assumed default")


def test_erg_parameters_match_source(plain):
    from molfusion_backend.agents import erg

    assert f"atomTypes = {erg.ERG_ATOM_TYPES}" in plain
    assert f"fuzzIncrement = {erg.ERG_FUZZ_INCREMENT}" in plain
    assert f"minPath = {erg.ERG_MIN_PATH}" in plain
    assert f"maxPath = {erg.ERG_MAX_PATH}" in plain
    assert str(erg.ERG_OUTPUT_DIM) in plain


def test_tfidf_contract_matches_source(plain):
    from molfusion_backend.tfidf import contract

    assert contract.ARTIFACT_ID in plain
    assert f"version {contract.ARTIFACT_VERSION}" in plain
    assert str(contract.DIMENSION) in plain
    assert f"min_df = {contract.MIN_DF}" in plain


def test_canonicalisation_and_tokeniser_ids_are_named(plain):
    from molfusion_backend.chemistry import CANONICAL_SMILES_NORMALIZATION_ID
    from molfusion_backend.smiles_tokenizer import SMILES_TOKENIZER_ID

    assert CANONICAL_SMILES_NORMALIZATION_ID in plain
    assert SMILES_TOKENIZER_ID in plain


def test_split_fractions_and_seeds_match_the_protocol(plain):
    assert "(0.875, 0.125, 0.0)" in plain
    assert "seeds 1–5" in plain or "seeds 1-5" in plain
    assert "0–4" in plain or "0-4" in plain
    assert str(protocol.TRAIN_FRACTION * 100).rstrip("0").rstrip(".") + "%" in plain


def test_statistical_constants_match_the_protocol(plain):
    assert f"{protocol.BOOTSTRAP_RESAMPLES:,} resamples" in plain
    assert str(protocol.ALPHA) in plain
    assert protocol.MULTIPLE_COMPARISON_CORRECTION.capitalize() in plain


def test_metrics_match_the_protocol(plain):
    assert protocol.PRIMARY_CLASSIFICATION_METRIC.upper() in plain
    assert protocol.PRIMARY_REGRESSION_METRIC.upper() in plain
    written = {"r2": "R²", "rmse": "RMSE", "spearman": "Spearman"}
    for metric in protocol.SECONDARY_REGRESSION_METRICS:
        assert written[metric] in plain


def test_hyperparameter_grid_values_match_source(plain):
    linear = pipelines.hyperparameter_grid(protocol.PROBE_LINEAR,
                                           protocol.TASK_CLASSIFICATION)
    for candidate in linear:
        assert str(candidate["C"]) in plain
    nonlinear = pipelines.hyperparameter_grid(protocol.PROBE_NONLINEAR,
                                              protocol.TASK_REGRESSION)
    leaves = sorted({c["max_leaf_nodes"] for c in nonlinear})
    for value in leaves:
        assert str(value) in plain


# ---------------------------------------------------------------------------
# the four statements most likely to be wrong if assumed
# ---------------------------------------------------------------------------


def test_class_imbalance_handling_is_reported_as_absent(plain):
    """class_weight is None and no caller sets it."""
    import inspect

    source = inspect.getsource(pipelines.build_pipeline)
    assert "class_weight: str | None = None" in inspect.getsource(pipelines) or \
        "class_weight" in source
    assert "No class-imbalance handling was applied" in plain
    # SMOTE and resampling appear in the draft only inside the sentence that
    # denies their use, so the check is that nothing claims they were used.
    assert ("No class weighting, resampling, SMOTE, or threshold adjustment "
            "was used") in plain
    for claimed in ("SMOTE was applied", "we oversampled", "class weights were used"):
        assert claimed.lower() not in plain.lower()


def test_scaling_is_described_as_representation_specific(plain):
    scaled = [r for r in protocol.TRACK_A_REPRESENTATIONS
              if pipelines.scaling_for(r, protocol.PROBE_LINEAR) == protocol.SCALING_STANDARD]
    assert len(scaled) == 3, "source says three representations are standardised"
    for name in scaled:
        assert name in plain
    assert "not uniform" in plain.lower() or "representation- and probe-specific" in plain


def test_no_scaling_under_the_nonlinear_probe_is_stated(plain):
    for name in protocol.TRACK_A_REPRESENTATIONS:
        assert pipelines.scaling_for(name, protocol.PROBE_NONLINEAR) == protocol.SCALING_NONE
    assert "no scaler" in plain.lower()


def test_tfidf_is_noted_as_already_normalised(plain):
    assert "already L2-normalised" in plain or "already L2 normalized" in plain


# ---------------------------------------------------------------------------
# track semantics
# ---------------------------------------------------------------------------


def test_a1_fixed_test_semantics_is_stated_explicitly(plain):
    assert "shipped TDC test partition was held fixed" in plain
    assert "five deterministic scaffold-based train/validation realizations" in plain


def test_a1_denies_five_independent_test_sets(plain):
    assert "do not generate five independent test sets" in plain


def test_a1_applies_no_cleaning(plain):
    assert "Track A1 applies no cleaning" in plain


def test_a2_is_not_called_external_validation(plain):
    assert "not external validation" in plain
    assert "supplementary robustness analysis" in plain


def test_a2_confound_is_stated_in_methods(plain):
    assert "not fully separable" in plain


def test_a2_does_not_claim_all_seeds_gave_distinct_partitions(plain):
    assert "not every endpoint produced five distinct test" in plain
    assert "19 of the 22" in plain


def test_a2_scaffold_chirality_matches_source(plain):
    from molfusion_backend.benchmark import tdc

    assert tdc.MOLFUSION_SCAFFOLD_INCLUDES_CHIRALITY is True
    assert "includeChirality = True" in plain


# ---------------------------------------------------------------------------
# stability decisions carried into Methods
# ---------------------------------------------------------------------------


def test_the_six_low_stability_endpoints_are_named(plain):
    for endpoint in publication.PRE_REGISTERED_LOW_STABILITY:
        assert endpoint in plain


def test_low_stability_endpoints_are_not_described_as_removed(plain):
    assert "remain in the overall benchmark" in plain
    for wrong in ("were excluded from the benchmark", "were dropped"):
        assert wrong not in plain


def test_vdss_lombardo_is_borderline_and_not_excluded(plain):
    assert "vdss_lombardo" in plain
    assert "BORDERLINE" in plain
    assert "not excluded" in plain
    assert "vdss_lombardo" not in publication.PRE_REGISTERED_LOW_STABILITY


def test_stability_threshold_matches_source(plain):
    assert str(publication.LOW_STABILITY_W_THRESHOLD) in plain


# ---------------------------------------------------------------------------
# SELFIES exclusion
# ---------------------------------------------------------------------------


def test_selfies_exclusion_is_explained_as_scope_not_failure(plain):
    assert "SELFIES" in plain
    assert "excluded from Track A" in plain
    assert "variable-length categorical token sequence" in plain
    assert "scope decision, not a negative finding" in plain
    for wrong in ("SELFIES failed", "SELFIES performed", "SELFIES was worse"):
        assert wrong not in plain


def test_selfies_is_registered_but_has_no_fixed_dimension():
    agents = {a["id"]: a for a in registry.list_agents()}
    assert agents["selfies_sequence"]["output_dim"] is None


# ---------------------------------------------------------------------------
# provenance honesty
# ---------------------------------------------------------------------------


def test_historical_and_hardened_provenance_are_distinguished(plain):
    assert "did not produce the results reported" in plain
    assert "not applied retroactively" in plain


def test_execution_commits_are_named(plain):
    for commit in ("459653b", "ddabb42", "2bcb467", "e6ae297", "15b78a2", "89335dc"):
        assert commit in plain


def test_provenance_defect_is_disclosed_without_overclaiming(plain):
    assert "logging defect" in plain
    assert "were unaffected" in plain
    for wrong in ("provenance was complete", "fully provenanced", "reconstructed the results"):
        assert wrong not in plain


def test_exact_shard_counts_are_routed_to_supplementary():
    body = EVIDENCE_MAP.read_text(encoding="utf-8")
    assert "338 of 616" in body, "the exact count must not be hidden entirely"
    assert "Supplementary" in body


def test_software_versions_match_the_frozen_run_reports(plain):
    import json

    report = Path("benchmark_runs/track_a1/run_report.json")
    if not report.exists():
        pytest.skip("run report not present")
    env = json.loads(report.read_text("utf-8"))["environment"]
    for key in ("python", "rdkit", "numpy", "scikit_learn"):
        assert env[key] in plain, f"{key} version {env[key]} not reported"


# ---------------------------------------------------------------------------
# Methods must contain no result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "performed best", "outperform", "achieved the best", "ranked highest",
    "was superior", "9 of 11", "9/11", "1.91", "1.77", "became the leader",
    "weakened under", "strongest",
])
def test_methods_contains_no_result(plain, phrase):
    assert phrase.lower() not in plain.lower(), f"'{phrase}' is a Results statement"


@pytest.mark.parametrize("phrase", [
    "remarkably", "surprisingly", "powerful", "state-of-the-art",
    "best-in-class", "superior",
])
def test_methods_avoids_promotional_language(plain, phrase):
    assert phrase.lower() not in plain.lower()


def test_methods_avoids_the_prohibited_collective_term(plain):
    assert publication.PROHIBITED_COLLECTIVE_TERM not in plain.lower()


def test_methods_contains_no_causal_exposure_language(plain):
    lowered = plain.lower()
    for wrong in ("leakage", "contamination", "exposure explains", "inflated"):
        assert wrong not in lowered


def test_exposure_uses_the_frozen_term(plain):
    assert "external unsupervised corpus exposure" in plain


def test_ci_is_not_described_as_a_significance_test(plain):
    assert "not used as a significance test" in plain
    assert "uncertainty visualisation only" in plain


# ---------------------------------------------------------------------------
# evidence map
# ---------------------------------------------------------------------------


def test_the_evidence_map_covers_all_twelve_subsections():
    body = EVIDENCE_MAP.read_text(encoding="utf-8")
    for index in range(1, 13):
        assert f"## 2.{index}" in body


def test_citation_placeholders_are_neutral(methods):
    placeholders = re.findall(r"\[CITATION: ([^\]]+)\]", methods)
    assert placeholders, "external claims need citation placeholders"
    for placeholder in placeholders:
        assert not re.search(r"\b(19|20)\d\d\b", placeholder), (
            f"fabricated bibliographic detail in '{placeholder}'")


def test_both_documents_cite_the_frozen_evidence_identity():
    identity = "5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18"
    for path in (METHODS, EVIDENCE_MAP):
        assert identity in path.read_text(encoding="utf-8")
