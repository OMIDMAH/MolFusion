import csv
import json

import numpy as np
import pytest

from molfusion_backend.artifacts.checksum import sha256_file
from molfusion_backend.corpus.errors import CorpusIdentityError, CorpusOutputExistsError
from molfusion_backend.corpus.serialization import write_corpus
from molfusion_backend.corpus.study.weighting import payload
from molfusion_backend.corpus.study.weighting.diagnostics import (
    WeightingDiagnostics,
    idf_comparison,
    pearson,
    spearman,
    vectorize,
)
from molfusion_backend.corpus.study.weighting.report import (
    CANDIDATE_IDF_MODE,
    CANDIDATE_NORM,
    CANDIDATE_TF_MODE,
    REPORT_FILENAME,
    VOCABULARY_PREVIEW_LIMIT,
    VOLATILE_STUDY_PATHS,
    deterministic_study_view,
    run_weighting_study,
    study_report_bytes,
)

FIXTURE_SMILES = sorted(
    {"CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "N", "O"}
    | {f"C{'C' * index}O" for index in range(1, 400)}
    | {f"c1ccccc1{'C' * index}" for index in range(1, 300)}
    | {f"CC(=O)N{'C' * index}" for index in range(1, 120)}
)


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "canonical_smiles.smi"
    sha256, _ = write_corpus(path, FIXTURE_SMILES)
    return path, sha256


def study(corpus, tmp_path, **kwargs):
    path, sha256 = corpus
    kwargs.setdefault("expected_sha256", sha256)
    kwargs.setdefault("expected_document_count", len(FIXTURE_SMILES))
    kwargs.setdefault("min_df", 5)
    kwargs.setdefault("dimension", 64)
    kwargs.setdefault("progress_every", 0)
    return run_weighting_study(path, tmp_path / kwargs.pop("out", "study"), **kwargs)


# ---------------------------------------------------------------------------
# correlation helpers
# ---------------------------------------------------------------------------


def test_pearson_detects_a_perfect_linear_relationship():
    x = np.arange(10, dtype=float)
    assert pearson(x, 3.0 * x + 1.0) == pytest.approx(1.0)
    assert pearson(x, -2.0 * x) == pytest.approx(-1.0)


def test_spearman_detects_a_monotone_but_nonlinear_relationship():
    x = np.arange(1, 11, dtype=float)
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert pearson(x, x**3) < 1.0


def test_spearman_averages_tied_ranks():
    """Token counts are small integers with huge tie groups; an arbitrary
    tie order would make the statistic depend on sort stability."""
    x = np.array([1.0, 1.0, 2.0, 2.0])
    assert spearman(x, np.array([5.0, 5.0, 9.0, 9.0])) == pytest.approx(1.0)


def test_correlation_of_a_constant_series_is_undefined_not_zero():
    x = np.arange(5, dtype=float)
    assert pearson(x, np.ones(5)) is None
    assert pearson(np.array([1.0]), np.array([2.0])) is None


# ---------------------------------------------------------------------------
# vectorization and OOV
# ---------------------------------------------------------------------------


def test_out_of_vocabulary_ngrams_contribute_nothing_and_raise_nothing():
    index = {("C",): 0, ("O",): 1}
    vector = vectorize(("C", "N", "N", "O", "P"), index, (1,), smiles_length=5)
    assert vector.indices.tolist() == [0, 1]
    assert vector.counts.tolist() == [1.0, 1.0]
    assert vector.nonzero == 2


def test_a_molecule_of_only_oov_ngrams_gives_an_empty_support():
    vector = vectorize(("N", "P"), {("C",): 0}, (1,), smiles_length=2)
    assert vector.nonzero == 0
    assert vector.indices.size == 0
    assert vector.counts.size == 0


def test_vectorize_sums_counts_across_orders_into_their_own_columns():
    index = {("C",): 0, ("C", "C"): 1}
    vector = vectorize(("C", "C", "C"), index, (1, 2), smiles_length=3)
    assert vector.indices.tolist() == [0, 1]
    assert vector.counts.tolist() == [3.0, 2.0]


def test_vectorize_support_is_sorted_by_column():
    index = {("O",): 5, ("C",): 2, ("N",): 9}
    vector = vectorize(("N", "O", "C"), index, (1,), smiles_length=3)
    assert vector.indices.tolist() == [2, 5, 9]


# ---------------------------------------------------------------------------
# IDF comparison
# ---------------------------------------------------------------------------


def test_idf_comparison_reports_both_formulas_and_their_difference():
    comparison = idf_comparison([5, 50, 500], 1000)
    assert comparison["n_documents"] == 1000
    assert comparison["terms"] == 3
    assert comparison["smoothed"]["count"] == 3
    assert comparison["unsmoothed"]["count"] == 3
    assert comparison["absolute_difference"]["max"] > 0
    assert comparison["formulas"]["smoothed"] == "idf(t) = ln((1 + N) / (1 + df(t))) + 1"


def test_smoothing_never_reorders_terms():
    """Both formulas are strictly decreasing in df, so only the spacing
    differs -- which is why the choice is about numbers, not ranking."""
    comparison = idf_comparison([5, 17, 100, 9999, 500_000], 1_000_000)
    assert comparison["rank_order_identical"] is True


def test_idf_difference_is_largest_for_the_rarest_terms():
    comparison = idf_comparison([5, 5, 900_000, 900_000], 1_000_000)
    bands = {row["document_frequency_band"]: row for row in comparison["by_document_frequency_band"]}
    rare = bands["(0,10]"]["mean_absolute_difference"]
    common = max(
        row["mean_absolute_difference"]
        for label, row in bands.items()
        if label != "(0,10]"
    )
    assert rare > common


# ---------------------------------------------------------------------------
# per-molecule diagnostics
# ---------------------------------------------------------------------------


def test_a_zero_support_molecule_is_counted_and_never_divides_by_zero():
    accumulator = WeightingDiagnostics(np.array([1.0, 2.0]), 2)
    accumulator.add(vectorize(("N",), {("C",): 0, ("O",): 1}, (1,), 1), "small")
    assert accumulator.all_zero_molecules == 1

    precision = accumulator.precision_report()["by_stratum"]["small"]
    assert precision["l2_vector_difference"]["max"] == 0.0
    # A zero vector has no defined cosine, so it is excluded rather than
    # recorded as a similarity of 1.0.
    assert precision["cosine_similarity"]["count"] == 0


def test_sublinear_tf_lowers_concentration_when_a_motif_repeats():
    accumulator = WeightingDiagnostics(np.array([1.0, 1.0]), 2)
    # One feature occurs 50 times, the other once.
    accumulator.add(vectorize(("C",) * 50 + ("O",), {("C",): 0, ("O",): 1}, (1,), 51), "small")

    block = accumulator.tf_report()["by_stratum"]["small"]
    assert block["max_feature_raw_tf"]["max"] == 50.0
    assert block["raw_top_share"]["mean"] == pytest.approx(50 / 51)
    assert block["sublinear_top_share"]["mean"] < block["raw_top_share"]["mean"]


def test_norm_report_records_correlations_per_stratum_and_pooled():
    accumulator = WeightingDiagnostics(np.array([1.0, 1.0]), 2)
    index = {("C",): 0, ("O",): 1}
    for count in range(2, 30):
        accumulator.add(vectorize(("C",) * count + ("O",), index, (1,), count + 1), "small")

    report = accumulator.norm_report()
    raw = report["by_stratum"]["small"]["raw_tf_no_norm"]
    assert raw["pearson_vs_token_count"] > 0.9
    assert report["pooled"]["l2_normalized"]["pearson_vs_token_count"] == 0.0


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_identity_gate_aborts_before_writing_anything(corpus, tmp_path):
    path, _ = corpus
    output = tmp_path / "study"
    with pytest.raises(CorpusIdentityError):
        run_weighting_study(path, output, expected_sha256="0" * 64, progress_every=0)
    assert not output.exists()


def test_document_count_mismatch_aborts(corpus, tmp_path):
    with pytest.raises(CorpusIdentityError):
        study(corpus, tmp_path, expected_document_count=len(FIXTURE_SMILES) + 1)


def test_study_never_modifies_the_corpus(corpus, tmp_path):
    path, sha256 = corpus
    before = path.read_bytes()
    study(corpus, tmp_path)
    assert path.read_bytes() == before
    assert sha256_file(path) == sha256


def test_report_records_provenance_and_declares_no_artifact(corpus, tmp_path):
    report = study(corpus, tmp_path)
    assert report["produces_production_artifact"] is False
    assert report["phase"] == "5F-C.1"
    assert report["corpus"]["verified_sha256"] == corpus[1]
    assert report["corpus"]["uses_downstream_labels"] is False
    assert report["run"]["software"]["python"]
    assert report["run"]["software"]["numpy"]


def test_report_states_the_candidate_contract(corpus, tmp_path):
    contract = study(corpus, tmp_path)["candidate_contract"]
    assert contract["tf_mode"] == CANDIDATE_TF_MODE
    assert contract["idf_mode"] == CANDIDATE_IDF_MODE
    assert contract["norm"] == CANDIDATE_NORM
    assert contract["use_idf"] is True
    assert contract["internal_arithmetic_dtype"] == "float64"


def test_vocabulary_section_separates_min_df_from_the_cap(corpus, tmp_path):
    vocabulary = study(corpus, tmp_path)["vocabulary"]
    assert vocabulary["selection"]["min_df"] == 5
    assert vocabulary["selection"]["dimension"] == 64
    assert vocabulary["selected_terms"] <= 64
    assert vocabulary["document_frequency_min"] >= 5
    assert vocabulary["indexing"]["depends_on_dict_insertion_order"] is False


def test_every_selected_term_respects_min_df(corpus, tmp_path):
    report = study(corpus, tmp_path)
    assert report["vocabulary"]["document_frequency_min"] >= report["vocabulary"]["selection"]["min_df"]


def test_sample_covers_the_strata_and_reports_its_sizes(corpus, tmp_path):
    sample = study(corpus, tmp_path)["sample"]
    assert sample["molecules"] == sum(sample["by_stratum"].values())
    assert set(sample["by_stratum"]) == {"small", "typical", "large", "very_long"}
    assert sample["definition"]["df_and_idf_source"] == "the full frozen corpus, never the sample"


def test_all_outputs_are_written_and_no_production_payload_is(corpus, tmp_path):
    study(corpus, tmp_path)
    written = {p.name for p in (tmp_path / "study").iterdir()}
    assert written == {
        "weighting_report.json",
        "idf_comparison.csv",
        "tf_concentration.csv",
        "norm_vs_length.csv",
        "precision.csv",
        "vocabulary_preview.csv",
        "corpus_pass_cache.json",
    }
    # Explicitly not a production artifact.
    assert "vocabulary.json" not in written
    assert "idf.npy" not in written
    assert "metadata.json" not in written


def test_vocabulary_preview_is_bounded_and_losslessly_encoded(corpus, tmp_path):
    study(corpus, tmp_path)
    with (tmp_path / "study" / "vocabulary_preview.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert 0 < len(rows) <= VOCABULARY_PREVIEW_LIMIT
    for row in rows:
        tokens = json.loads(row["tokens"])
        assert isinstance(tokens, list)
        assert len(tokens) == int(row["order"])


def test_report_on_disk_matches_the_returned_report(corpus, tmp_path):
    report = study(corpus, tmp_path)
    written = (tmp_path / "study" / REPORT_FILENAME).read_bytes()
    assert written == study_report_bytes(report)
    assert written.endswith(b"\n")
    assert b"\r\n" not in written


def test_csv_outputs_use_lf_only(corpus, tmp_path):
    study(corpus, tmp_path)
    for name in ("idf_comparison.csv", "precision.csv", "norm_vs_length.csv"):
        assert b"\r\n" not in (tmp_path / "study" / name).read_bytes()


def test_existing_output_is_not_overwritten_without_force(corpus, tmp_path):
    study(corpus, tmp_path)
    with pytest.raises(CorpusOutputExistsError):
        study(corpus, tmp_path)
    study(corpus, tmp_path, force=True)


# ---------------------------------------------------------------------------
# determinism and caching
# ---------------------------------------------------------------------------


def test_two_runs_produce_the_same_scientific_result(corpus, tmp_path):
    first = study(corpus, tmp_path, out="one")
    second = study(corpus, tmp_path, out="two")
    assert deterministic_study_view(first) == deterministic_study_view(second)


def test_two_runs_produce_byte_identical_tables(corpus, tmp_path):
    study(corpus, tmp_path, out="one")
    study(corpus, tmp_path, out="two")
    for name in ("idf_comparison.csv", "tf_concentration.csv", "norm_vs_length.csv",
                 "precision.csv", "vocabulary_preview.csv"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_a_cached_corpus_pass_reproduces_the_uncached_result(corpus, tmp_path):
    fresh = study(corpus, tmp_path, out="run")
    assert fresh["run"]["corpus_pass_cached"] is False
    cached = study(corpus, tmp_path, out="run", force=True)
    assert cached["run"]["corpus_pass_cached"] is True
    assert deterministic_study_view(fresh) == deterministic_study_view(cached)


def test_a_cache_for_different_parameters_is_not_reused(corpus, tmp_path):
    study(corpus, tmp_path, out="run", dimension=64)
    other = study(corpus, tmp_path, out="run", dimension=32, force=True)
    assert other["run"]["corpus_pass_cached"] is False
    assert other["vocabulary"]["selected_terms"] <= 32


def test_only_timings_and_cache_state_are_excused_from_determinism():
    assert {key for _, key in VOLATILE_STUDY_PATHS} == {
        "started_at",
        "elapsed_seconds",
        "corpus_pass_seconds",
        "diagnostics_seconds",
        "corpus_pass_cached",
    }
    assert {section for section, _ in VOLATILE_STUDY_PATHS} == {"run"}
