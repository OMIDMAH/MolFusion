import math

import numpy as np
import pytest

from molfusion_backend.corpus.study.weighting.weights import (
    IDF_SMOOTHED,
    IDF_UNSMOOTHED,
    NORM_L2,
    NORM_NONE,
    TF_RAW,
    TF_SUBLINEAR,
    idf_formula,
    inverse_document_frequency,
    l2_normalize,
    normalize,
    term_frequency,
    tf_formula,
    tfidf,
    weighting_definition,
)

# Known natural logarithms, written out so the tests below assert against
# arithmetic rather than against the implementation's own output.
LN2 = 0.6931471805599453
LN3 = 1.0986122886681098
LN4_3 = 0.2876820724517809  # ln(4/3)
LN3_2 = 0.4054651081081644  # ln(3/2)


# ---------------------------------------------------------------------------
# term frequency
# ---------------------------------------------------------------------------


def test_raw_tf_is_the_count_itself():
    assert term_frequency(np.array([0, 1, 2, 7]), TF_RAW).tolist() == [0.0, 1.0, 2.0, 7.0]


def test_sublinear_tf_is_one_plus_natural_log():
    weighted = term_frequency(np.array([0, 1, 2, 4]), TF_SUBLINEAR)
    assert weighted[0] == 0.0
    assert weighted[1] == pytest.approx(1.0)
    assert weighted[2] == pytest.approx(1.0 + LN2)
    assert weighted[3] == pytest.approx(1.0 + 2 * LN2)


def test_sublinear_tf_of_zero_is_exactly_zero_not_negative_infinity():
    """ln(0) is -inf; the guarded branch must never evaluate it."""
    weighted = term_frequency(np.zeros(5), TF_SUBLINEAR)
    assert np.all(weighted == 0.0)
    assert np.all(np.isfinite(weighted))


def test_sublinear_tf_compresses_repetition():
    """A motif repeated 100 times weighs ~5.6, not 100 -- the entire point."""
    raw = term_frequency(np.array([100]), TF_RAW)[0]
    sublinear = term_frequency(np.array([100]), TF_SUBLINEAR)[0]
    assert raw == 100.0
    assert sublinear == pytest.approx(1.0 + math.log(100))
    assert sublinear < 6.0


def test_both_tf_modes_agree_on_a_count_of_one():
    assert term_frequency(np.array([1]), TF_RAW)[0] == 1.0
    assert term_frequency(np.array([1]), TF_SUBLINEAR)[0] == pytest.approx(1.0)


def test_unknown_tf_mode_is_rejected():
    with pytest.raises(ValueError):
        term_frequency(np.array([1]), "binary")


# ---------------------------------------------------------------------------
# inverse document frequency
# ---------------------------------------------------------------------------


def test_smoothed_idf_matches_the_documented_formula():
    """idf(t) = ln((1 + N) / (1 + df(t))) + 1"""
    values = inverse_document_frequency(np.array([2, 3]), 3, IDF_SMOOTHED)
    assert values[0] == pytest.approx(1.0 + LN4_3)
    assert values[1] == pytest.approx(1.0)


def test_unsmoothed_idf_matches_the_documented_formula():
    """idf(t) = ln(N / df(t)) + 1"""
    values = inverse_document_frequency(np.array([2, 3]), 3, IDF_UNSMOOTHED)
    assert values[0] == pytest.approx(1.0 + LN3_2)
    assert values[1] == pytest.approx(1.0)


def test_a_term_in_every_document_weighs_exactly_one_under_both_formulas():
    """The trailing +1 floor: a universal feature is damped, never deleted."""
    assert inverse_document_frequency(np.array([1000]), 1000, IDF_UNSMOOTHED)[0] == pytest.approx(1.0)
    assert inverse_document_frequency(np.array([1000]), 1000, IDF_SMOOTHED)[0] == pytest.approx(1.0)


def test_idf_is_strictly_decreasing_in_document_frequency():
    frequencies = np.array([5, 50, 500, 5000, 50000])
    for mode in (IDF_SMOOTHED, IDF_UNSMOOTHED):
        values = inverse_document_frequency(frequencies, 100_000, mode)
        assert np.all(np.diff(values) < 0)


def test_smoothed_idf_is_always_below_unsmoothed_for_a_rare_term():
    smoothed = inverse_document_frequency(np.array([5]), 1_000_000, IDF_SMOOTHED)[0]
    unsmoothed = inverse_document_frequency(np.array([5]), 1_000_000, IDF_UNSMOOTHED)[0]
    assert smoothed < unsmoothed


def test_unsmoothed_idf_refuses_a_zero_document_frequency():
    """Not a division-by-zero crash but a stated error: a df of 0 means the
    df vector does not belong to the vocabulary it is being applied to."""
    with pytest.raises(ValueError, match="min_df"):
        inverse_document_frequency(np.array([0]), 100, IDF_UNSMOOTHED)


def test_smoothed_idf_tolerates_a_zero_document_frequency():
    assert np.isfinite(inverse_document_frequency(np.array([0]), 100, IDF_SMOOTHED)[0])


def test_idf_rejects_a_non_positive_corpus():
    with pytest.raises(ValueError):
        inverse_document_frequency(np.array([1]), 0, IDF_SMOOTHED)


def test_idf_rejects_negative_document_frequency():
    with pytest.raises(ValueError):
        inverse_document_frequency(np.array([-1]), 100, IDF_SMOOTHED)


def test_unknown_idf_mode_is_rejected():
    with pytest.raises(ValueError):
        inverse_document_frequency(np.array([1]), 10, "probabilistic")


def test_formulas_are_reported_as_text():
    assert idf_formula(IDF_SMOOTHED) == "idf(t) = ln((1 + N) / (1 + df(t))) + 1"
    assert idf_formula(IDF_UNSMOOTHED) == "idf(t) = ln(N / df(t)) + 1"
    assert tf_formula(TF_RAW) == "tf(t,d) = count(t in d)"
    assert "1 + ln(count" in tf_formula(TF_SUBLINEAR)


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------


def test_l2_normalization_gives_unit_length():
    normalized = l2_normalize(np.array([3.0, 4.0]))
    assert normalized.tolist() == pytest.approx([0.6, 0.8])
    assert float(np.linalg.norm(normalized)) == pytest.approx(1.0)


def test_l2_normalization_of_a_zero_vector_stays_zero_and_is_finite():
    """The contract that keeps a legitimately empty molecule from becoming
    NaN and silently poisoning everything downstream."""
    normalized = l2_normalize(np.zeros(4096))
    assert np.all(normalized == 0.0)
    assert np.all(np.isfinite(normalized))
    assert not np.any(np.isnan(normalized))


def test_l2_normalization_handles_a_zero_row_beside_nonzero_rows():
    matrix = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    normalized = l2_normalize(matrix)
    assert normalized[0].tolist() == pytest.approx([0.6, 0.8])
    assert normalized[1].tolist() == [0.0, 0.0]
    assert normalized[2].tolist() == pytest.approx([1.0, 0.0])
    assert np.all(np.isfinite(normalized))


def test_l2_normalization_preserves_direction():
    vector = np.array([1.0, 2.0, 3.0])
    scaled = l2_normalize(vector * 17.0)
    assert scaled.tolist() == pytest.approx(l2_normalize(vector).tolist())


def test_norm_none_leaves_the_vector_alone():
    vector = np.array([3.0, 4.0])
    assert normalize(vector, NORM_NONE).tolist() == [3.0, 4.0]


def test_unknown_norm_is_rejected():
    with pytest.raises(ValueError):
        normalize(np.array([1.0]), "l1")


# ---------------------------------------------------------------------------
# the composed transformation
# ---------------------------------------------------------------------------


def test_tfidf_applies_normalization_last():
    """Normalizing before the IDF multiply would make IDF partly cosmetic."""
    counts = np.array([1.0, 4.0])
    idf = np.array([1.0, 3.0])
    unnormalized = tfidf(counts, idf, tf_mode=TF_RAW, norm=NORM_NONE)
    assert unnormalized.tolist() == pytest.approx([1.0, 12.0])
    normalized = tfidf(counts, idf, tf_mode=TF_RAW, norm=NORM_L2)
    assert normalized.tolist() == pytest.approx(l2_normalize(unnormalized).tolist())


def test_tfidf_of_an_all_zero_count_vector_is_a_zero_vector():
    result = tfidf(np.zeros(8), np.full(8, 2.5), tf_mode=TF_SUBLINEAR, norm=NORM_L2)
    assert result.shape == (8,)
    assert np.all(result == 0.0)
    assert np.all(np.isfinite(result))


def test_output_dtype_is_applied_only_after_float64_arithmetic():
    counts = np.array([1.0, 3.0])
    idf = np.array([1.1, 2.2])
    exact = tfidf(counts, idf, dtype=np.float64)
    reduced = tfidf(counts, idf, dtype=np.float32)
    assert reduced.dtype == np.float32
    assert exact.dtype == np.float64
    assert reduced.astype(np.float64).tolist() == pytest.approx(exact.tolist(), abs=1e-6)


def test_weighting_definition_records_the_whole_contract():
    definition = weighting_definition(TF_SUBLINEAR, IDF_SMOOTHED, NORM_L2, "float64", "float32")
    assert definition["smooth_idf"] is True
    assert definition["use_idf"] is True
    assert definition["log_base"] == "e"
    assert definition["idf_formula"] == "idf(t) = ln((1 + N) / (1 + df(t))) + 1"
    assert definition["order_of_operations"] == "tf(counts) -> multiply by idf -> normalize"
    assert definition["internal_arithmetic_dtype"] == "float64"
    assert definition["idf_storage_dtype"] == "float64"
    assert definition["runtime_output_dtype"] == "float32"
    assert "not a failure" in definition["zero_vector"]
    assert "no UNK dimension" in definition["oov"]


def test_unsmoothed_definition_reports_smooth_idf_false():
    definition = weighting_definition(TF_RAW, IDF_UNSMOOTHED, NORM_NONE, "float64", "float64")
    assert definition["smooth_idf"] is False
    assert definition["norm"] == "none"
