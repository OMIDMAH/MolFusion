"""A fully worked TF-IDF example, computed by hand and pinned.

Section 18 of the phase brief: the mathematical contract must be
independently verifiable, and an implementation must not be tested only
against itself. Every expected number below is derived from the stated
formulas using plain `math` and explicit arithmetic -- a different code
path from the vectorized NumPy implementation under test -- and the IDF
constants are additionally written out as decimal literals.

The corpus, in full:

    D1 = ("O", "B", "O")
    D2 = ("O", "B")
    D3 = ("O", "C")
    N  = 3

Unigram document frequencies:  ("O",) = 3   ("B",) = 2   ("C",) = 1
Bigram document frequencies:   ("O","B") = 2   ("B","O") = 1   ("O","C") = 1

With min_df = 2 the vocabulary is exactly three terms:

    ("O",) df 3,  ("B",) df 2,  ("O","B") df 2

Selection ranking, (-df, tokens):   ("O",) < ("B",) < ("O","B")
Index ordering, lexicographic:      ("B",) < ("O",) < ("O","B")

The two orders differ, which is the point of the fixture: it would pass
under either rule if they happened to agree.
"""

import math

import numpy as np
import pytest

from molfusion_backend.corpus.study.ngrams import NgramFrequencyAccumulator
from molfusion_backend.corpus.study.weighting import payload, weights
from molfusion_backend.corpus.study.weighting.diagnostics import vectorize

D1 = ("O", "B", "O")
D2 = ("O", "B")
D3 = ("O", "C")
DOCUMENTS = (D1, D2, D3)
N = 3

# ln(4/3) and ln(3/2), written out. These are the only two nontrivial
# logarithms the fixture needs.
LN_4_3 = 0.2876820724517809
LN_3_2 = 0.4054651081081644
LN_2 = 0.6931471805599453

EXPECTED_DF = {
    ("O",): 3,
    ("B",): 2,
    ("C",): 1,
    ("O", "B"): 2,
    ("B", "O"): 1,
    ("O", "C"): 1,
}

# idf = ln((1+N)/(1+df)) + 1, N = 3
EXPECTED_SMOOTHED_IDF = {
    ("O",): 1.0,               # ln(4/4) + 1
    ("B",): 1.0 + LN_4_3,      # ln(4/3) + 1
    ("O", "B"): 1.0 + LN_4_3,
}

# idf = ln(N/df) + 1, N = 3
EXPECTED_UNSMOOTHED_IDF = {
    ("O",): 1.0,               # ln(3/3) + 1
    ("B",): 1.0 + LN_3_2,      # ln(3/2) + 1
    ("O", "B"): 1.0 + LN_3_2,
}


@pytest.fixture()
def accumulated():
    accumulator = NgramFrequencyAccumulator((1, 2))
    for tokens in DOCUMENTS:
        accumulator.add_document(tokens, holdout=False)
    return {order: accumulator.entries(order) for order in (1, 2)}


@pytest.fixture()
def terms(accumulated):
    selected = payload.select_terms(accumulated, min_df=2, dimension=10)
    return payload.assign_indices(selected, order=payload.INDEX_ORDER_LEXICOGRAPHIC)


# ---------------------------------------------------------------------------
# the inputs
# ---------------------------------------------------------------------------


def test_document_frequencies_are_as_hand_counted(accumulated):
    observed = {
        entry.ngram: entry.document_frequency
        for entries in accumulated.values()
        for entry in entries
    }
    assert observed == EXPECTED_DF


def test_min_df_selects_exactly_three_terms(accumulated):
    selected = payload.select_terms(accumulated, min_df=2, dimension=10)
    assert [entry.ngram for entry in selected] == [("O",), ("B",), ("O", "B")]


def test_selection_ranking_and_index_order_genuinely_differ(terms):
    assert [term.tokens for term in terms] == [("B",), ("O",), ("O", "B")]
    assert [term.selection_rank for term in terms] == [1, 0, 2]


def test_indices_and_document_frequencies_line_up(terms):
    assert [term.index for term in terms] == [0, 1, 2]
    assert payload.document_frequencies(terms) == [2, 3, 2]


# ---------------------------------------------------------------------------
# IDF against hand-computed constants
# ---------------------------------------------------------------------------


def test_smoothed_idf_matches_hand_computation(terms):
    idf = weights.inverse_document_frequency(
        np.array(payload.document_frequencies(terms)), N, weights.IDF_SMOOTHED
    )
    for term in terms:
        assert idf[term.index] == pytest.approx(EXPECTED_SMOOTHED_IDF[term.tokens])
    assert idf.tolist() == pytest.approx([1.2876820724517809, 1.0, 1.2876820724517809])


def test_unsmoothed_idf_matches_hand_computation(terms):
    idf = weights.inverse_document_frequency(
        np.array(payload.document_frequencies(terms)), N, weights.IDF_UNSMOOTHED
    )
    for term in terms:
        assert idf[term.index] == pytest.approx(EXPECTED_UNSMOOTHED_IDF[term.tokens])
    assert idf.tolist() == pytest.approx([1.4054651081081644, 1.0, 1.4054651081081644])


# ---------------------------------------------------------------------------
# the full transformation, end to end
# ---------------------------------------------------------------------------


def _expected_vector(counts_by_token, idf_by_token, tf_mode, normalize):
    """Recompute the expected vector in plain Python, no NumPy."""
    order = [("B",), ("O",), ("O", "B")]
    values = []
    for tokens in order:
        count = counts_by_token.get(tokens, 0)
        if count == 0:
            tf = 0.0
        elif tf_mode == "raw":
            tf = float(count)
        else:
            tf = 1.0 + math.log(count)
        values.append(tf * idf_by_token[tokens])
    if not normalize:
        return values
    length = math.sqrt(sum(value * value for value in values))
    if length == 0.0:
        return values
    return [value / length for value in values]


def test_d1_counts_are_as_hand_counted(terms):
    """D1 = (O, B, O): O twice, B once, and the bigram (O,B) once.
    Its other bigram (B,O) has df 1 and is not in the vocabulary."""
    vector = vectorize(D1, payload.term_index(terms), (1, 2), smiles_length=3)
    assert vector.indices.tolist() == [0, 1, 2]
    assert vector.counts.tolist() == [1.0, 2.0, 1.0]


def test_d1_sublinear_smoothed_l2_matches_hand_computation(terms):
    counts = {("B",): 1, ("O",): 2, ("O", "B"): 1}
    expected = _expected_vector(counts, EXPECTED_SMOOTHED_IDF, "sublinear", normalize=True)

    idf = weights.inverse_document_frequency(
        np.array(payload.document_frequencies(terms)), N, weights.IDF_SMOOTHED
    )
    produced = weights.tfidf(
        np.array([1.0, 2.0, 1.0]), idf, tf_mode=weights.TF_SUBLINEAR, norm=weights.NORM_L2
    )

    assert produced.tolist() == pytest.approx(expected)
    assert float(np.linalg.norm(produced)) == pytest.approx(1.0)
    # And spelled out completely, so the numbers are readable in the diff:
    #   B  : 1            * (1 + ln(4/3)) = 1.2876820724517809
    #   O  : (1 + ln 2)   * 1             = 1.6931471805599453
    #   OB : 1            * (1 + ln(4/3)) = 1.2876820724517809
    unnormalized = [1.0 + LN_4_3, 1.0 + LN_2, 1.0 + LN_4_3]
    length = math.sqrt(sum(v * v for v in unnormalized))
    assert produced.tolist() == pytest.approx([v / length for v in unnormalized])


def test_d1_raw_tf_differs_from_sublinear_on_the_repeated_token(terms):
    idf = weights.inverse_document_frequency(
        np.array(payload.document_frequencies(terms)), N, weights.IDF_SMOOTHED
    )
    counts = np.array([1.0, 2.0, 1.0])
    raw = weights.tfidf(counts, idf, tf_mode=weights.TF_RAW, norm=weights.NORM_NONE)
    sublinear = weights.tfidf(counts, idf, tf_mode=weights.TF_SUBLINEAR, norm=weights.NORM_NONE)

    # Only ("O",) is repeated, so only column 1 may differ.
    assert raw[0] == pytest.approx(sublinear[0])
    assert raw[2] == pytest.approx(sublinear[2])
    assert raw[1] == pytest.approx(2.0)
    assert sublinear[1] == pytest.approx(1.0 + LN_2)
    assert sublinear[1] < raw[1]


def test_d3_drops_its_out_of_vocabulary_ngrams(terms):
    """D3 = (O, C). ("C",) has df 1 and ("O","C") has df 1, so neither is a
    feature; only ("O",) survives, and nothing errors."""
    vector = vectorize(D3, payload.term_index(terms), (1, 2), smiles_length=2)
    assert vector.indices.tolist() == [1]
    assert vector.counts.tolist() == [1.0]


def test_a_document_with_no_vocabulary_term_yields_a_zero_vector(terms):
    idf = weights.inverse_document_frequency(
        np.array(payload.document_frequencies(terms)), N, weights.IDF_SMOOTHED
    )
    vector = vectorize(("C", "C"), payload.term_index(terms), (1, 2), smiles_length=2)
    assert vector.nonzero == 0

    dense = np.zeros(len(terms))
    result = weights.tfidf(dense, idf, tf_mode=weights.TF_SUBLINEAR, norm=weights.NORM_L2)
    assert result.tolist() == [0.0, 0.0, 0.0]
    assert np.all(np.isfinite(result))


def test_every_document_transforms_without_error(terms):
    idf = weights.inverse_document_frequency(
        np.array(payload.document_frequencies(terms)), N, weights.IDF_SMOOTHED
    )
    index = payload.term_index(terms)
    for tokens in DOCUMENTS:
        vector = vectorize(tokens, index, (1, 2), smiles_length=len(tokens))
        dense = np.zeros(len(terms))
        dense[vector.indices] = vector.counts
        result = weights.tfidf(dense, idf, tf_mode=weights.TF_SUBLINEAR, norm=weights.NORM_L2)
        assert float(np.linalg.norm(result)) == pytest.approx(1.0)
