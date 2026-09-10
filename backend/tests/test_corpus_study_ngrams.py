import pytest

from molfusion_backend.corpus.study.ngrams import (
    BAND_COUNT,
    TOKEN_COUNT_BAND_EDGES,
    NgramFrequencyAccumulator,
    band_labels,
    document_ngram_counts,
    iter_ngrams,
    token_count_band,
)
from molfusion_backend.smiles_tokenizer import tokenize_smiles

# The worked example from the phase brief.
ACETIC_ACID_TOKENS = ("C", "C", "(", "=", "O", ")", "O")


# ---------------------------------------------------------------------------
# n-gram generation
# ---------------------------------------------------------------------------


def test_unigrams_are_exact():
    assert list(iter_ngrams(ACETIC_ACID_TOKENS, 1)) == [
        ("C",),
        ("C",),
        ("(",),
        ("=",),
        ("O",),
        (")",),
        ("O",),
    ]


def test_bigrams_are_exact():
    assert list(iter_ngrams(ACETIC_ACID_TOKENS, 2)) == [
        ("C", "C"),
        ("C", "("),
        ("(", "="),
        ("=", "O"),
        ("O", ")"),
        (")", "O"),
    ]


def test_trigrams_are_exact():
    assert list(iter_ngrams(ACETIC_ACID_TOKENS, 3)) == [
        ("C", "C", "("),
        ("C", "(", "="),
        ("(", "=", "O"),
        ("=", "O", ")"),
        ("O", ")", "O"),
    ]


def test_ngram_count_is_length_minus_order_plus_one():
    for order in (1, 2, 3):
        assert len(list(iter_ngrams(ACETIC_ACID_TOKENS, order))) == 7 - order + 1


def test_molecule_shorter_than_the_order_yields_nothing():
    """No padding, no synthetic n-grams: a two-token molecule has no
    trigram, and inventing one would put a feature in a vector that the
    molecule does not contain."""
    assert list(iter_ngrams(("C", "O"), 3)) == []
    assert document_ngram_counts(("C", "O"), 3) == {}


def test_order_must_be_positive():
    with pytest.raises(ValueError):
        list(iter_ngrams(ACETIC_ACID_TOKENS, 0))
    with pytest.raises(ValueError):
        document_ngram_counts(ACETIC_ACID_TOKENS, -1)


def test_multi_character_tokens_are_never_re_split():
    """("Cl", "C") must never collide with ("C", "lC").

    Both concatenate to "ClC", so a string-keyed counter would merge them.
    Tuple keys keep them distinct, which is the whole reason the study
    never concatenates.
    """
    chloro = document_ngram_counts(tokenize_smiles("ClC"), 2)
    assert chloro == {("Cl", "C"): 1}
    assert ("C", "lC") not in chloro
    assert list(chloro)[0] != ("C", "lC")


def test_bracket_atoms_stay_whole_inside_ngrams():
    tokens = tokenize_smiles("C[C@@H](N)O")
    assert ("C", "[C@@H]") in document_ngram_counts(tokens, 2)


# ---------------------------------------------------------------------------
# DF vs TF
# ---------------------------------------------------------------------------


def test_repeated_ngram_in_one_molecule_increments_df_once():
    accumulator = NgramFrequencyAccumulator((1,))
    accumulator.add_document(("C",) * 20, holdout=False)

    entry = accumulator.entries(1)[0]
    assert entry.ngram == ("C",)
    assert entry.document_frequency == 1
    assert entry.term_frequency == 20


def test_repeated_occurrence_increments_tf_many_times():
    accumulator = NgramFrequencyAccumulator((2,))
    accumulator.add_document(("C", "C", "C", "C"), holdout=False)

    entry = accumulator.entries(2)[0]
    assert entry.ngram == ("C", "C")
    assert entry.document_frequency == 1
    assert entry.term_frequency == 3


def test_df_and_tf_accumulate_across_molecules_independently():
    accumulator = NgramFrequencyAccumulator((1,))
    accumulator.add_document(("C", "C", "C"), holdout=False)
    accumulator.add_document(("C", "O"), holdout=False)
    accumulator.add_document(("O",), holdout=False)

    by_ngram = {entry.ngram: entry for entry in accumulator.entries(1)}
    assert by_ngram[("C",)].document_frequency == 2
    assert by_ngram[("C",)].term_frequency == 4
    assert by_ngram[("O",)].document_frequency == 2
    assert by_ngram[("O",)].term_frequency == 2


def test_fit_and_holdout_counts_are_kept_apart_and_sum_to_the_corpus():
    accumulator = NgramFrequencyAccumulator((1,))
    accumulator.add_document(("C", "C"), holdout=False)
    accumulator.add_document(("C",), holdout=True)

    entry = accumulator.entries(1)[0]
    assert entry.document_frequency_fit == 1
    assert entry.term_frequency_fit == 2
    assert entry.document_frequency_holdout == 1
    assert entry.term_frequency_holdout == 1
    assert entry.document_frequency == 2
    assert entry.term_frequency == 3
    assert accumulator.fit_document_count == 1
    assert accumulator.holdout_document_count == 1
    assert accumulator.document_count == 2


def test_token_totals_are_tracked_per_subset():
    accumulator = NgramFrequencyAccumulator((1,))
    accumulator.add_document(("C", "C", "O"), holdout=False)
    accumulator.add_document(("N",), holdout=True)

    assert accumulator.token_count(holdout=False) == 3
    assert accumulator.token_count(holdout=True) == 1


def test_entries_are_sorted_by_token_tuple_not_insertion_order():
    accumulator = NgramFrequencyAccumulator((1,))
    for token in ("O", "C", "N", "[nH]", "Cl"):
        accumulator.add_document((token,), holdout=False)

    assert [entry.ngram for entry in accumulator.entries(1)] == sorted(
        [("O",), ("C",), ("N",), ("[nH]",), ("Cl",)]
    )


def test_counts_do_not_depend_on_document_order():
    documents = [("C", "C", "O"), ("C", "N"), ("O", "O", "O"), ("N",)]

    forward = NgramFrequencyAccumulator((1, 2))
    for tokens in documents:
        forward.add_document(tokens, holdout=False)
    backward = NgramFrequencyAccumulator((1, 2))
    for tokens in reversed(documents):
        backward.add_document(tokens, holdout=False)

    for order in (1, 2):
        assert forward.entries(order) == backward.entries(order)


# ---------------------------------------------------------------------------
# length bands
# ---------------------------------------------------------------------------


def test_token_count_band_edges_are_upper_inclusive():
    assert token_count_band(1) == 0
    assert token_count_band(32) == 0
    assert token_count_band(33) == 1
    assert token_count_band(512) == BAND_COUNT - 2
    assert token_count_band(513) == BAND_COUNT - 1
    assert token_count_band(1617) == BAND_COUNT - 1


def test_band_labels_describe_every_band():
    labels = band_labels()
    assert len(labels) == BAND_COUNT
    assert labels[0] == f"(0,{TOKEN_COUNT_BAND_EDGES[0]}]"
    assert labels[-1].endswith("inf)")


def test_band_restricted_tf_recovers_short_molecule_counts():
    """The long-molecule diagnostic must be re-derivable without a second
    pass over the corpus."""
    accumulator = NgramFrequencyAccumulator((1,))
    accumulator.add_document(("C",) * 10, holdout=False)  # band 0
    accumulator.add_document(("C",) * 600, holdout=False)  # band 5

    entry = accumulator.entries(1)[0]
    assert entry.term_frequency_fit == 610
    assert entry.term_frequency_fit_up_to_band(0) == 10
    assert entry.term_frequency_fit_up_to_band(3) == 10
    assert entry.term_frequency_fit_up_to_band(BAND_COUNT - 1) == 610
    assert entry.document_frequency_fit_up_to_band(0) == 1
    assert entry.document_frequency_fit_up_to_band(BAND_COUNT - 1) == 2


def test_documents_by_band_counts_molecules_not_occurrences():
    accumulator = NgramFrequencyAccumulator((1,))
    accumulator.add_document(("C",) * 10, holdout=False)
    accumulator.add_document(("C",) * 40, holdout=False)
    accumulator.add_document(("C",) * 40, holdout=True)

    assert accumulator.documents_by_band(holdout=False)[:2] == (1, 1)
    assert accumulator.documents_by_band(holdout=True)[:2] == (0, 1)
