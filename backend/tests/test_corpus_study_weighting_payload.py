import json

import pytest

from molfusion_backend.corpus.study.ngrams import NgramEntry, NgramFrequencyAccumulator
from molfusion_backend.corpus.study.weighting.payload import (
    FROZEN_DIMENSION,
    FROZEN_MIN_DF,
    FROZEN_NGRAM_ORDERS,
    INDEX_ORDER_LEXICOGRAPHIC,
    INDEX_ORDER_RANKING,
    VOCABULARY_PAYLOAD_SCHEMA,
    assign_indices,
    decode_tokens,
    document_frequencies,
    encode_tokens,
    index_ordering_definition,
    payload_schema,
    select_terms,
    selection_definition,
    term_index,
)


def entry(ngram, df):
    return NgramEntry(
        ngram=ngram,
        order=len(ngram),
        document_frequency=df,
        term_frequency=df,
        document_frequency_fit=df,
        term_frequency_fit=df,
        document_frequency_holdout=0,
        term_frequency_holdout=0,
        document_frequency_fit_bands=(df, 0, 0, 0, 0, 0),
        term_frequency_fit_bands=(df, 0, 0, 0, 0, 0),
    )


# Constructed so ranking order and lexicographic order disagree: ("Z",)
# is the most frequent term but sorts last.
SAMPLE = {
    1: [entry(("A",), 10), entry(("M",), 50), entry(("Z",), 100), entry(("Q",), 3)],
    2: [entry(("A", "Z"), 50), entry(("Z", "A"), 7), entry(("Q", "Q"), 2)],
}


# ---------------------------------------------------------------------------
# the frozen Phase 5F-C constants
# ---------------------------------------------------------------------------


def test_frozen_selection_constants_are_pinned():
    assert FROZEN_NGRAM_ORDERS == (1, 2, 3)
    assert FROZEN_MIN_DF == 5
    assert FROZEN_DIMENSION == 4096


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_min_df_is_applied_before_the_cap():
    """Pruning first is what makes min_df a guarantee rather than a
    suggestion the cap could override."""
    selected = select_terms(SAMPLE, min_df=5, dimension=100)
    assert {e.ngram for e in selected} == {("A",), ("M",), ("Z",), ("A", "Z"), ("Z", "A")}
    assert all(e.document_frequency >= 5 for e in selected)


def test_cap_keeps_the_highest_document_frequency_terms():
    selected = select_terms(SAMPLE, min_df=1, dimension=3)
    assert [e.ngram for e in selected] == [("Z",), ("A", "Z"), ("M",)]


def test_ties_are_broken_lexicographically_during_selection():
    """("A","Z") and ("M",) both have df 50; the tuple decides."""
    selected = select_terms(SAMPLE, min_df=50, dimension=10)
    assert [e.ngram for e in selected] == [("Z",), ("A", "Z"), ("M",)]


def test_selection_respects_the_requested_orders():
    selected = select_terms(SAMPLE, min_df=1, dimension=100, orders=(1,))
    assert all(e.order == 1 for e in selected)


def test_a_cap_larger_than_the_eligible_set_keeps_everything():
    selected = select_terms(SAMPLE, min_df=1, dimension=1000)
    assert len(selected) == 7


def test_selection_does_not_depend_on_input_order():
    reversed_sample = {order: list(reversed(v)) for order, v in SAMPLE.items()}
    assert [e.ngram for e in select_terms(SAMPLE, min_df=1, dimension=5)] == [
        e.ngram for e in select_terms(reversed_sample, min_df=1, dimension=5)
    ]


# ---------------------------------------------------------------------------
# index ordering
# ---------------------------------------------------------------------------


def test_lexicographic_indexing_is_independent_of_document_frequency():
    terms = assign_indices(select_terms(SAMPLE, min_df=5, dimension=10),
                           order=INDEX_ORDER_LEXICOGRAPHIC)
    assert [t.tokens for t in terms] == [
        ("A",),
        ("A", "Z"),
        ("M",),
        ("Z",),
        ("Z", "A"),
    ]
    assert [t.index for t in terms] == [0, 1, 2, 3, 4]


def test_ranking_indexing_puts_the_most_frequent_term_first():
    terms = assign_indices(select_terms(SAMPLE, min_df=5, dimension=10),
                           order=INDEX_ORDER_RANKING)
    assert [t.tokens for t in terms] == [
        ("Z",),
        ("A", "Z"),
        ("M",),
        ("A",),
        ("Z", "A"),
    ]


def test_the_two_orderings_genuinely_differ_on_this_fixture():
    selected = select_terms(SAMPLE, min_df=5, dimension=10)
    lexicographic = [t.tokens for t in assign_indices(selected, order=INDEX_ORDER_LEXICOGRAPHIC)]
    ranked = [t.tokens for t in assign_indices(selected, order=INDEX_ORDER_RANKING)]
    assert lexicographic != ranked
    assert set(lexicographic) == set(ranked)


def test_selection_rank_survives_reindexing():
    """The ranking that chose a feature stays auditable after the ordering
    that positions it has been applied."""
    terms = assign_indices(select_terms(SAMPLE, min_df=5, dimension=10),
                           order=INDEX_ORDER_LEXICOGRAPHIC)
    by_token = {t.tokens: t for t in terms}
    assert by_token[("Z",)].selection_rank == 0
    assert by_token[("Z",)].index == 3
    assert sorted(t.selection_rank for t in terms) == [0, 1, 2, 3, 4]


def test_indices_are_contiguous_from_zero():
    terms = assign_indices(select_terms(SAMPLE, min_df=1, dimension=100))
    assert [t.index for t in terms] == list(range(len(terms)))


def test_document_frequencies_are_returned_in_index_order():
    terms = assign_indices(select_terms(SAMPLE, min_df=5, dimension=10),
                           order=INDEX_ORDER_LEXICOGRAPHIC)
    assert document_frequencies(terms) == [10, 50, 50, 100, 7]


def test_term_index_maps_tokens_to_columns():
    terms = assign_indices(select_terms(SAMPLE, min_df=5, dimension=10),
                           order=INDEX_ORDER_LEXICOGRAPHIC)
    index = term_index(terms)
    assert index[("A",)] == 0
    assert index[("Z", "A")] == 4
    assert ("Q",) not in index


def test_unknown_index_ordering_is_rejected():
    with pytest.raises(ValueError):
        assign_indices(select_terms(SAMPLE, min_df=1, dimension=3), order="insertion")


def test_indexing_is_deterministic_across_repeated_calls():
    selected = select_terms(SAMPLE, min_df=1, dimension=100)
    first = [t.tokens for t in assign_indices(selected)]
    second = [t.tokens for t in assign_indices(list(reversed(selected)))]
    assert first == second


# ---------------------------------------------------------------------------
# lossless serialization
# ---------------------------------------------------------------------------


def test_tokens_are_encoded_as_a_json_array():
    assert encode_tokens(("C", "(", "=")) == '["C", "(", "="]'


def test_encoding_round_trips_exactly():
    for tokens in (("C",), ("Cl", "C"), ("C", "lC"), ("[C@@H]", "(", "[nH]"), ("%12",)):
        assert decode_tokens(encode_tokens(tokens)) == tokens


def test_the_ambiguous_pair_stays_distinguishable():
    """("Cl","C") and ("C","lC") both concatenate to "ClC"; the encoding
    must keep them apart, which is the reason it is not a join."""
    left = encode_tokens(("Cl", "C"))
    right = encode_tokens(("C", "lC"))
    assert left != right
    assert decode_tokens(left) != decode_tokens(right)
    assert "".join(("Cl", "C")) == "".join(("C", "lC"))


def test_encoding_preserves_a_token_containing_a_separator_character():
    """A whitespace- or comma-joined key would break on these; a JSON array
    does not."""
    for tokens in (("C", " "), ("C", ","), ("C", '"'), ("C", "]")):
        assert decode_tokens(encode_tokens(tokens)) == tokens


def test_decoding_rejects_a_non_array_payload():
    with pytest.raises(ValueError):
        decode_tokens('"CC"')
    with pytest.raises(ValueError):
        decode_tokens("[1, 2]")


def test_encoded_tokens_are_valid_json():
    assert json.loads(encode_tokens(("C", "(", "="))) == ["C", "(", "="]


# ---------------------------------------------------------------------------
# reported definitions
# ---------------------------------------------------------------------------


def test_payload_schema_states_the_invariants_and_the_record_shape():
    schema = payload_schema()
    assert schema["schema"] == VOCABULARY_PAYLOAD_SCHEMA
    assert set(schema["term_record"]) == {
        "index",
        "tokens",
        "order",
        "document_frequency",
    }
    assert schema["example"]["tokens"] == ["C", "(", "="]
    assert "fit_corpus_sha256" in schema["required_header_fields"]
    assert "min_df" in schema["required_header_fields"]
    assert "dimension" in schema["required_header_fields"]
    assert any("0..dimension-1" in rule for rule in schema["invariants"])


def test_selection_definition_names_the_binding_constraint():
    definition = selection_definition(5, 4096, (1, 2, 3))
    assert definition["min_df"] == 5
    assert definition["min_df_units"] == "absolute number of full-corpus documents"
    assert definition["dimension"] == 4096
    assert definition["selection_ranking"] == "(-document_frequency, ngram_tuple)"
    assert definition["sklearn_max_features_used"] is False
    assert "dimension cap" in definition["binding_constraint"]


def test_index_ordering_definition_disclaims_implicit_sources():
    definition = index_ordering_definition(INDEX_ORDER_LEXICOGRAPHIC)
    assert definition["selection_is_separate_from_indexing"] is True
    assert definition["depends_on_dict_insertion_order"] is False
    assert definition["depends_on_sklearn_vocabulary_construction"] is False


def test_selection_from_a_real_accumulator():
    accumulator = NgramFrequencyAccumulator((1, 2))
    for _ in range(6):
        accumulator.add_document(("C", "C", "O"), holdout=False)
    accumulator.add_document(("N",), holdout=False)

    entries = {order: accumulator.entries(order) for order in (1, 2)}
    terms = assign_indices(select_terms(entries, min_df=5, dimension=10))
    # ("N",) has df 1 and is pruned; the rest survive.
    assert [t.tokens for t in terms] == [("C",), ("C", "C"), ("C", "O"), ("O",)]
    assert document_frequencies(terms) == [6, 6, 6, 6]
