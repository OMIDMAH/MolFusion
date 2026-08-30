import json

import pytest

from molfusion_backend.tfidf.contract import INDEX_ORDER, MAX_FEATURES, MIN_DF, NGRAM_ORDERS
from molfusion_backend.tfidf.errors import TfidfVocabularyError
from molfusion_backend.tfidf.vocabulary import (
    VOCABULARY_SCHEMA_VERSION,
    Vocabulary,
    VocabularyEntry,
    composition_by_order,
    parse_vocabulary,
    select_vocabulary,
    validate_vocabulary,
    vocabulary_bytes,
    vocabulary_payload,
)

# Constructed so selection order and index order disagree: ("Z",) is the
# most frequent term but sorts last, and two terms tie at the cap.
COUNTS = {
    ("Z",): 100,
    ("M",): 50,
    ("A", "Z"): 50,
    ("A",): 10,
    ("Z", "A"): 7,
    ("Q",): 3,
    ("Q", "Q"): 2,
}


def entry(index, tokens, df, rank=0):
    return VocabularyEntry(
        index=index, tokens=tokens, order=len(tokens), document_frequency=df, selection_rank=rank
    )


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_min_df_is_applied_before_the_cap():
    """Pruning first is what makes min_df a guarantee rather than a
    suggestion the cap could override."""
    vocabulary, boundary = select_vocabulary(COUNTS, min_df=5, max_features=100)
    assert {e.tokens for e in vocabulary.entries} == {
        ("Z",),
        ("M",),
        ("A", "Z"),
        ("A",),
        ("Z", "A"),
    }
    assert all(e.document_frequency >= 5 for e in vocabulary.entries)
    assert boundary["eligible_terms_at_min_df"] == 5


def test_cap_keeps_the_highest_document_frequency_terms():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=1, max_features=3)
    by_rank = sorted(vocabulary.entries, key=lambda e: e.selection_rank)
    assert [e.tokens for e in by_rank] == [("Z",), ("A", "Z"), ("M",)]


def test_ties_are_broken_by_ascending_lexicographic_tuple():
    """("A","Z") and ("M",) both have df 50; the tuple decides, and
    ("A","Z") < ("M",)."""
    vocabulary, _ = select_vocabulary(COUNTS, min_df=1, max_features=2)
    by_rank = sorted(vocabulary.entries, key=lambda e: e.selection_rank)
    assert [e.tokens for e in by_rank] == [("Z",), ("A", "Z")]


def test_boundary_evidence_records_a_tie_split_by_the_cap():
    vocabulary, boundary = select_vocabulary(COUNTS, min_df=1, max_features=2)
    assert boundary["boundary_document_frequency"] == 50
    assert boundary["last_selected_ngram"] == ["A", "Z"]
    assert boundary["first_excluded_ngram"] == ["M"]
    assert boundary["first_excluded_document_frequency"] == 50
    assert boundary["terms_tied_at_boundary_df"] == 2
    assert boundary["tied_terms_selected"] == 1
    assert boundary["tied_terms_excluded"] == 1
    assert boundary["cap_is_binding"] is True


def test_boundary_reports_a_non_binding_cap():
    _, boundary = select_vocabulary(COUNTS, min_df=1, max_features=100)
    assert boundary["cap_is_binding"] is False
    assert boundary["first_excluded_ngram"] is None
    assert boundary["first_excluded_document_frequency"] is None


def test_selection_is_independent_of_mapping_iteration_order():
    forward, _ = select_vocabulary(COUNTS, min_df=1, max_features=4)
    reversed_counts = dict(reversed(list(COUNTS.items())))
    backward, _ = select_vocabulary(reversed_counts, min_df=1, max_features=4)
    assert [e.tokens for e in forward.entries] == [e.tokens for e in backward.entries]
    assert [e.selection_rank for e in forward.entries] == [
        e.selection_rank for e in backward.entries
    ]


def test_selected_terms_are_indexed_lexicographically_not_by_rank():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    assert [e.tokens for e in vocabulary.entries] == [
        ("A",),
        ("A", "Z"),
        ("M",),
        ("Z",),
        ("Z", "A"),
    ]
    # The most frequent term is not at index 0 -- indexing is deliberately
    # independent of document frequency.
    by_token = {e.tokens: e for e in vocabulary.entries}
    assert by_token[("Z",)].index == 3
    assert by_token[("Z",)].selection_rank == 0


def test_selection_rank_is_preserved_for_every_entry():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    assert sorted(e.selection_rank for e in vocabulary.entries) == [0, 1, 2, 3, 4]


def test_document_frequencies_are_returned_in_index_order():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    assert vocabulary.document_frequencies() == [10, 50, 50, 100, 7]


def test_index_map_covers_every_term():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    index_map = vocabulary.index_map()
    assert len(index_map) == vocabulary.dimension
    assert index_map[("A",)] == 0
    assert ("Q",) not in index_map


def test_selection_rejects_impossible_parameters():
    with pytest.raises(TfidfVocabularyError):
        select_vocabulary(COUNTS, min_df=0, max_features=10)
    with pytest.raises(TfidfVocabularyError):
        select_vocabulary(COUNTS, min_df=5, max_features=0)


def test_selection_rejects_a_corpus_where_nothing_survives():
    with pytest.raises(TfidfVocabularyError, match="no n-gram survived"):
        select_vocabulary(COUNTS, min_df=1000, max_features=10)


def test_composition_by_order():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    assert composition_by_order(vocabulary) == {"1": 3, "2": 2}


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_payload_carries_the_required_fields():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    payload = vocabulary_payload(vocabulary)
    assert payload["schema_version"] == VOCABULARY_SCHEMA_VERSION
    assert payload["dimension"] == 5
    assert payload["index_order"] == INDEX_ORDER
    assert payload["ngram_orders"] == list(NGRAM_ORDERS)
    first = payload["entries"][0]
    assert set(first) >= {"index", "tokens", "order", "document_frequency"}
    assert first["tokens"] == ["A"]


def test_tokens_are_arrays_never_joined_strings():
    """("Cl","C") and ("C","lC") concatenate identically; the payload must
    keep them distinguishable."""
    vocabulary = Vocabulary((entry(0, ("C", "lC"), 9), entry(1, ("Cl", "C"), 9)))
    payload = vocabulary_payload(vocabulary)
    assert payload["entries"][0]["tokens"] == ["C", "lC"]
    assert payload["entries"][1]["tokens"] == ["Cl", "C"]
    assert payload["entries"][0]["tokens"] != payload["entries"][1]["tokens"]


def test_bytes_are_utf8_lf_with_a_final_newline():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    raw = vocabulary_bytes(vocabulary)
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    assert json.loads(raw.decode("utf-8"))["dimension"] == 5


def test_bytes_are_deterministic():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    other, _ = select_vocabulary(dict(reversed(list(COUNTS.items()))), min_df=5, max_features=100)
    assert vocabulary_bytes(vocabulary) == vocabulary_bytes(other)


def test_round_trip_preserves_every_entry():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    restored = parse_vocabulary(json.loads(vocabulary_bytes(vocabulary).decode("utf-8")))
    assert restored == vocabulary


def test_feature_names_are_lossless_and_stable():
    vocabulary = Vocabulary((entry(0, ("C", "lC"), 9), entry(1, ("Cl", "C"), 9)))
    assert vocabulary.feature_names() == ['["C", "lC"]', '["Cl", "C"]']


# ---------------------------------------------------------------------------
# parsing rejects malformed payloads
# ---------------------------------------------------------------------------


def _payload(entries, dimension=None, schema_version=VOCABULARY_SCHEMA_VERSION):
    return {
        "schema_version": schema_version,
        "dimension": len(entries) if dimension is None else dimension,
        "entries": entries,
    }


def _record(index=0, tokens=("C",), order=None, df=10):
    return {
        "index": index,
        "tokens": list(tokens),
        "order": len(tokens) if order is None else order,
        "document_frequency": df,
    }


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("not an object", id="not-an-object"),
        pytest.param(_payload([_record()], schema_version=99), id="wrong-schema-version"),
        pytest.param(_payload([]), id="no-entries"),
        pytest.param(_payload([_record()], dimension=7), id="dimension-mismatch"),
        pytest.param(_payload(["nope"]), id="entry-not-an-object"),
        pytest.param(_payload([_record(tokens=())]), id="empty-tokens"),
        pytest.param(_payload([{**_record(), "tokens": "CC"}]), id="tokens-as-string"),
        pytest.param(_payload([{**_record(), "tokens": [1, 2]}]), id="tokens-not-strings"),
        pytest.param(_payload([{**_record(), "index": "0"}]), id="index-not-int"),
        pytest.param(_payload([_record(tokens=("C", "O"), order=3)]), id="order-disagrees"),
        pytest.param(_payload([{**_record(), "document_frequency": "10"}]), id="df-not-int"),
    ],
)
def test_malformed_payloads_are_rejected(payload):
    with pytest.raises(TfidfVocabularyError):
        parse_vocabulary(payload)


# ---------------------------------------------------------------------------
# semantic validation
# ---------------------------------------------------------------------------


def test_validation_accepts_a_well_formed_vocabulary():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    validate_vocabulary(vocabulary, dimension=5, min_df=5, orders=(1, 2, 3))


def test_validation_rejects_a_wrong_dimension():
    vocabulary, _ = select_vocabulary(COUNTS, min_df=5, max_features=100)
    with pytest.raises(TfidfVocabularyError, match="expected 4096"):
        validate_vocabulary(vocabulary, dimension=4096)


def test_validation_rejects_duplicate_indices():
    vocabulary = Vocabulary((entry(0, ("A",), 9), entry(0, ("B",), 9)))
    with pytest.raises(TfidfVocabularyError, match="duplicate indices"):
        validate_vocabulary(vocabulary)


def test_validation_rejects_an_index_gap():
    vocabulary = Vocabulary((entry(0, ("A",), 9), entry(2, ("B",), 9)))
    with pytest.raises(TfidfVocabularyError, match="contiguous range"):
        validate_vocabulary(vocabulary)


def test_validation_rejects_duplicate_ngrams():
    vocabulary = Vocabulary((entry(0, ("A",), 9), entry(1, ("A",), 9)))
    with pytest.raises(TfidfVocabularyError, match="duplicate n-grams"):
        validate_vocabulary(vocabulary)


def test_validation_rejects_non_lexicographic_index_order():
    vocabulary = Vocabulary((entry(0, ("B",), 9), entry(1, ("A",), 9)))
    with pytest.raises(TfidfVocabularyError, match="lexicographic"):
        validate_vocabulary(vocabulary)


def test_validation_rejects_a_term_below_min_df():
    vocabulary = Vocabulary((entry(0, ("A",), 4),))
    with pytest.raises(TfidfVocabularyError, match="below min_df"):
        validate_vocabulary(vocabulary, min_df=5)


def test_validation_rejects_an_order_outside_the_contract():
    vocabulary = Vocabulary((entry(0, ("A", "B", "C", "D"), 9),))
    with pytest.raises(TfidfVocabularyError, match="outside"):
        validate_vocabulary(vocabulary, orders=(1, 2, 3))


def test_validation_rejects_an_order_that_disagrees_with_its_tokens():
    vocabulary = Vocabulary(
        (VocabularyEntry(index=0, tokens=("A", "B"), order=1, document_frequency=9, selection_rank=0),)
    )
    with pytest.raises(TfidfVocabularyError, match="disagrees"):
        validate_vocabulary(vocabulary)


def test_frozen_selection_constants_are_pinned():
    assert MIN_DF == 5
    assert MAX_FEATURES == 4096
    assert NGRAM_ORDERS == (1, 2, 3)
    assert INDEX_ORDER == "lexicographic_token_tuple_after_selection"
