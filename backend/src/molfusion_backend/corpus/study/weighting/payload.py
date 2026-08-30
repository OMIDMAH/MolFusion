"""Vocabulary selection, index ordering, and lossless serialization.

Three separate decisions live here, and conflating them is the mistake
this module exists to prevent:

    selection  which 4,096 n-grams are features        (frozen in 5F-C)
    ordering   which vector column each one occupies   (frozen here)
    encoding   how a feature is written to disk        (frozen here)

Selection is a scientific choice about coverage. Ordering is an interface
choice: it decides what column 37 of every vector MolFusion ever emits
means, so it has to be stated rather than inherited from whatever order a
dict happened to have. Encoding is a correctness choice, because the
obvious encoding is lossy.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from molfusion_backend.corpus.study.ngrams import Ngram, NgramEntry
from molfusion_backend.corpus.study.vocabulary import (
    RANKING_DOCUMENT_FREQUENCY,
    SCOPE_CORPUS,
    frequency,
    rank_entries,
)

# The Phase 5F-C decision, restated as executable constants. Both are
# recorded because only one of them binds: min_df = 5 leaves ~9,383
# eligible (1,3) terms on ChEMBL 37, and the 4,096 cap then does the real
# pruning. Metadata that showed only min_df would imply the dimension was
# a consequence of the rarity floor, which it is not.
FROZEN_NGRAM_ORDERS = (1, 2, 3)
FROZEN_MIN_DF = 5
FROZEN_DIMENSION = 4096

# Index ordering. Selection ranks by document frequency; indexing does not.
# See `index_ordering_definition()` for why the two are deliberately
# different rules.
INDEX_ORDER_RANKING = "selection_ranking"
INDEX_ORDER_LEXICOGRAPHIC = "lexicographic_token_tuple"
INDEX_ORDERS = (INDEX_ORDER_RANKING, INDEX_ORDER_LEXICOGRAPHIC)

VOCABULARY_PAYLOAD_SCHEMA = "molfusion_ngram_vocabulary_v1"


@dataclass(frozen=True)
class VocabularyTerm:
    """One selected feature, with everything needed to re-derive its weight."""

    index: int
    tokens: Ngram
    order: int
    document_frequency: int
    selection_rank: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tokens": list(self.tokens),
            "order": self.order,
            "document_frequency": self.document_frequency,
        }


def select_terms(
    entries_by_order: dict[int, Sequence[NgramEntry]],
    *,
    min_df: int = FROZEN_MIN_DF,
    dimension: int = FROZEN_DIMENSION,
    orders: Sequence[int] = FROZEN_NGRAM_ORDERS,
    scope: str = SCOPE_CORPUS,
) -> list[NgramEntry]:
    """Apply the frozen Phase 5F-C selection rule, in its frozen order.

    Prune by absolute document frequency first, then rank the survivors by
    `(-document_frequency, ngram_tuple)`, then keep the leading `dimension`
    of them. Pruning first is not merely tidier: it is what makes `min_df`
    a real guarantee rather than a suggestion the cap could override.

    Scope defaults to the whole corpus, because the production vocabulary
    is fitted on all 2,897,639 molecules -- the Phase 5F-C holdout was an
    analysis device and has no role here.
    """
    wanted = set(orders)
    eligible = [
        entry
        for order in sorted(entries_by_order)
        if order in wanted
        for entry in entries_by_order[order]
        if frequency(entry, RANKING_DOCUMENT_FREQUENCY, scope) >= min_df
    ]
    ranked = rank_entries(eligible, RANKING_DOCUMENT_FREQUENCY, scope)
    return ranked[:dimension]


def assign_indices(
    selected: Sequence[NgramEntry],
    *,
    order: str = INDEX_ORDER_LEXICOGRAPHIC,
    scope: str = SCOPE_CORPUS,
) -> list[VocabularyTerm]:
    """Turn a selection into indexed vocabulary terms.

    `selection_rank` is retained on every term regardless of index order,
    so the ranking that chose a feature stays auditable after the ordering
    that positions it has been applied.
    """
    ranks = {entry.ngram: rank for rank, entry in enumerate(selected)}

    if order == INDEX_ORDER_RANKING:
        ordered = list(selected)
    elif order == INDEX_ORDER_LEXICOGRAPHIC:
        ordered = sorted(selected, key=lambda entry: entry.ngram)
    else:
        raise ValueError(f"unknown index ordering: {order!r}")

    return [
        VocabularyTerm(
            index=index,
            tokens=entry.ngram,
            order=entry.order,
            document_frequency=frequency(entry, RANKING_DOCUMENT_FREQUENCY, scope),
            selection_rank=ranks[entry.ngram],
        )
        for index, entry in enumerate(ordered)
    ]


def term_index(terms: Sequence[VocabularyTerm]) -> dict[Ngram, int]:
    """Lookup table from n-gram tuple to vector column."""
    return {term.tokens: term.index for term in terms}


def document_frequencies(terms: Sequence[VocabularyTerm]) -> list[int]:
    """DF per vector column, in index order -- the shape an IDF vector needs."""
    return [term.document_frequency for term in terms]


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def encode_tokens(tokens: Ngram) -> str:
    """A JSON array of token strings: the lossless n-gram encoding.

    The tempting alternative -- joining tokens into one string -- is
    genuinely lossy for this tokenizer, because a multi-character token can
    be re-split at a different boundary. ("Cl","C") and ("C","lC") both
    join to "ClC", so a concatenated key cannot distinguish two different
    features, and a whitespace separator only moves the problem to any
    token that could contain a space. A JSON array has no separator to
    collide with and round-trips exactly.
    """
    return json.dumps(list(tokens), ensure_ascii=False)


def decode_tokens(encoded: str) -> Ngram:
    decoded = json.loads(encoded)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise ValueError(f"not a token array: {encoded!r}")
    return tuple(decoded)


def payload_schema() -> dict[str, Any]:
    """The vocabulary payload shape Phase 5F-D should write.

    Described rather than written: this phase freezes the semantics, and
    the file itself is Phase 5F-D's to produce.
    """
    return {
        "schema": VOCABULARY_PAYLOAD_SCHEMA,
        "term_record": {
            "index": "int, 0-based vector column",
            "tokens": "array of token strings, lossless; never a joined string",
            "order": "int, len(tokens); redundant but makes the file self-checking",
            "document_frequency": "int, full-corpus DF used to derive this term's IDF",
        },
        "example": {
            "index": 123,
            "tokens": ["C", "(", "="],
            "order": 3,
            "document_frequency": 1048576,
        },
        "required_header_fields": [
            "fit_corpus_sha256",
            "document_count",
            "normalization_id",
            "tokenizer_id",
            "serialization_id",
            "ngram_orders",
            "min_df",
            "dimension",
            "selection_ranking",
            "index_ordering",
            "weighting",
        ],
        "invariants": [
            "len(terms) == dimension",
            "indices are exactly 0..dimension-1 with no gaps or repeats",
            "token arrays are unique",
            "every document_frequency >= min_df",
            "the IDF vector is ordered by index, not by selection rank",
        ],
    }


def index_ordering_definition(order: str) -> dict[str, Any]:
    return {
        "index_ordering": order,
        "rule": (
            "select the top `dimension` terms by (-document_frequency, ngram_tuple), "
            "then assign vector indices by ascending lexicographic token tuple"
            if order == INDEX_ORDER_LEXICOGRAPHIC
            else "vector index equals selection rank"
        ),
        "selection_is_separate_from_indexing": True,
        "depends_on_dict_insertion_order": False,
        "depends_on_sklearn_vocabulary_construction": False,
    }


def selection_definition(min_df: int, dimension: int, orders: Sequence[int]) -> dict[str, Any]:
    return {
        "ngram_orders": list(orders),
        "min_df": min_df,
        "min_df_units": "absolute number of full-corpus documents",
        "dimension": dimension,
        "selection_ranking": "(-document_frequency, ngram_tuple)",
        "steps": [
            "generate token n-grams of the given orders",
            f"drop terms with full-corpus document frequency < {min_df}",
            "sort survivors by (-document_frequency, ngram_tuple)",
            f"keep the first {dimension}",
        ],
        "binding_constraint": (
            "the dimension cap; min_df alone leaves substantially more than "
            "`dimension` eligible terms on this corpus"
        ),
        "vocabulary_owner": "MolFusion",
        "sklearn_max_features_used": False,
    }


__all__ = [
    "FROZEN_DIMENSION",
    "FROZEN_MIN_DF",
    "FROZEN_NGRAM_ORDERS",
    "INDEX_ORDERS",
    "INDEX_ORDER_LEXICOGRAPHIC",
    "INDEX_ORDER_RANKING",
    "VOCABULARY_PAYLOAD_SCHEMA",
    "VocabularyTerm",
    "assign_indices",
    "decode_tokens",
    "document_frequencies",
    "encode_tokens",
    "index_ordering_definition",
    "payload_schema",
    "select_terms",
    "selection_definition",
    "term_index",
]
