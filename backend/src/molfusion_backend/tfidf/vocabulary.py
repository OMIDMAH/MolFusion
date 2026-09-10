"""Vocabulary selection, indexing, serialization and validation.

Three separate decisions live here, and conflating them is the mistake
this module exists to prevent:

    selection  which n-grams are features   -- by document frequency
    indexing   which column each occupies   -- lexicographic, afterwards
    encoding   how a feature is written     -- a JSON array of tokens

Selection is a scientific choice; indexing is an interface choice that
fixes what column 37 of every vector MolFusion ever emits means; encoding
is a correctness choice, because the obvious encoding is lossy.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from molfusion_backend.tfidf.contract import (
    INDEX_ORDER,
    MAX_FEATURES,
    MIN_DF,
    NGRAM_ORDERS,
    SELECTION_KEY,
)
from molfusion_backend.tfidf.errors import TfidfVocabularyError
from molfusion_backend.tfidf.ngrams import Ngram

VOCABULARY_SCHEMA_VERSION = 1

# Prefix for a per-column feature name; see Vocabulary.feature_names().
FEATURE_NAME_PREFIX = "ngram"


def encode_feature_tokens(tokens: Ngram) -> str:
    """Compact JSON array of tokens, as embedded in a feature name.

    Separators are given explicitly so the encoding does not depend on
    `json.dumps` default spacing: a feature name is part of the API and CSV
    surface, and it should not shift because a default changed.
    """
    return json.dumps(list(tokens), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class VocabularyEntry:
    """One feature: its column, its tokens, and the DF its IDF derives from."""

    index: int
    tokens: Ngram
    order: int
    document_frequency: int
    selection_rank: int

    def as_payload(self) -> dict[str, Any]:
        # Field order here is the field order on disk. Fixed literally so
        # the payload bytes never depend on dict construction order.
        return {
            "index": self.index,
            "tokens": list(self.tokens),
            "order": self.order,
            "document_frequency": self.document_frequency,
            "selection_rank": self.selection_rank,
        }


@dataclass(frozen=True)
class Vocabulary:
    """The frozen feature set, in index order."""

    entries: tuple[VocabularyEntry, ...]

    @property
    def dimension(self) -> int:
        return len(self.entries)

    def index_map(self) -> dict[Ngram, int]:
        """n-gram tuple -> vector column, the lookup a transform needs."""
        return {entry.tokens: entry.index for entry in self.entries}

    def document_frequencies(self) -> list[int]:
        """DF per column, in index order -- the shape an IDF vector needs."""
        return [entry.document_frequency for entry in self.entries]

    def feature_names(self) -> list[str]:
        """Stable, unambiguous per-column feature names.

        Format: `ngram<order>:<compact JSON token array>`, e.g.
        `ngram3:["C","(","="]`.

        The JSON array is what makes a name lossless: ("Cl","C") and
        ("C","lC") join to the same string, so a concatenated name could
        not tell two genuinely different features apart. It round-trips
        through `json.loads` after the prefix is stripped.

        The order prefix is redundant with the array's length and is kept
        because it makes a name self-describing and groups the vocabulary
        by order. Neither part can contain ";", which is what the frontend
        CSV export uses to join names into one column.
        """
        return [
            f"{FEATURE_NAME_PREFIX}{entry.order}:{encode_feature_tokens(entry.tokens)}"
            for entry in self.entries
        ]


def select_vocabulary(
    document_frequency: Mapping[Ngram, int],
    *,
    min_df: int = MIN_DF,
    max_features: int = MAX_FEATURES,
) -> tuple[Vocabulary, dict[str, Any]]:
    """Apply the frozen selection rule and return the vocabulary plus the
    boundary evidence that shows the selection was deterministic.

    Order of operations is fixed and matters: prune by `min_df` **first**,
    then rank, then cap. Pruning first is what makes `min_df` a guarantee
    rather than a suggestion the cap could override.
    """
    if min_df < 1:
        raise TfidfVocabularyError(f"min_df must be >= 1, got {min_df}")
    if max_features < 1:
        raise TfidfVocabularyError(f"max_features must be >= 1, got {max_features}")

    eligible = [(ngram, df) for ngram, df in document_frequency.items() if df >= min_df]
    # The frozen ranking key. The tie-break is the token tuple itself,
    # compared element by element, so the order is total: no two distinct
    # n-grams can tie, and no column is ever awarded by insertion order,
    # hash seed, or sort stability.
    ranked = sorted(eligible, key=lambda item: (-item[1], item[0]))
    selected = ranked[:max_features]

    if not selected:
        raise TfidfVocabularyError(
            f"no n-gram survived min_df={min_df}; the corpus or the counts are wrong"
        )

    ranks = {ngram: rank for rank, (ngram, _) in enumerate(selected)}
    ordered = sorted(selected, key=lambda item: item[0])  # lexicographic index order
    entries = tuple(
        VocabularyEntry(
            index=index,
            tokens=ngram,
            order=len(ngram),
            document_frequency=df,
            selection_rank=ranks[ngram],
        )
        for index, (ngram, df) in enumerate(ordered)
    )

    boundary = _boundary_evidence(ranked, selected, max_features)
    boundary["eligible_terms_at_min_df"] = len(eligible)
    boundary["distinct_ngrams_total"] = len(document_frequency)
    return Vocabulary(entries), boundary


def _boundary_evidence(
    ranked: Sequence[tuple[Ngram, int]],
    selected: Sequence[tuple[Ngram, int]],
    max_features: int,
) -> dict[str, Any]:
    """What happened exactly at the cap.

    A cap that lands in the middle of a group of equal-DF terms is where a
    non-deterministic selection would show itself, so the boundary is
    recorded rather than assumed: how many terms share the boundary DF, how
    many of them were taken, and which n-grams sit on either side of the
    cut.
    """
    last_selected, boundary_df = selected[-1]
    tied = [ngram for ngram, df in ranked if df == boundary_df]
    tied_selected = [ngram for ngram, df in selected if df == boundary_df]

    first_excluded: Ngram | None = None
    first_excluded_df: int | None = None
    if len(ranked) > max_features:
        first_excluded, first_excluded_df = ranked[max_features]

    return {
        "boundary_document_frequency": boundary_df,
        "last_selected_ngram": list(last_selected),
        "first_excluded_ngram": list(first_excluded) if first_excluded is not None else None,
        "first_excluded_document_frequency": first_excluded_df,
        "terms_tied_at_boundary_df": len(tied),
        "tied_terms_selected": len(tied_selected),
        "tied_terms_excluded": len(tied) - len(tied_selected),
        "tie_resolution": (
            "ascending lexicographic n-gram token tuple; the ranking key "
            f"{SELECTION_KEY} is a total order, so the cut is unique"
        ),
        "cap_is_binding": len(ranked) > max_features,
    }


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def vocabulary_payload(vocabulary: Vocabulary) -> dict[str, Any]:
    return {
        "schema_version": VOCABULARY_SCHEMA_VERSION,
        "dimension": vocabulary.dimension,
        "index_order": INDEX_ORDER,
        "selection_key": SELECTION_KEY,
        "ngram_orders": list(NGRAM_ORDERS),
        "entries": [entry.as_payload() for entry in vocabulary.entries],
    }


def vocabulary_bytes(vocabulary: Vocabulary) -> bytes:
    """Deterministic UTF-8/LF JSON with an explicit final newline.

    `sort_keys=False` with literal field order, `ensure_ascii=False` so a
    token is written as itself, `indent=2` for auditability, and raw bytes
    so no platform newline translation can touch it. Two builds of the same
    vocabulary produce identical bytes and therefore an identical SHA-256.
    """
    text = json.dumps(vocabulary_payload(vocabulary), indent=2, sort_keys=False, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def parse_vocabulary(raw: Any) -> Vocabulary:
    """Rebuild a Vocabulary from a parsed payload, or raise.

    Strict on purpose: this is the boundary where a corrupted or
    hand-edited artifact must be caught, and every invariant checked here
    is one that would otherwise silently misalign columns with weights.
    """
    if not isinstance(raw, dict):
        raise TfidfVocabularyError("vocabulary payload must be a JSON object")
    if raw.get("schema_version") != VOCABULARY_SCHEMA_VERSION:
        raise TfidfVocabularyError(
            f"unsupported vocabulary schema_version: {raw.get('schema_version')!r}"
        )

    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise TfidfVocabularyError("vocabulary payload has no entries")

    declared_dimension = raw.get("dimension")
    if declared_dimension != len(raw_entries):
        raise TfidfVocabularyError(
            f"declared dimension {declared_dimension!r} does not match "
            f"{len(raw_entries)} entries"
        )

    entries = []
    for position, record in enumerate(raw_entries):
        if not isinstance(record, dict):
            raise TfidfVocabularyError(f"entry {position} is not an object")
        tokens = record.get("tokens")
        if not isinstance(tokens, list) or not tokens or not all(
            isinstance(token, str) for token in tokens
        ):
            raise TfidfVocabularyError(
                f"entry {position} tokens must be a non-empty array of strings, got {tokens!r}"
            )
        index = record.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TfidfVocabularyError(f"entry {position} has a non-integer index: {index!r}")
        order = record.get("order")
        if order != len(tokens):
            raise TfidfVocabularyError(
                f"entry {position} declares order {order!r} but carries {len(tokens)} tokens"
            )
        document_frequency = record.get("document_frequency")
        if not isinstance(document_frequency, int) or isinstance(document_frequency, bool):
            raise TfidfVocabularyError(
                f"entry {position} has a non-integer document_frequency: {document_frequency!r}"
            )
        selection_rank = record.get("selection_rank", position)
        entries.append(
            VocabularyEntry(
                index=index,
                tokens=tuple(tokens),
                order=order,
                document_frequency=document_frequency,
                selection_rank=selection_rank,
            )
        )

    return Vocabulary(tuple(entries))


def validate_vocabulary(
    vocabulary: Vocabulary,
    *,
    dimension: int | None = None,
    min_df: int = MIN_DF,
    orders: Iterable[int] = NGRAM_ORDERS,
) -> None:
    """Assert every frozen vocabulary invariant, or raise the one that broke."""
    entries = vocabulary.entries
    if dimension is not None and len(entries) != dimension:
        raise TfidfVocabularyError(
            f"vocabulary has {len(entries)} entries, expected {dimension}"
        )

    indices = [entry.index for entry in entries]
    if indices != list(range(len(entries))):
        if len(set(indices)) != len(indices):
            raise TfidfVocabularyError("vocabulary contains duplicate indices")
        raise TfidfVocabularyError(
            "vocabulary indices are not the contiguous range 0..dimension-1"
        )

    tokens = [entry.tokens for entry in entries]
    if len(set(tokens)) != len(tokens):
        raise TfidfVocabularyError("vocabulary contains duplicate n-grams")

    if tokens != sorted(tokens):
        raise TfidfVocabularyError(
            f"vocabulary is not in {INDEX_ORDER} order; index assignment must follow "
            "ascending lexicographic token tuples"
        )

    allowed = set(orders)
    for entry in entries:
        if entry.order != len(entry.tokens):
            raise TfidfVocabularyError(
                f"index {entry.index}: order {entry.order} disagrees with "
                f"{len(entry.tokens)} tokens"
            )
        if entry.order not in allowed:
            raise TfidfVocabularyError(
                f"index {entry.index}: n-gram order {entry.order} outside {sorted(allowed)}"
            )
        if entry.document_frequency < min_df:
            raise TfidfVocabularyError(
                f"index {entry.index}: document_frequency {entry.document_frequency} "
                f"is below min_df {min_df}"
            )


def composition_by_order(vocabulary: Vocabulary) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in vocabulary.entries:
        key = str(entry.order)
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


__all__ = [
    "FEATURE_NAME_PREFIX",
    "VOCABULARY_SCHEMA_VERSION",
    "Vocabulary",
    "VocabularyEntry",
    "composition_by_order",
    "encode_feature_tokens",
    "parse_vocabulary",
    "select_vocabulary",
    "validate_vocabulary",
    "vocabulary_bytes",
    "vocabulary_payload",
]
