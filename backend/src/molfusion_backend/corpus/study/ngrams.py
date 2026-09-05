"""Token n-gram generation and exact DF/TF accumulation.

N-grams are tuples of Phase 5F-A tokens, never concatenated strings. That
is not a stylistic preference: concatenation is genuinely lossy for this
tokenizer, because a multi-character token can be re-split at a different
boundary. ("Cl", "C") and ("C", "lC") both concatenate to "ClC", so a
string-keyed counter would merge two distinct n-grams into one feature and
quietly corrupt every count derived from it. Tuple keys make that
impossible, and JSON arrays preserve the property on the way out.

Two frequencies are tracked per n-gram, and they are not interchangeable:

    DF  molecules containing the n-gram at least once
    TF  total occurrences across all molecules

A molecule whose SMILES contains "C" twenty times contributes DF += 1 and
TF += 20. Which of the two should drive feature selection is exactly the
question Phase 5F-C exists to answer, so both are carried everywhere.

Counts are additionally split by study subset and by molecule token-count
band. The bands cost almost nothing (the n-gram vocabulary is small) and
they are what makes the long-molecule sensitivity question answerable
after the fact: TF restricted to short molecules can be re-derived from
band sums without a second pass over the corpus.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from molfusion_backend.tfidf.ngrams import Ngram, document_ngram_counts, iter_ngrams

NGRAM_ORDERS = (1, 2, 3)

# Upper edges, inclusive, of the molecule token-count bands. Six bands:
# (0,32], (32,64], (64,128], (128,256], (256,512], (512,inf). Placed
# around the frozen corpus's median token count of 48 so the bands
# straddle the typical molecule rather than bunching below it, and left
# open at the top because the corpus reaches 1617 tokens.
TOKEN_COUNT_BAND_EDGES = (32, 64, 128, 256, 512)
BAND_COUNT = len(TOKEN_COUNT_BAND_EDGES) + 1

SUBSET_FIT = 0
SUBSET_HOLDOUT = 1
SUBSET_COUNT = 2

# One flat record per n-gram: DF slots first, then TF slots, each indexed
# by subset * BAND_COUNT + band. A single list beats parallel dicts
# because the hot loop then does one dictionary lookup per n-gram per
# document instead of one lookup per counter.
_DF_OFFSET = 0
_TF_OFFSET = SUBSET_COUNT * BAND_COUNT
_RECORD_SIZE = 2 * SUBSET_COUNT * BAND_COUNT


def token_count_band(token_count: int) -> int:
    """Index of the length band a molecule of `token_count` tokens falls in."""
    for index, edge in enumerate(TOKEN_COUNT_BAND_EDGES):
        if token_count <= edge:
            return index
    return BAND_COUNT - 1


@dataclass(frozen=True)
class NgramEntry:
    """One n-gram's accumulated frequencies, as a read-only view."""

    ngram: Ngram
    order: int
    document_frequency: int
    term_frequency: int
    document_frequency_fit: int
    term_frequency_fit: int
    document_frequency_holdout: int
    term_frequency_holdout: int
    document_frequency_fit_bands: tuple[int, ...]
    term_frequency_fit_bands: tuple[int, ...]

    def term_frequency_fit_up_to_band(self, band: int) -> int:
        """Fit-subset TF restricted to molecules in bands <= `band`.

        The long-molecule diagnostic: comparing this with the unrestricted
        fit TF shows how much of an n-gram's weight comes from unusually
        long records repeating one motif.
        """
        return sum(self.term_frequency_fit_bands[: band + 1])

    def document_frequency_fit_up_to_band(self, band: int) -> int:
        return sum(self.document_frequency_fit_bands[: band + 1])


class NgramFrequencyAccumulator:
    """Exact per-n-gram DF and TF, accumulated one molecule at a time.

    Exact, not sketched: the measured n-gram vocabulary of canonical SMILES
    is small (thousands, not millions -- a consequence of a token alphabet
    of a few hundred symbols), so approximate counting would trade away
    auditability for memory the study does not need. The study report's
    `resource_usage` section records the measurement backing that claim.
    """

    def __init__(self, orders: Sequence[int] = NGRAM_ORDERS) -> None:
        self.orders = tuple(orders)
        self._counts: dict[int, dict[Ngram, list[int]]] = {order: {} for order in self.orders}
        self._documents = [0] * SUBSET_COUNT
        self._documents_by_band = [[0] * BAND_COUNT for _ in range(SUBSET_COUNT)]
        self._tokens = [0] * SUBSET_COUNT

    def add_document(self, tokens: Sequence[str], *, holdout: bool) -> None:
        subset = SUBSET_HOLDOUT if holdout else SUBSET_FIT
        band = token_count_band(len(tokens))
        slot = subset * BAND_COUNT + band
        df_index = _DF_OFFSET + slot
        tf_index = _TF_OFFSET + slot

        self._documents[subset] += 1
        self._documents_by_band[subset][band] += 1
        self._tokens[subset] += len(tokens)

        for order in self.orders:
            table = self._counts[order]
            for ngram, occurrences in document_ngram_counts(tokens, order).items():
                record = table.get(ngram)
                if record is None:
                    record = table[ngram] = [0] * _RECORD_SIZE
                record[df_index] += 1
                record[tf_index] += occurrences

    # -- read-out ----------------------------------------------------------

    @property
    def document_count(self) -> int:
        return sum(self._documents)

    @property
    def fit_document_count(self) -> int:
        return self._documents[SUBSET_FIT]

    @property
    def holdout_document_count(self) -> int:
        return self._documents[SUBSET_HOLDOUT]

    def documents_by_band(self, *, holdout: bool) -> tuple[int, ...]:
        return tuple(self._documents_by_band[SUBSET_HOLDOUT if holdout else SUBSET_FIT])

    def token_count(self, *, holdout: bool) -> int:
        return self._tokens[SUBSET_HOLDOUT if holdout else SUBSET_FIT]

    def distinct_count(self, order: int) -> int:
        return len(self._counts[order])

    def entries(self, order: int) -> list[NgramEntry]:
        """Every n-gram of `order`, sorted by token tuple.

        Sorted on the way out so no consumer ever observes dictionary
        insertion order: two runs over the same corpus presented in
        different document orders must produce identical study output.
        """
        table = self._counts[order]
        entries = []
        for ngram in sorted(table):
            record = table[ngram]
            df_fit = record[_DF_OFFSET : _DF_OFFSET + BAND_COUNT]
            df_holdout = record[_DF_OFFSET + BAND_COUNT : _DF_OFFSET + 2 * BAND_COUNT]
            tf_fit = record[_TF_OFFSET : _TF_OFFSET + BAND_COUNT]
            tf_holdout = record[_TF_OFFSET + BAND_COUNT : _TF_OFFSET + 2 * BAND_COUNT]
            entries.append(
                NgramEntry(
                    ngram=ngram,
                    order=order,
                    document_frequency=sum(df_fit) + sum(df_holdout),
                    term_frequency=sum(tf_fit) + sum(tf_holdout),
                    document_frequency_fit=sum(df_fit),
                    term_frequency_fit=sum(tf_fit),
                    document_frequency_holdout=sum(df_holdout),
                    term_frequency_holdout=sum(tf_holdout),
                    document_frequency_fit_bands=tuple(df_fit),
                    term_frequency_fit_bands=tuple(tf_fit),
                )
            )
        return entries

    def definition(self) -> dict[str, Any]:
        return {
            "ngram_key": "tuple of Phase 5F-A tokens; never a concatenated string",
            "orders": list(self.orders),
            "document_frequency": "molecules containing the n-gram at least once",
            "term_frequency": "total occurrences across all molecules",
            "token_count_band_edges": list(TOKEN_COUNT_BAND_EDGES),
            "token_count_bands": band_labels(),
        }


def band_labels() -> list[str]:
    """Human-readable band intervals, in band-index order."""
    labels = []
    lower = 0
    for edge in TOKEN_COUNT_BAND_EDGES:
        labels.append(f"({lower},{edge}]")
        lower = edge
    labels.append(f"({lower},inf)")
    return labels


__all__ = [
    "BAND_COUNT",
    "NGRAM_ORDERS",
    "Ngram",
    "NgramEntry",
    "NgramFrequencyAccumulator",
    "TOKEN_COUNT_BAND_EDGES",
    "band_labels",
    "document_ngram_counts",
    "iter_ngrams",
    "token_count_band",
]
