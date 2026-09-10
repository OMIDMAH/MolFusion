"""Token n-gram generation: what a "feature" is, before any weighting.

Promoted here from the Phase 5F-C study package. It is contract-bearing
code -- it defines what MolFusion means by an n-gram -- so production owns
it and the study now imports it, rather than production reaching into a
package named `study` for its own definitions.

N-grams are tuples of Phase 5F-A tokens, never concatenated strings.
Concatenation is genuinely lossy for this tokenizer because a
multi-character token can be re-split at a different boundary:

    ("Cl", "C")  and  ("C", "lC")   both concatenate to "ClC"

so a string-keyed feature map would merge two distinct features and
corrupt every count and weight derived from them. Tuple keys make the
collision impossible, and the artifact serializes each feature as a JSON
array of tokens to preserve the property on disk.
"""

from collections import Counter
from collections.abc import Iterator, Sequence

Ngram = tuple[str, ...]


def iter_ngrams(tokens: Sequence[str], order: int) -> Iterator[Ngram]:
    """Yield every contiguous token n-gram of `order`, left to right.

    A molecule shorter than `order` tokens yields nothing at all -- it has
    no n-gram of that size, and padding it to produce one would invent a
    feature the molecule does not contain.
    """
    if order < 1:
        raise ValueError(f"n-gram order must be >= 1, got {order}")
    for start in range(len(tokens) - order + 1):
        yield tuple(tokens[start : start + order])


def document_ngram_counts(tokens: Sequence[str], order: int) -> dict[Ngram, int]:
    """Within-document occurrence counts for one n-gram order.

    The number of keys is the document's distinct-n-gram count (its
    document-frequency contribution) and the values sum to its occurrence
    count (its term-frequency contribution), so one pass yields both.
    """
    if order < 1:
        raise ValueError(f"n-gram order must be >= 1, got {order}")
    if order == 1:
        return dict(Counter((token,) for token in tokens))
    return dict(Counter(zip(*(tokens[offset:] for offset in range(order)))))


def document_ngram_counts_over_orders(
    tokens: Sequence[str], orders: Sequence[int]
) -> dict[Ngram, int]:
    """Counts across several orders, merged into one mapping.

    Safe to merge because n-grams of different orders are tuples of
    different lengths and can never collide as keys.
    """
    merged: dict[Ngram, int] = {}
    for order in orders:
        merged.update(document_ngram_counts(tokens, order))
    return merged


__all__ = [
    "Ngram",
    "document_ngram_counts",
    "document_ngram_counts_over_orders",
    "iter_ngrams",
]
