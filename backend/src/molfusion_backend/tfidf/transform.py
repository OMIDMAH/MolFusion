"""Reference transformation: tokens in, a frozen TF-IDF vector out.

This is the mathematics a future FeatureAgent will call. It is *not* the
agent: no registry entry, no agent class, no API surface, no feature
metadata. Phase 5F-D stops at the artifact and the arithmetic that reads
it.

MolFusion-owned rather than sklearn-backed, decided in Phase 5F-C.1: by
this point MolFusion already owns the tokenizer, the vocabulary, the index
order, the IDF vector, the TF rule and the normalization, so sklearn would
contribute only a multiply and a divide -- while adding a dependency whose
private state a stored vectorizer would pin. The parity tests check this
implementation against sklearn rather than delegating to it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from molfusion_backend.tfidf import weighting
from molfusion_backend.tfidf.errors import TfidfIdfError
from molfusion_backend.tfidf.ngrams import Ngram, document_ngram_counts_over_orders
from molfusion_backend.tfidf.contract import NGRAM_ORDERS


@dataclass(frozen=True)
class TfidfTransformer:
    """A frozen vocabulary and IDF vector, ready to transform token sequences."""

    index_map: dict[Ngram, int]
    idf: np.ndarray
    dimension: int
    orders: tuple[int, ...] = NGRAM_ORDERS

    def __post_init__(self) -> None:
        if self.idf.shape != (self.dimension,):
            raise TfidfIdfError(
                f"IDF shape {self.idf.shape} does not match dimension ({self.dimension},)"
            )
        if len(self.index_map) != self.dimension:
            raise TfidfIdfError(
                f"index map has {len(self.index_map)} terms for dimension {self.dimension}"
            )

    def counts(self, tokens: Sequence[str]) -> np.ndarray:
        """Dense within-molecule counts over the vocabulary.

        Out-of-vocabulary n-grams are simply not looked up again: they add
        no column, raise nothing, and leave no trace. That is the whole of
        the OOV contract.
        """
        dense = np.zeros(self.dimension, dtype=np.float64)
        for ngram, count in document_ngram_counts_over_orders(tokens, self.orders).items():
            column = self.index_map.get(ngram)
            if column is not None:
                dense[column] += count
        return dense

    def transform(self, tokens: Sequence[str]) -> np.ndarray:
        """The frozen vector: sublinear TF, times IDF, L2-normalized, float32.

        A molecule retaining no vocabulary term yields exactly
        `np.zeros(dimension, dtype=float32)` -- a valid result, never NaN,
        never an error.
        """
        return weighting.tfidf(
            self.counts(tokens),
            self.idf,
            tf_mode=weighting.FROZEN_TF_MODE,
            norm=weighting.FROZEN_NORM,
            dtype=weighting.RUNTIME_DTYPE,
        )

    def transform_many(self, documents: Sequence[Sequence[str]]) -> np.ndarray:
        """One row per document, same contract per row."""
        matrix = np.zeros((len(documents), self.dimension), dtype=np.float64)
        for row, tokens in enumerate(documents):
            matrix[row] = self.counts(tokens)
        return weighting.tfidf(
            matrix,
            self.idf,
            tf_mode=weighting.FROZEN_TF_MODE,
            norm=weighting.FROZEN_NORM,
            dtype=weighting.RUNTIME_DTYPE,
        )


def zero_vector(dimension: int) -> np.ndarray:
    """The vector a molecule with no retained term produces."""
    return np.zeros(dimension, dtype=weighting.RUNTIME_DTYPE)


__all__ = ["TfidfTransformer", "zero_vector"]
