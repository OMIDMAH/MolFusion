"""The TF-IDF numerical weighting primitives, written out in full.

Phase 5F-C froze *which* token n-grams become features. This module is
about the separate question of *how* a retained feature becomes a number,
and it exists so that question has one written answer rather than a
library flag.

Recording `smooth_idf=True` somewhere would not be a contract: it names a
setting in a particular version of a particular library, and says nothing
about what the arithmetic is. Every formula below is therefore spelled out
in the docstrings, implemented directly, and pinned by tests against
values derived by hand. A future reader can reproduce any artifact number
from the recorded document frequencies with a calculator.

Two term-frequency rules, two inverse-document-frequency rules, two
normalizations. All four dimensions are kept selectable here because this
is a study module; Phase 5F-D freezes one combination and records it.
"""

import math
from typing import Any

import numpy as np

# -- term frequency ---------------------------------------------------------

TF_RAW = "raw"
TF_SUBLINEAR = "sublinear"
TF_MODES = (TF_RAW, TF_SUBLINEAR)

# -- inverse document frequency ---------------------------------------------

IDF_SMOOTHED = "smoothed"
IDF_UNSMOOTHED = "unsmoothed"
IDF_MODES = (IDF_SMOOTHED, IDF_UNSMOOTHED)

# -- normalization ----------------------------------------------------------

NORM_NONE = "none"
NORM_L2 = "l2"
NORMS = (NORM_NONE, NORM_L2)

# Natural logarithm everywhere. Stated because "log" is ambiguous across
# implementations and a base change rescales every IDF by a constant --
# harmless under L2 normalization, not harmless in a stored artifact whose
# numbers someone may later compare against a hand calculation.
LOG_BASE = "e"


def term_frequency(counts: np.ndarray, mode: str = TF_SUBLINEAR) -> np.ndarray:
    """Weight raw within-document counts.

        raw:        tf = count
        sublinear:  tf = 1 + ln(count)   for count > 0
                    tf = 0               for count = 0

    The sublinear branch is guarded rather than clipped: ln(0) is -inf, and
    a count of zero must produce exactly 0.0, not a tiny negative number or
    a warning. `np.log` is only evaluated where counts are positive.
    """
    values = np.asarray(counts, dtype=np.float64)
    if mode == TF_RAW:
        return values
    if mode == TF_SUBLINEAR:
        weighted = np.zeros_like(values)
        positive = values > 0
        weighted[positive] = 1.0 + np.log(values[positive])
        return weighted
    raise ValueError(f"unknown term-frequency mode: {mode!r}")


def inverse_document_frequency(
    document_frequencies: np.ndarray, n_documents: int, mode: str = IDF_SMOOTHED
) -> np.ndarray:
    """Corpus-level term weights, in natural log.

        unsmoothed:  idf(t) = ln( N / df(t) ) + 1
        smoothed:    idf(t) = ln( (1 + N) / (1 + df(t)) ) + 1

    The trailing `+ 1` is the standard floor that keeps a term appearing in
    every document at weight 1 rather than 0, so a universally present
    feature is damped rather than deleted. Both forms are strictly
    decreasing in df, so the ordering of terms by weight is the same under
    either; only the spacing differs.

    Smoothing is equivalent to imagining one extra document containing
    every term. With the frozen `min_df = 5` there is no zero-df term in
    the vocabulary, so smoothing is not needed to avoid division by zero --
    which is exactly why the choice between them has to be justified
    numerically rather than by that reflex.
    """
    frequencies = np.asarray(document_frequencies, dtype=np.float64)
    if n_documents <= 0:
        raise ValueError(f"n_documents must be positive, got {n_documents}")
    if np.any(frequencies < 0):
        raise ValueError("document frequencies cannot be negative")

    if mode == IDF_SMOOTHED:
        return np.log((1.0 + n_documents) / (1.0 + frequencies)) + 1.0
    if mode == IDF_UNSMOOTHED:
        if np.any(frequencies == 0):
            raise ValueError(
                "unsmoothed IDF is undefined for a term with df = 0; the frozen "
                "vocabulary has min_df = 5, so this indicates a df vector that "
                "does not belong to the vocabulary it is being applied to"
            )
        return np.log(n_documents / frequencies) + 1.0
    raise ValueError(f"unknown inverse-document-frequency mode: {mode!r}")


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 length, leaving all-zero rows at zero.

    The zero row is the case that matters. A molecule that tokenizes
    cleanly but contains no vocabulary n-gram is a legitimate outcome, not
    an error, and dividing it by its own norm would turn a well-defined
    zero vector into NaN -- a value that then propagates silently through
    every downstream similarity, mean, and model input. Zero norms are
    therefore replaced by 1 before the division, which leaves those rows
    exactly zero.
    """
    values = np.asarray(vectors, dtype=np.float64)
    single = values.ndim == 1
    matrix = values[np.newaxis, :] if single else values

    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
    safe = np.where(norms > 0.0, norms, 1.0)
    normalized = matrix / safe[:, np.newaxis]
    return normalized[0] if single else normalized


def normalize(vectors: np.ndarray, norm: str = NORM_L2) -> np.ndarray:
    if norm == NORM_NONE:
        return np.asarray(vectors, dtype=np.float64)
    if norm == NORM_L2:
        return l2_normalize(vectors)
    raise ValueError(f"unknown normalization: {norm!r}")


def tfidf(
    counts: np.ndarray,
    idf: np.ndarray,
    *,
    tf_mode: str = TF_SUBLINEAR,
    norm: str = NORM_L2,
    dtype: Any = np.float64,
) -> np.ndarray:
    """The full transformation: counts -> TF -> multiply by IDF -> normalize.

    Order matters and is fixed here: normalization is the *last* step, so
    it acts on the TF-IDF values and not on raw counts. Normalizing first
    would make the IDF weighting partly cosmetic.

    The output dtype is applied only at the end. All arithmetic happens in
    float64 regardless, so a float32 output is a float64 computation that
    was rounded once, rather than a computation carried out in float32.
    """
    weighted = term_frequency(counts, tf_mode) * np.asarray(idf, dtype=np.float64)
    return normalize(weighted, norm).astype(dtype, copy=False)


def idf_formula(mode: str) -> str:
    """The formula as text, for the report and the future artifact metadata."""
    if mode == IDF_SMOOTHED:
        return "idf(t) = ln((1 + N) / (1 + df(t))) + 1"
    if mode == IDF_UNSMOOTHED:
        return "idf(t) = ln(N / df(t)) + 1"
    raise ValueError(f"unknown inverse-document-frequency mode: {mode!r}")


def tf_formula(mode: str) -> str:
    if mode == TF_RAW:
        return "tf(t,d) = count(t in d)"
    if mode == TF_SUBLINEAR:
        return "tf(t,d) = 1 + ln(count(t in d)) if count > 0 else 0"
    raise ValueError(f"unknown term-frequency mode: {mode!r}")


def weighting_definition(
    tf_mode: str, idf_mode: str, norm: str, idf_dtype: str, output_dtype: str
) -> dict[str, Any]:
    """The complete numerical contract as report-ready data."""
    return {
        "tf_mode": tf_mode,
        "tf_formula": tf_formula(tf_mode),
        "idf_mode": idf_mode,
        "idf_formula": idf_formula(idf_mode),
        "smooth_idf": idf_mode == IDF_SMOOTHED,
        "use_idf": True,
        "log_base": LOG_BASE,
        "norm": norm,
        "normalization_formula": (
            "x / ||x||_2, with all-zero vectors left unchanged at zero"
            if norm == NORM_L2
            else "none"
        ),
        "order_of_operations": "tf(counts) -> multiply by idf -> normalize",
        "internal_arithmetic_dtype": "float64",
        "idf_storage_dtype": idf_dtype,
        "runtime_output_dtype": output_dtype,
        "zero_vector": (
            "a molecule that tokenizes but retains no vocabulary term yields an "
            "all-zero vector of the full dimension; this is a valid result, not "
            "a failure, and normalization leaves it zero rather than producing NaN"
        ),
        "oov": (
            "an n-gram outside the frozen vocabulary contributes nothing: no UNK "
            "dimension, no vocabulary growth, no refit, no exception"
        ),
    }


def natural_log(value: float) -> float:
    """Exposed so a test can assert the base without importing numpy."""
    return math.log(value)


__all__ = [
    "IDF_MODES",
    "IDF_SMOOTHED",
    "IDF_UNSMOOTHED",
    "LOG_BASE",
    "NORMS",
    "NORM_L2",
    "NORM_NONE",
    "TF_MODES",
    "TF_RAW",
    "TF_SUBLINEAR",
    "idf_formula",
    "inverse_document_frequency",
    "l2_normalize",
    "natural_log",
    "normalize",
    "tf_formula",
    "term_frequency",
    "tfidf",
    "weighting_definition",
]
