"""The frozen TF-IDF arithmetic, written out in full.

Promoted here from the Phase 5F-C.1 study package, which measured the
alternatives; this is now the production home of the contract and the
study imports it. The mode parameters are kept because a formula that can
only produce one answer cannot be tested against the other one, and the
frozen choices are named as constants rather than hard-coded.

Recording `smooth_idf=True` somewhere would not be a contract: it names a
setting in one version of one library and says nothing about the
arithmetic. Every formula below is spelled out, implemented directly, and
pinned by tests against values derived by hand, so any artifact number can
be reproduced from the recorded document frequencies with a calculator.
"""

import math
from typing import Any

import numpy as np

TF_RAW = "raw"
TF_SUBLINEAR = "sublinear"
TF_MODES = (TF_RAW, TF_SUBLINEAR)

IDF_SMOOTHED = "smoothed"
IDF_UNSMOOTHED = "unsmoothed"
IDF_MODES = (IDF_SMOOTHED, IDF_UNSMOOTHED)

NORM_NONE = "none"
NORM_L2 = "l2"
NORMS = (NORM_NONE, NORM_L2)

# Natural logarithm everywhere. Recorded because "log" is ambiguous across
# implementations, and a base change rescales every IDF by a constant --
# invisible after L2 normalization, very visible to anyone checking a
# stored IDF by hand.
LOG_BASE = "e"

# The Phase 5F-C.1 decisions, as the values production actually uses.
FROZEN_TF_MODE = TF_SUBLINEAR
FROZEN_IDF_MODE = IDF_SMOOTHED
FROZEN_NORM = NORM_L2
FROZEN_IDF_DTYPE = "float64"
FROZEN_RUNTIME_DTYPE = "float32"

# Explicitly little-endian, not the platform-native alias. `np.float64`
# resolves to native byte order, so serializing it would stamp the
# building machine's endianness into the payload header ("<f8" here,
# ">f8" on a big-endian host) and two correct builds on different
# architectures would disagree byte for byte. Pinning the order makes
# the payload a property of the data alone. On a little-endian host
# this is the same dtype and the same bytes.
IDF_DTYPE = np.dtype("<f8")
# Runtime output is returned in memory and never serialized, so it uses
# the native alias.
RUNTIME_DTYPE = np.float32


def term_frequency(counts: np.ndarray, mode: str = FROZEN_TF_MODE) -> np.ndarray:
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
    document_frequencies: np.ndarray, n_documents: int, mode: str = FROZEN_IDF_MODE
) -> np.ndarray:
    """Corpus-level term weights, in natural log.

        unsmoothed:  idf(t) = ln( N / df(t) ) + 1
        smoothed:    idf(t) = ln( (1 + N) / (1 + df(t)) ) + 1

    The trailing `+ 1` is the standard floor that keeps a term appearing in
    every document at weight 1 rather than 0, so a universally present
    feature is damped rather than deleted. Both forms are strictly
    decreasing in df, so they induce the same term ordering; only the
    spacing differs.
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
                "vocabulary has min_df >= 1, so this indicates a df vector that "
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
    replaced by 1 before the division, which leaves those rows exactly zero.
    """
    values = np.asarray(vectors, dtype=np.float64)
    single = values.ndim == 1
    matrix = values[np.newaxis, :] if single else values

    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
    safe = np.where(norms > 0.0, norms, 1.0)
    normalized = matrix / safe[:, np.newaxis]
    return normalized[0] if single else normalized


def normalize(vectors: np.ndarray, norm: str = FROZEN_NORM) -> np.ndarray:
    if norm == NORM_NONE:
        return np.asarray(vectors, dtype=np.float64)
    if norm == NORM_L2:
        return l2_normalize(vectors)
    raise ValueError(f"unknown normalization: {norm!r}")


def tfidf(
    counts: np.ndarray,
    idf: np.ndarray,
    *,
    tf_mode: str = FROZEN_TF_MODE,
    norm: str = FROZEN_NORM,
    dtype: Any = RUNTIME_DTYPE,
) -> np.ndarray:
    """counts -> TF -> multiply by IDF -> normalize -> cast.

    Order is fixed: normalization is the *last* step, so it acts on the
    TF-IDF values rather than on raw counts -- normalizing first would make
    the IDF weighting partly cosmetic.

    The output dtype is applied only at the end. All arithmetic happens in
    float64 regardless, so a float32 result is a float64 computation
    rounded once, not a computation carried out in float32.
    """
    weighted = term_frequency(counts, tf_mode) * np.asarray(idf, dtype=np.float64)
    return normalize(weighted, norm).astype(dtype, copy=False)


def idf_formula(mode: str = FROZEN_IDF_MODE) -> str:
    """The formula as text, for the config payload and the build report."""
    if mode == IDF_SMOOTHED:
        return "idf(t) = ln((1 + N) / (1 + df(t))) + 1"
    if mode == IDF_UNSMOOTHED:
        return "idf(t) = ln(N / df(t)) + 1"
    raise ValueError(f"unknown inverse-document-frequency mode: {mode!r}")


def tf_formula(mode: str = FROZEN_TF_MODE) -> str:
    if mode == TF_RAW:
        return "tf(t,d) = count(t in d)"
    if mode == TF_SUBLINEAR:
        return "tf(t,d) = 1 + ln(count(t in d)) if count > 0 else 0"
    raise ValueError(f"unknown term-frequency mode: {mode!r}")


def natural_log(value: float) -> float:
    """Exposed so a test can assert the base without importing numpy."""
    return math.log(value)


__all__ = [
    "FROZEN_IDF_DTYPE",
    "FROZEN_IDF_MODE",
    "FROZEN_NORM",
    "FROZEN_RUNTIME_DTYPE",
    "FROZEN_TF_MODE",
    "IDF_DTYPE",
    "IDF_MODES",
    "IDF_SMOOTHED",
    "IDF_UNSMOOTHED",
    "LOG_BASE",
    "NORMS",
    "NORM_L2",
    "NORM_NONE",
    "RUNTIME_DTYPE",
    "TF_MODES",
    "TF_RAW",
    "TF_SUBLINEAR",
    "idf_formula",
    "inverse_document_frequency",
    "l2_normalize",
    "natural_log",
    "normalize",
    "term_frequency",
    "tf_formula",
    "tfidf",
]
