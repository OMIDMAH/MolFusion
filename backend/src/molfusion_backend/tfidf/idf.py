"""The IDF payload: computation, storage, and validation against the formula.

Stored as a plain NumPy float64 array. Not a pickle, not joblib, not a
fitted estimator, not an object dtype -- an artifact meant to be
rebuildable years from now must be readable without executing anything and
without the class that wrote it still existing.

Alignment is the invariant that matters: `idf[i]` is the weight of the
vocabulary entry at index `i`. Everything here exists to make a
misalignment impossible to ship silently.
"""

from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np

from molfusion_backend.tfidf import weighting
from molfusion_backend.tfidf.errors import TfidfIdfError

# Tolerance for re-deriving a stored IDF from its recorded document
# frequency. The recomputation runs the identical expression in the
# identical dtype, so exact equality is the expectation and this bound
# exists only to absorb a legitimately different-but-equivalent evaluation
# order rather than to paper over a wrong formula.
IDF_RECOMPUTE_TOLERANCE = 1e-12


def compute_idf(document_frequencies: Sequence[int], n_documents: int) -> np.ndarray:
    """The IDF vector for a vocabulary, in index order.

    `idf[i] = ln((1 + N) / (1 + df[i])) + 1`, in float64.
    """
    if not document_frequencies:
        raise TfidfIdfError("cannot compute IDF for an empty vocabulary")
    values = weighting.inverse_document_frequency(
        np.asarray(document_frequencies, dtype=np.float64),
        n_documents,
        weighting.FROZEN_IDF_MODE,
    )
    return values.astype(weighting.IDF_DTYPE, copy=False)


def idf_bytes(idf: np.ndarray) -> bytes:
    """The `.npy` payload bytes.

    Written through an in-memory buffer so the caller controls the exact
    filename (`np.save` appends `.npy` to a path argument) and so the same
    bytes can be hashed and written without a round trip through disk.
    `allow_pickle=False` is the default for a numeric array and is stated
    anyway: this payload must never be able to carry executable content.
    """
    buffer = BytesIO()
    np.save(buffer, np.ascontiguousarray(idf, dtype=weighting.IDF_DTYPE), allow_pickle=False)
    return buffer.getvalue()


def load_idf(path: Path) -> np.ndarray:
    """Read an `idf.npy`, refusing anything that could execute on load."""
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TfidfIdfError(f"could not load IDF payload {path}: {exc}") from exc
    if not isinstance(values, np.ndarray):
        raise TfidfIdfError(f"IDF payload {path} did not contain an array")
    return values


def validate_idf(
    idf: np.ndarray,
    *,
    dimension: int,
    document_frequencies: Sequence[int] | None = None,
    n_documents: int | None = None,
) -> None:
    """Assert the IDF payload's shape, dtype, finiteness and formula."""
    if idf.ndim != 1 or idf.shape != (dimension,):
        raise TfidfIdfError(f"IDF shape {idf.shape} does not match dimension ({dimension},)")
    if idf.dtype != np.dtype(weighting.IDF_DTYPE):
        raise TfidfIdfError(
            f"IDF dtype is {idf.dtype}, expected {np.dtype(weighting.IDF_DTYPE)}"
        )
    if not np.all(np.isfinite(idf)):
        raise TfidfIdfError("IDF contains non-finite values")
    if not np.all(idf > 0.0):
        raise TfidfIdfError("IDF contains non-positive values")

    if document_frequencies is None or n_documents is None:
        return

    if len(document_frequencies) != dimension:
        raise TfidfIdfError(
            f"{len(document_frequencies)} document frequencies for {dimension} IDF values"
        )
    expected = compute_idf(document_frequencies, n_documents)
    deviation = np.abs(idf - expected)
    worst = int(np.argmax(deviation))
    if deviation[worst] > IDF_RECOMPUTE_TOLERANCE:
        raise TfidfIdfError(
            "stored IDF does not reproduce "
            f"{weighting.idf_formula(weighting.FROZEN_IDF_MODE)} at index {worst}: "
            f"stored {idf[worst]!r}, recomputed {expected[worst]!r} "
            f"from df={document_frequencies[worst]}, N={n_documents}"
        )


__all__ = [
    "IDF_RECOMPUTE_TOLERANCE",
    "compute_idf",
    "idf_bytes",
    "load_idf",
    "validate_idf",
]
