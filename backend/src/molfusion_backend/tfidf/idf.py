"""The IDF payload: computation, storage, and validation against the formula.

Stored as a plain NumPy numeric array. Not a pickle, not joblib, not a
fitted estimator, not an object dtype -- an artifact meant to be
rebuildable years from now must be readable without executing anything and
without the class that wrote it still existing.

The serialization path is pinned rather than left to defaults, because the
default is only *usually* stable:

  * the `.npy` format version is chosen by NumPy from the array's
    properties; it is written explicitly here so a future NumPy cannot
    quietly promote the header;
  * the dtype is explicitly little-endian (`<f8`), not the platform-native
    `float64` alias, so the building machine's endianness is not stamped
    into the header;
  * the array is made C-contiguous, which fixes `fortran_order` to False.

With those three pinned, the file is exactly a fixed 10-byte prefix, a
fixed ASCII header naming only `descr`/`fortran_order`/`shape`, and the
raw doubles. It carries no timestamp, no path, no username, no library
version, and no machine identity -- nothing but the numbers.

Alignment is the other invariant that matters: `idf[i]` is the weight of
the vocabulary entry at index `i`. Everything here exists to make a
misalignment impossible to ship silently.
"""

import ast
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from molfusion_backend.tfidf import weighting
from molfusion_backend.tfidf.errors import TfidfIdfError

# Tolerance for re-deriving a stored IDF from its recorded document
# frequency. The recomputation runs the identical expression in the
# identical dtype, so exact equality is the expectation and this bound
# exists only to absorb a legitimately different-but-equivalent evaluation
# order rather than to paper over a wrong formula.
IDF_RECOMPUTE_TOLERANCE = 1e-12

# The pinned `.npy` serialization. Version 1.0 is the oldest and most
# widely readable form and is sufficient for a 1-D numeric array; naming it
# means a NumPy upgrade cannot change the payload bytes without this
# constant changing too.
IDF_NPY_VERSION = (1, 0)
IDF_NPY_MAGIC = b"\x93NUMPY"
IDF_NPY_DESCR = "<f8"
IDF_NPY_FORTRAN_ORDER = False


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
    """The `.npy` payload bytes, through a fully pinned path.

    `numpy.lib.format.write_array` is used directly rather than `np.save`
    so the format version is stated instead of inferred. The array is
    coerced to C-contiguous little-endian float64 first, which fixes every
    remaining header field. `allow_pickle=False` guarantees the payload can
    never carry executable content -- redundant for a numeric dtype, and
    stated because "redundant today" is not "redundant after the next
    edit".
    """
    # Checked before coercion, not after: casting an object array to
    # float64 either raises an unrelated TypeError or silently succeeds for
    # object arrays that happen to hold numbers. Either way the refusal
    # should name the actual problem.
    incoming = np.asarray(idf)
    if incoming.dtype.hasobject:
        raise TfidfIdfError(
            f"refusing to serialize an object-dtype array (dtype {incoming.dtype})"
        )
    if incoming.ndim != 1:
        raise TfidfIdfError(f"IDF payload must be 1-D, got shape {incoming.shape}")

    array = np.ascontiguousarray(incoming, dtype=weighting.IDF_DTYPE)

    buffer = BytesIO()
    np.lib.format.write_array(
        buffer, array, version=IDF_NPY_VERSION, allow_pickle=False
    )
    return buffer.getvalue()


def inspect_idf_payload(path: Path) -> dict[str, Any]:
    """The payload's structural facts, read from its bytes.

    Used both to validate an artifact and to assert in tests that the file
    contains nothing beyond a header and the raw array.
    """
    raw = Path(path).read_bytes()
    if raw[:6] != IDF_NPY_MAGIC:
        raise TfidfIdfError(f"{path} is not a .npy file")
    version = (raw[6], raw[7])
    if version != IDF_NPY_VERSION:
        raise TfidfIdfError(
            f"{path} uses .npy format version {version}, expected {IDF_NPY_VERSION}"
        )
    header_length = int.from_bytes(raw[8:10], "little")
    header = raw[10 : 10 + header_length]
    if not all(byte < 128 for byte in header):
        raise TfidfIdfError(f"{path} has a non-ASCII header")
    try:
        fields = dict(ast.literal_eval(header.decode("ascii")))
    except Exception as exc:  # noqa: BLE001 - any parse failure is a bad payload
        raise TfidfIdfError(f"{path} has an unparseable header: {exc}") from exc

    return {
        "version": version,
        "descr": fields.get("descr"),
        "fortran_order": fields.get("fortran_order"),
        "shape": fields.get("shape"),
        "header_fields": sorted(fields),
        "header_bytes": 10 + header_length,
        "data_bytes": len(raw) - 10 - header_length,
        "total_bytes": len(raw),
    }


def load_idf(path: Path) -> np.ndarray:
    """Read an `idf.npy`, refusing anything that could execute on load."""
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TfidfIdfError(f"could not load IDF payload {path}: {exc}") from exc
    if not isinstance(values, np.ndarray):
        raise TfidfIdfError(f"IDF payload {path} did not contain an array")
    if values.dtype.hasobject:
        raise TfidfIdfError(f"IDF payload {path} contains an object-dtype array")
    return values


def validate_idf_payload(path: Path, *, dimension: int) -> None:
    """Assert the serialized form itself, not just the array it decodes to.

    Two payloads can decode to the same numbers and still differ in bytes,
    which would break the artifact's reproducibility claim without
    affecting any vector. Checking the header catches that.
    """
    facts = inspect_idf_payload(path)
    if facts["descr"] != IDF_NPY_DESCR:
        raise TfidfIdfError(
            f"{path} declares dtype {facts['descr']!r}, expected {IDF_NPY_DESCR!r}; "
            "the payload must be little-endian float64 regardless of build host"
        )
    if facts["fortran_order"] is not IDF_NPY_FORTRAN_ORDER:
        raise TfidfIdfError(f"{path} declares fortran_order {facts['fortran_order']!r}")
    if facts["shape"] != (dimension,):
        raise TfidfIdfError(f"{path} declares shape {facts['shape']}, expected ({dimension},)")
    if facts["header_fields"] != ["descr", "fortran_order", "shape"]:
        raise TfidfIdfError(
            f"{path} header carries unexpected fields: {facts['header_fields']}"
        )
    if facts["data_bytes"] != dimension * 8:
        raise TfidfIdfError(
            f"{path} carries {facts['data_bytes']} data bytes, expected {dimension * 8}"
        )


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
    "IDF_NPY_DESCR",
    "IDF_NPY_FORTRAN_ORDER",
    "IDF_NPY_MAGIC",
    "IDF_NPY_VERSION",
    "IDF_RECOMPUTE_TOLERANCE",
    "compute_idf",
    "idf_bytes",
    "inspect_idf_payload",
    "load_idf",
    "validate_idf",
    "validate_idf_payload",
]
