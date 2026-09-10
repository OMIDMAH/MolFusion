"""The frozen benchmark-data serialization contract and release identity.

A checksum is only meaningful if the bytes it covers are defined. Phase 6A.1
therefore does not hash whatever a dataset library happened to write: it
re-serializes every endpoint through one explicit contract and hashes that.
The contract is deliberately boring and fully specified below, so a future
reader can regenerate the same bytes without owning the library that
produced the data.

Nothing here imports PyTDC. Acquisition needs that package; verification
must not, or the frozen data could only ever be checked by the tool that
produced it.
"""

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# --- the serialization contract ---------------------------------------
#
# Every clause exists to remove one source of byte-level drift:
#
#   UTF-8, no BOM      a BOM would change the checksum on some editors
#   LF only            the repository is developed on Windows, where the
#                      csv module would otherwise emit CRLF
#   fixed column order columns are named here, never taken from whatever
#                      order a frame happens to carry
#   source row order   preserved exactly, because row order is part of the
#                      upstream dataset's identity and reordering it would
#                      quietly define a different dataset
#   QUOTE_MINIMAL      one quoting rule, applied by the stdlib
#   final newline      present, so the file is a well-formed text file
#
SERIALIZATION_ID = "molfusion_frozen_csv_v1"
ENCODING = "utf-8"
LINE_TERMINATOR = "\n"
QUOTING = csv.QUOTE_MINIMAL
FIELDS: tuple[str, ...] = ("Drug_ID", "Drug", "Y")

SERIALIZATION_CONTRACT = {
    "serialization_id": SERIALIZATION_ID,
    "encoding": ENCODING,
    "byte_order_mark": False,
    "line_terminator": "LF",
    "quoting": "QUOTE_MINIMAL",
    "fields": list(FIELDS),
    "row_order": "source order preserved",
    "final_newline": True,
    "float_format": "repr() shortest round-trip form",
}


def format_cell(value: Any) -> str:
    """One stable text form per value.

    Floats use ``repr``: in Python 3 that is the shortest string that reads
    back as the identical float, so the serialization is both round-trip
    exact and independent of any formatting choice a dataframe library
    might otherwise impose.
    """
    if isinstance(value, float):
        return repr(value)
    return str(value)


def frozen_csv_bytes(rows: Iterable[Sequence[Any]], fields: Sequence[str] = FIELDS) -> bytes:
    """Serialize a header plus rows under the contract, returning raw bytes."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator=LINE_TERMINATOR, quoting=QUOTING)
    writer.writerow(list(fields))
    for row in rows:
        writer.writerow([format_cell(cell) for cell in row])
    return buffer.getvalue().encode(ENCODING)


def read_frozen_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a frozen file back as text, without interpreting the values.

    Values stay strings here on purpose. Parsing a label into a float is a
    decision belonging to the audit, not to the storage layer, and doing it
    at read time would make the round-trip test unable to detect a
    formatting change.
    """
    with open(path, encoding=ENCODING, newline="") as handle:
        reader = csv.reader(handle)
        rows = [list(row) for row in reader]
    if not rows:
        raise ValueError(f"{path} is empty; a frozen file always has a header")
    return rows[0], rows[1:]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def molecule_set_identity(canonical_smiles: Iterable[str]) -> str:
    """A hash of a *set* of molecules, independent of order or repetition.

    Used to prove statements about split membership -- that the official
    test set is the same set of molecules at every seed, for instance.
    Sorting and de-duplicating first is what makes the claim about the set
    rather than about one listing of it.
    """
    unique = sorted(set(canonical_smiles))
    payload = "\n".join(unique).encode(ENCODING)
    return hashlib.sha256(payload).hexdigest()


def release_identity(
    *,
    release_name: str,
    protocol_version: str,
    endpoints: dict[str, dict[str, Any]],
) -> str:
    """A content-derived identity for one frozen benchmark-data release.

    Derived only from things that change the science: the release name, the
    protocol version, and each endpoint's frozen checksums and split
    identities. Deliberately *not* derived from timestamps, absolute paths,
    directory metadata, or acquisition order -- re-freezing the same data
    tomorrow must produce the same identity, or the identity is recording
    when the work happened rather than what the data is.
    """
    summary = {
        "release_name": release_name,
        "protocol_version": protocol_version,
        "serialization_id": SERIALIZATION_ID,
        "endpoints": {
            name: {
                "train_val_sha256": entry["train_val"]["sha256"],
                "test_sha256": entry["test"]["sha256"],
                "test_set_identity": entry["split_identity"]["test_set_sha256"],
                "seed_identities": entry["split_identity"]["seeds"],
            }
            for name, entry in sorted(endpoints.items())
        },
    }
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode(ENCODING)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "FIELDS",
    "SERIALIZATION_CONTRACT",
    "SERIALIZATION_ID",
    "format_cell",
    "frozen_csv_bytes",
    "molecule_set_identity",
    "read_frozen_csv",
    "release_identity",
    "sha256_bytes",
    "sha256_file",
]
