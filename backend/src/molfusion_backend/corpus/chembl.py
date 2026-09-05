"""Streaming structure extraction from an official ChEMBL SQLite release.

Reads exactly one thing: the structural table. No assay, activity, target,
publication, or development-phase table is opened, so the resulting corpus
cannot be conditioned on any downstream supervised signal (see the leakage
policy in docs/reproducibility.md).
"""

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from molfusion_backend.corpus.errors import CorpusSourceError

# The authoritative structure table in every modern ChEMBL release, and the
# only columns this builder reads from it.
STRUCTURE_TABLE = "compound_structures"
STRUCTURE_KEY_COLUMN = "molregno"
STRUCTURE_SMILES_COLUMN = "canonical_smiles"

# Optional; joined only to make failure reports name a human-usable ChEMBL
# accession instead of a bare internal molregno. The corpus contents do not
# depend on it, so a release (or a test fixture) without this table still
# builds correctly.
COMPOUND_TABLE = "molecule_dictionary"
COMPOUND_ID_COLUMN = "chembl_id"


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One structure-bearing row, before any MolFusion processing."""

    molregno: int
    chembl_id: str | None
    smiles: str | None


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """Column names of `table`, or an empty set if it does not exist.

    Uses PRAGMA rather than assuming the documented ChEMBL schema, so an
    unexpected release layout produces a clear error naming what was
    missing instead of an opaque "no such column" deep in the scan.
    """
    # PRAGMA does not accept bound parameters for the table name; the value
    # is a module-level constant, never caller-supplied.
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _require_structure_schema(connection: sqlite3.Connection) -> None:
    columns = _table_columns(connection, STRUCTURE_TABLE)
    if not columns:
        raise CorpusSourceError(
            f"Source database has no {STRUCTURE_TABLE!r} table; this does not look "
            "like an official ChEMBL SQLite release."
        )

    missing = {STRUCTURE_KEY_COLUMN, STRUCTURE_SMILES_COLUMN} - columns
    if missing:
        raise CorpusSourceError(
            f"{STRUCTURE_TABLE!r} is missing required column(s) {sorted(missing)}; "
            f"found {sorted(columns)}."
        )


def open_source_database(path: Path) -> sqlite3.Connection:
    """Open an official ChEMBL SQLite release read-only.

    Read-only both to make the build incapable of mutating a multi-GB
    shared source asset and so that its checksum stays meaningful as
    provenance across rebuilds.
    """
    if not path.is_file():
        raise CorpusSourceError(f"Source database not found: {path}")

    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise CorpusSourceError(f"Could not open source database {path}: {exc}") from exc

    try:
        _require_structure_schema(connection)
    except CorpusSourceError:
        connection.close()
        raise

    return connection


def has_compound_dictionary(connection: sqlite3.Connection) -> bool:
    columns = _table_columns(connection, COMPOUND_TABLE)
    return {STRUCTURE_KEY_COLUMN, COMPOUND_ID_COLUMN} <= columns


def structure_query(with_compound_ids: bool) -> str:
    """The extraction SQL, exposed so the build report can record verbatim
    which query produced the corpus."""
    if with_compound_ids:
        return (
            f"SELECT s.{STRUCTURE_KEY_COLUMN}, d.{COMPOUND_ID_COLUMN}, "
            f"s.{STRUCTURE_SMILES_COLUMN} "
            f"FROM {STRUCTURE_TABLE} AS s "
            f"LEFT JOIN {COMPOUND_TABLE} AS d "
            f"ON d.{STRUCTURE_KEY_COLUMN} = s.{STRUCTURE_KEY_COLUMN} "
            f"ORDER BY s.{STRUCTURE_KEY_COLUMN}"
        )
    return (
        f"SELECT s.{STRUCTURE_KEY_COLUMN}, NULL, s.{STRUCTURE_SMILES_COLUMN} "
        f"FROM {STRUCTURE_TABLE} AS s "
        f"ORDER BY s.{STRUCTURE_KEY_COLUMN}"
    )


def iter_source_records(connection: sqlite3.Connection) -> Iterator[SourceRecord]:
    """Stream every structure-bearing row, ordered by molregno.

    Streaming (a plain cursor iterated row by row) rather than fetchall()
    or a DataFrame: a full ChEMBL release is millions of rows and the
    builder must never hold the source table in memory.

    The ORDER BY is not what makes the corpus deterministic -- the final
    lexicographic sort does that, and the corpus bytes are identical for
    any source row order. It exists so that a diagnostic tied to *scan*
    order, such as which record trips an abort first, is reproducible.
    """
    query = structure_query(has_compound_dictionary(connection))
    cursor = connection.execute(query)
    try:
        for molregno, chembl_id, smiles in cursor:
            yield SourceRecord(molregno=molregno, chembl_id=chembl_id, smiles=smiles)
    finally:
        cursor.close()
