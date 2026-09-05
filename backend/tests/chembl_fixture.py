"""A tiny stand-in for an official ChEMBL SQLite release.

Deliberately not a downloaded slice of the real database: the test suite
must never depend on network access or a multi-GB asset. This builds only
the structural tables the corpus builder reads, with rows chosen to cover
every branch of the per-record processing contract.
"""

import sqlite3
from pathlib import Path

# (molregno, chembl_id, canonical_smiles) with the reason each row exists.
# Kept as one table so a reader can see the whole fixture -- and the
# expectations derived from it -- in a single place.
FIXTURE_ROWS: list[tuple[int, str, str | None]] = [
    # Ordinary valid molecules.
    (1, "CHEMBL1", "CCO"),
    (2, "CHEMBL2", "c1ccccc1"),
    (3, "CHEMBL3", "CC(=O)Oc1ccccc1C(=O)O"),
    # Written differently, same molecule as CHEMBL1: must collapse to one
    # document *after* canonicalization, not before.
    (4, "CHEMBL4", "OCC"),
    # Kekule benzene: same molecule as CHEMBL2 under the aromatic contract.
    (5, "CHEMBL5", "C1=CC=CC=C1"),
    # Opposite stereoisomers: must stay two distinct documents.
    (6, "CHEMBL6", "C[C@H](O)c1ccccc1"),
    (7, "CHEMBL7", "C[C@@H](O)c1ccccc1"),
    # Charged, disconnected (a salt -- kept whole, never stripped).
    (8, "CHEMBL8", "CC(=O)[O-].[Na+]"),
    (9, "CHEMBL9", "C[N+](C)(C)C"),
    # Isotopic labelling must survive.
    (10, "CHEMBL10", "[13CH3]CO"),
    # Exact duplicate string of CHEMBL3.
    (11, "CHEMBL11", "CC(=O)Oc1ccccc1C(=O)O"),
    # --- rows that must be excluded, each in its own category ---
    (12, "CHEMBL12", None),          # SQL NULL
    (13, "CHEMBL13", ""),            # empty string
    (14, "CHEMBL14", "   "),         # whitespace-only
    (15, "CHEMBL15", "not_a_molecule"),  # RDKit parse failure
    (16, "CHEMBL16", "C(C"),             # unbalanced branch
]

# The expected corpus, written out by hand in the exact order the contract
# requires, never computed with sorted() or with the code under test --
# otherwise the sorting and deduplication tests would assert only that the
# builder agrees with itself.
#
# Every fixture SMILES above happens to already be in RDKit canonical form
# (verified directly against Chem.MolToSmiles), so these strings are both
# the inputs and the expected canonical outputs. The ordering below is
# Python's Unicode code-point ordering, where 'C' (67) < '[' (91) < 'c'
# (99) -- which is why the aromatic entry sorts last and the bracket-atom
# entry sorts after every 'C' entry.
EXPECTED_DOCUMENTS = [
    "CC(=O)Oc1ccccc1C(=O)O",  # CHEMBL3 + CHEMBL11 (exact duplicate)
    "CC(=O)[O-].[Na+]",       # salt: both components retained
    "CCO",                    # CHEMBL1 + CHEMBL4 ("OCC")
    "C[C@@H](O)c1ccccc1",     # stereoisomer, distinct from the next
    "C[C@H](O)c1ccccc1",
    "C[N+](C)(C)C",
    "[13CH3]CO",
    "c1ccccc1",               # CHEMBL2 + CHEMBL5 (Kekule)
]

EXPECTED_ROWS_EXAMINED = len(FIXTURE_ROWS)
EXPECTED_NULL_SMILES = 1
EXPECTED_EMPTY_SMILES = 2          # "" and "   "
EXPECTED_PARSE_FAILURES = 2        # "not_a_molecule" and "C(C"
EXPECTED_ZERO_ATOM = 0
EXPECTED_VALID_PRE_DEDUP = 11
EXPECTED_UNIQUE = 8
EXPECTED_DUPLICATES = EXPECTED_VALID_PRE_DEDUP - EXPECTED_UNIQUE


def create_chembl_fixture(
    path: Path,
    rows: list[tuple[int, str, str | None]] | None = None,
    include_compound_dictionary: bool = True,
) -> Path:
    """Write a minimal ChEMBL-shaped SQLite database to `path`.

    Only `compound_structures` (and optionally `molecule_dictionary`) are
    created: no activity, assay, or target table exists in the fixture, so
    a builder that tried to read one would fail rather than quietly
    succeed.
    """
    rows = FIXTURE_ROWS if rows is None else rows

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE compound_structures ("
            "  molregno INTEGER PRIMARY KEY,"
            "  canonical_smiles TEXT,"
            "  standard_inchi TEXT,"
            "  standard_inchi_key TEXT"
            ")"
        )
        connection.executemany(
            "INSERT INTO compound_structures (molregno, canonical_smiles) VALUES (?, ?)",
            [(molregno, smiles) for molregno, _chembl_id, smiles in rows],
        )

        if include_compound_dictionary:
            connection.execute(
                "CREATE TABLE molecule_dictionary ("
                "  molregno INTEGER PRIMARY KEY,"
                "  chembl_id TEXT,"
                "  pref_name TEXT"
                ")"
            )
            connection.executemany(
                "INSERT INTO molecule_dictionary (molregno, chembl_id) VALUES (?, ?)",
                [(molregno, chembl_id) for molregno, chembl_id, _smiles in rows],
            )

        connection.commit()
    finally:
        connection.close()

    return path
