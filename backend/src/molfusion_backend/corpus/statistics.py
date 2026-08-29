"""Deterministic corpus-level statistics.

Phase 5F-B scope only: shape of the corpus (how many molecules, how long
they are, how many tokens they carry, how many are salts or carry stereo).
Deliberately no vocabulary, document-frequency, n-gram, or TF-IDF
analysis -- that is Phase 5F-C, and doing it here would bake fitting
decisions into the corpus that is supposed to be independent of them.
"""

import statistics as stdlib_statistics
from array import array
from dataclasses import asdict, dataclass, field
from typing import Any

# Substring probes, valid *only* against canonical SMILES produced by the
# Phase 5F-A normalizer:
#   "." separates disconnected components and appears nowhere else;
#   "@" appears only as a chirality marker inside a bracket atom, and
#   "/" and "\" only as directional (double-bond stereo) bonds.
# They would not be safe against arbitrary SMILES-like text, which is why
# these run after normalization rather than on the raw ChEMBL strings.
_COMPONENT_SEPARATOR = "."
_STEREO_MARKERS = ("@", "/", "\\")

# Unsigned 32-bit: SMILES lengths and token counts are small non-negative
# integers, and an array is ~10x lighter than a list of Python ints across
# millions of records.
_COUNTER_TYPECODE = "I"


def _summarize(values: "array[int]") -> dict[str, Any]:
    """min/max/mean/median over an integer sample, or nulls when empty.

    Deterministic: the sum of Python ints is exact, and the division and
    median that follow are ordinary IEEE-754 operations over a fixed
    multiset, so two builds of the same corpus agree bit for bit. The mean
    is rounded only for report readability -- rounding a deterministic
    value keeps it deterministic.
    """
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}

    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 6),
        "median": stdlib_statistics.median(ordered),
    }


@dataclass
class CorpusStatisticsAccumulator:
    """Accumulates statistics one corpus document at a time.

    Incremental so the builder never needs a second pass over the corpus,
    and so nothing bigger than two integer arrays is retained: the token
    tuple for each document is consumed here and dropped, never stored.
    """

    document_count: int = 0
    disconnected_component_count: int = 0
    stereochemistry_count: int = 0
    _smiles_lengths: "array[int]" = field(
        default_factory=lambda: array(_COUNTER_TYPECODE), repr=False
    )
    _token_counts: "array[int]" = field(
        default_factory=lambda: array(_COUNTER_TYPECODE), repr=False
    )

    def add(self, smiles: str, token_count: int) -> None:
        self.document_count += 1
        self._smiles_lengths.append(len(smiles))
        self._token_counts.append(token_count)

        if _COMPONENT_SEPARATOR in smiles:
            self.disconnected_component_count += 1
        if any(marker in smiles for marker in _STEREO_MARKERS):
            self.stereochemistry_count += 1

    def as_report(self) -> dict[str, Any]:
        return {
            "document_count": self.document_count,
            "smiles_length": _summarize(self._smiles_lengths),
            "token_count": _summarize(self._token_counts),
            "with_disconnected_components": self.disconnected_component_count,
            "with_stereochemistry": self.stereochemistry_count,
        }


@dataclass
class RecordCounts:
    """Full accounting of every source row the builder examined.

    Three identities must hold for every build, and are checked in
    validate() so a record can never vanish without appearing in a
    category:

        rows_examined   = null + empty + parse failures + zero-atom
                          + tokenization failures + valid_pre_dedup
        valid_pre_dedup = duplicate_canonical_smiles
                          + unique_canonical_smiles
        document_count  = unique_canonical_smiles

    `tokenization_failures` counts source rows whose canonical SMILES
    violated the Phase 5F-A lossless invariant. A build aborts on the first
    violation unless failures were explicitly allowed, so this is normally
    0. It sits alongside the other exclusion categories because such a
    record is dropped before deduplication, not after.
    """

    rows_examined: int = 0
    null_smiles: int = 0
    empty_smiles: int = 0
    rdkit_parse_failures: int = 0
    zero_atom_molecules: int = 0
    tokenization_failures: int = 0
    valid_pre_dedup: int = 0
    duplicate_canonical_smiles: int = 0
    unique_canonical_smiles: int = 0
    document_count: int = 0

    def validate(self) -> None:
        excluded = (
            self.null_smiles
            + self.empty_smiles
            + self.rdkit_parse_failures
            + self.zero_atom_molecules
            + self.tokenization_failures
        )
        if excluded + self.valid_pre_dedup != self.rows_examined:
            raise ValueError(
                "Record accounting does not balance: "
                f"{excluded} excluded + {self.valid_pre_dedup} valid "
                f"!= {self.rows_examined} rows examined."
            )

        deduplicated = self.duplicate_canonical_smiles + self.unique_canonical_smiles
        if deduplicated != self.valid_pre_dedup:
            raise ValueError(
                "Deduplication accounting does not balance: "
                f"{self.duplicate_canonical_smiles} duplicates + "
                f"{self.unique_canonical_smiles} unique "
                f"!= {self.valid_pre_dedup} valid records."
            )

        if self.unique_canonical_smiles != self.document_count:
            raise ValueError(
                "Document accounting does not balance: "
                f"{self.unique_canonical_smiles} unique canonical SMILES "
                f"!= {self.document_count} documents."
            )

    def as_report(self) -> dict[str, int]:
        return asdict(self)
