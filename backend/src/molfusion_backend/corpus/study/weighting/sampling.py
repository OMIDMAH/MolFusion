"""A deterministic, size-stratified molecule sample for numerical diagnostics.

The weighting diagnostics do not need all 2,897,639 molecules: they ask
how vector magnitude tracks molecule length and how two TF rules differ,
and a few thousand molecules per size stratum answers that as well as
millions would. What they *do* need is a sample that is reproducible and
that actually contains long molecules -- the corpus is so dominated by
33-64 token molecules that a uniform sample would be ~0.4% long molecules
and the long-molecule question would be answered by noise.

So the sample is stratified by size and drawn by hash, never by line
position (which would leak the corpus's lexicographic sort) and never by
`random` (which would depend on a seed and a Python version). The rule is
the same family as the Phase 5F-C split and is pinned just as tightly.

All document frequencies and IDF values used in the study still come from
the full frozen corpus. The sample is only ever the set of molecules the
vectors are computed *for*.
"""

import hashlib
from collections.abc import Sequence
from typing import Any

SAMPLE_ID = "sha256_utf8_digest_bigendian_unsigned_mod10000_stratified_v1"
SAMPLE_MODULUS = 10_000
SAMPLE_BYTE_ORDER = "big"
SAMPLE_SIGNED = False

# Strata by token count, upper edge inclusive. The names are the ones the
# brief asks the analysis to cover.
STRATUM_SMALL = "small"
STRATUM_TYPICAL = "typical"
STRATUM_LARGE = "large"
STRATUM_VERY_LONG = "very_long"

STRATUM_EDGES = (
    (STRATUM_SMALL, 32),
    (STRATUM_TYPICAL, 64),
    (STRATUM_LARGE, 256),
    (STRATUM_VERY_LONG, None),
)
STRATUM_NAMES = tuple(name for name, _ in STRATUM_EDGES)

# Acceptance rate per stratum, in parts per SAMPLE_MODULUS. Chosen so each
# stratum yields a few thousand molecules given the frozen corpus's size
# distribution, which is extremely uneven: ~2.07M molecules fall in
# `typical` and only ~13k in `very_long`, so a single global rate would
# either drown the study in typical molecules or reduce `very_long` to a
# handful. These are deliberately unequal and the realized counts are
# reported, so nothing here silently reweights a corpus-level statistic --
# every per-stratum figure is read within its stratum.
STRATUM_ACCEPTANCE = {
    STRATUM_SMALL: 150,
    STRATUM_TYPICAL: 25,
    STRATUM_LARGE: 110,
    STRATUM_VERY_LONG: 4000,
}


def sample_bucket(smiles: str) -> int:
    """The molecule's position in [0, SAMPLE_MODULUS), by hash alone."""
    digest = hashlib.sha256(smiles.encode("utf-8")).digest()
    value = int.from_bytes(digest, byteorder=SAMPLE_BYTE_ORDER, signed=SAMPLE_SIGNED)
    return value % SAMPLE_MODULUS


def stratum_for(token_count: int) -> str:
    """Which size stratum a molecule of `token_count` tokens belongs to."""
    for name, edge in STRATUM_EDGES:
        if edge is None or token_count <= edge:
            return name
    return STRATUM_VERY_LONG


def is_sampled(smiles: str, token_count: int) -> bool:
    """True if this molecule is in the diagnostic sample.

    A pure function of the molecule and its own length, so membership does
    not depend on which other molecules exist, on corpus order, or on how
    many have been seen so far.
    """
    return sample_bucket(smiles) < STRATUM_ACCEPTANCE[stratum_for(token_count)]


def sampling_definition() -> dict[str, Any]:
    return {
        "sample_id": SAMPLE_ID,
        "digest": "sha256",
        "digest_input": "canonical SMILES encoded as UTF-8",
        "digest_bytes_used": "all 32",
        "integer_conversion": {"byte_order": SAMPLE_BYTE_ORDER, "signed": SAMPLE_SIGNED},
        "modulus": SAMPLE_MODULUS,
        "strata": [
            {
                "name": name,
                "token_count": (
                    f"<= {edge}" if edge is not None else f"> {STRATUM_EDGES[-2][1]}"
                ),
                "acceptance_per_modulus": STRATUM_ACCEPTANCE[name],
            }
            for name, edge in STRATUM_EDGES
        ],
        "note": (
            "stratified so long molecules are actually represented; per-stratum "
            "rates are unequal by design, so figures are reported within strata "
            "and never pooled into a corpus-level average"
        ),
        "df_and_idf_source": "the full frozen corpus, never the sample",
    }


def empty_strata() -> dict[str, list[Any]]:
    return {name: [] for name in STRATUM_NAMES}


def stratum_sizes(sample: dict[str, Sequence[Any]]) -> dict[str, int]:
    return {name: len(sample.get(name, ())) for name in STRATUM_NAMES}


__all__ = [
    "SAMPLE_ID",
    "SAMPLE_MODULUS",
    "STRATUM_ACCEPTANCE",
    "STRATUM_EDGES",
    "STRATUM_LARGE",
    "STRATUM_NAMES",
    "STRATUM_SMALL",
    "STRATUM_TYPICAL",
    "STRATUM_VERY_LONG",
    "empty_strata",
    "is_sampled",
    "sample_bucket",
    "sampling_definition",
    "stratum_for",
    "stratum_sizes",
]
