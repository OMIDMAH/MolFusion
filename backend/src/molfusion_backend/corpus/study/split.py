"""The deterministic, corpus-only study split.

Phase 5F-C needs an estimate of how a vocabulary fitted on ChEMBL behaves
on molecules it has never seen. The only honest way to get that without
touching a downstream ADMET dataset is to hold part of the reference
corpus back, so this module defines a split that depends on *nothing* but
the canonical SMILES string itself.

Deliberately not a positional or random split:

  * position would leak the corpus's lexicographic sort -- a contiguous
    slice of a sorted corpus is a slice of chemical space, not a sample of
    it, and bucket membership would silently change if the corpus were
    ever re-sorted;
  * `random` would make the split depend on a seed, an implementation, and
    a Python version rather than on the data.

Hashing the SMILES instead makes membership a pure function of the
molecule: the same molecule lands in the same bucket forever, on any
platform, in any order, and adding or removing other molecules cannot move
it. Every step of the integer conversion is pinned below, because a
byte-order or signedness change would silently reshuffle the split while
still "looking" deterministic.

This is an ANALYSIS split only. The Phase 5F-D production artifact is
still fitted on all 2,897,639 reference molecules; nothing here is a
train/test split for any downstream benchmark.
"""

import hashlib

# Pinned end to end: UTF-8 encoding of the canonical SMILES, the full
# 32-byte SHA-256 digest, big-endian unsigned integer conversion, modulo
# 20. Bump the version suffix if any one of those changes.
STUDY_SPLIT_ID = "sha256_utf8_digest_bigendian_unsigned_mod20_bucket0_holdout_v1"
STUDY_SPLIT_BUCKET_COUNT = 20
STUDY_HOLDOUT_BUCKET = 0
STUDY_SPLIT_DIGEST = "sha256"
STUDY_SPLIT_BYTE_ORDER = "big"
STUDY_SPLIT_SIGNED = False


def study_bucket(smiles: str) -> int:
    """The molecule's study bucket in [0, STUDY_SPLIT_BUCKET_COUNT).

    Uses the whole digest rather than a prefix: taking only the first 8
    bytes would be just as deterministic but throws away entropy for no
    gain, and "which prefix" is one more thing that could drift.
    """
    digest = hashlib.sha256(smiles.encode("utf-8")).digest()
    value = int.from_bytes(digest, byteorder=STUDY_SPLIT_BYTE_ORDER, signed=STUDY_SPLIT_SIGNED)
    return value % STUDY_SPLIT_BUCKET_COUNT


def is_study_holdout(smiles: str) -> bool:
    """True for the ~5% study holdout, False for the ~95% study fit subset."""
    return study_bucket(smiles) == STUDY_HOLDOUT_BUCKET


def split_definition() -> dict[str, object]:
    """The split rule as report-ready data, so a study report states the
    exact algorithm rather than only its outcome."""
    return {
        "split_id": STUDY_SPLIT_ID,
        "digest": STUDY_SPLIT_DIGEST,
        "digest_input": "canonical SMILES encoded as UTF-8",
        "digest_bytes_used": "all 32",
        "integer_conversion": {
            "byte_order": STUDY_SPLIT_BYTE_ORDER,
            "signed": STUDY_SPLIT_SIGNED,
        },
        "bucket_count": STUDY_SPLIT_BUCKET_COUNT,
        "holdout_bucket": STUDY_HOLDOUT_BUCKET,
        "fit_buckets": f"1-{STUDY_SPLIT_BUCKET_COUNT - 1}",
        "purpose": "analysis-only unseen-molecule estimate; not a downstream train/test split",
    }


__all__ = [
    "STUDY_HOLDOUT_BUCKET",
    "STUDY_SPLIT_BUCKET_COUNT",
    "STUDY_SPLIT_BYTE_ORDER",
    "STUDY_SPLIT_DIGEST",
    "STUDY_SPLIT_ID",
    "STUDY_SPLIT_SIGNED",
    "is_study_holdout",
    "split_definition",
    "study_bucket",
]
