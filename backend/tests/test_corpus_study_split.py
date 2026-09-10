import hashlib

import pytest

from molfusion_backend.corpus.study.split import (
    STUDY_HOLDOUT_BUCKET,
    STUDY_SPLIT_BUCKET_COUNT,
    STUDY_SPLIT_ID,
    is_study_holdout,
    split_definition,
    study_bucket,
)

# Pinned by construction, not by observation: these are what
# int.from_bytes(sha256(smiles.encode()).digest(), "big") % 20 evaluates
# to. A change to the encoding, the digest, the byte order, the signedness,
# or the modulus moves molecules between subsets and silently invalidates
# every coverage number the study reports, so the values are frozen here.
PINNED_BUCKETS = {
    "CCO": 18,
    "CCN": 10,
    "CCC": 18,
    "CO": 17,
    "CN": 8,
    "CS": 3,
    "N": 19,
    "O": 4,
    "CCCl": 14,
    "CCBr": 7,
    "c1ccccc1": 17,
    "CC(=O)O": 6,
    "C[C@H](N)C(=O)O": 17,
    "CC(=O)[O-].[Na+]": 4,
    "I": 0,
    "CCCCCCCCCCCCC": 0,
}


@pytest.mark.parametrize(("smiles", "bucket"), sorted(PINNED_BUCKETS.items()))
def test_study_bucket_is_pinned(smiles, bucket):
    assert study_bucket(smiles) == bucket


def test_bucket_matches_the_documented_algorithm_exactly():
    """The rule stated in the report must be the rule that runs."""
    for smiles in PINNED_BUCKETS:
        digest = hashlib.sha256(smiles.encode("utf-8")).digest()
        expected = int.from_bytes(digest, byteorder="big", signed=False) % 20
        assert study_bucket(smiles) == expected


def test_holdout_is_exactly_bucket_zero():
    assert is_study_holdout("I")
    assert is_study_holdout("CCCCCCCCCCCCC")
    assert not is_study_holdout("CCO")
    assert not is_study_holdout("c1ccccc1")


def test_bucket_is_always_in_range():
    for index in range(500):
        assert 0 <= study_bucket("C" * (index + 1) + "O") < STUDY_SPLIT_BUCKET_COUNT


def test_split_is_independent_of_corpus_order_and_neighbours():
    """Membership is a pure function of the molecule.

    The whole point of hashing rather than slicing: adding, removing, or
    reordering other molecules cannot move this one.
    """
    before = {smiles: study_bucket(smiles) for smiles in PINNED_BUCKETS}
    for filler in range(1000):
        study_bucket(f"C{'C' * filler}N")
    after = {smiles: study_bucket(smiles) for smiles in reversed(list(PINNED_BUCKETS))}
    assert before == after


def test_different_molecules_are_not_forced_apart_or_together():
    """A one-character difference must be able to change the bucket, and
    identical strings must never differ."""
    assert study_bucket("CCO") == study_bucket("CCO")
    assert study_bucket("CCO") != study_bucket("CCN")


def test_split_definition_records_every_pinned_choice():
    definition = split_definition()
    assert definition["split_id"] == STUDY_SPLIT_ID
    assert definition["digest"] == "sha256"
    assert definition["integer_conversion"] == {"byte_order": "big", "signed": False}
    assert definition["bucket_count"] == STUDY_SPLIT_BUCKET_COUNT
    assert definition["holdout_bucket"] == STUDY_HOLDOUT_BUCKET
    assert "not a downstream train/test split" in definition["purpose"]


def test_split_id_is_frozen():
    """The identifier travels with every study report; changing it silently
    would make two incomparable studies look comparable."""
    assert STUDY_SPLIT_ID == "sha256_utf8_digest_bigendian_unsigned_mod20_bucket0_holdout_v1"


def test_holdout_share_is_close_to_one_twentieth():
    sample = [f"C{'C' * index}O" for index in range(20_000)]
    holdout = sum(1 for smiles in sample if is_study_holdout(smiles))
    assert 0.04 < holdout / len(sample) < 0.06
