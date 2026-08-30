import hashlib

import pytest

from molfusion_backend.corpus.study.weighting.sampling import (
    SAMPLE_ID,
    SAMPLE_MODULUS,
    STRATUM_ACCEPTANCE,
    STRATUM_LARGE,
    STRATUM_NAMES,
    STRATUM_SMALL,
    STRATUM_TYPICAL,
    STRATUM_VERY_LONG,
    empty_strata,
    is_sampled,
    sample_bucket,
    sampling_definition,
    stratum_for,
    stratum_sizes,
)

# Pinned by construction: int.from_bytes(sha256(smiles), "big") % 10000.
PINNED_BUCKETS = {
    "CCO": 2618,
    "CCN": 4390,
    "c1ccccc1": 7297,
    "CC(=O)O": 2986,
    "N": 7419,
    "O": 5484,
    "I": 6540,
    "CCCl": 7114,
}


@pytest.mark.parametrize(("smiles", "bucket"), sorted(PINNED_BUCKETS.items()))
def test_sample_bucket_is_pinned(smiles, bucket):
    assert sample_bucket(smiles) == bucket


def test_bucket_matches_the_documented_algorithm():
    for smiles in PINNED_BUCKETS:
        digest = hashlib.sha256(smiles.encode("utf-8")).digest()
        assert sample_bucket(smiles) == (
            int.from_bytes(digest, byteorder="big", signed=False) % 10_000
        )


def test_bucket_is_always_in_range():
    for index in range(1000):
        assert 0 <= sample_bucket(f"C{'C' * index}O") < SAMPLE_MODULUS


def test_sample_id_is_frozen():
    assert SAMPLE_ID == "sha256_utf8_digest_bigendian_unsigned_mod10000_stratified_v1"


# ---------------------------------------------------------------------------
# strata
# ---------------------------------------------------------------------------


def test_stratum_edges_are_upper_inclusive():
    assert stratum_for(1) == STRATUM_SMALL
    assert stratum_for(32) == STRATUM_SMALL
    assert stratum_for(33) == STRATUM_TYPICAL
    assert stratum_for(64) == STRATUM_TYPICAL
    assert stratum_for(65) == STRATUM_LARGE
    assert stratum_for(256) == STRATUM_LARGE
    assert stratum_for(257) == STRATUM_VERY_LONG
    assert stratum_for(1617) == STRATUM_VERY_LONG


def test_every_stratum_has_an_acceptance_rate():
    assert set(STRATUM_ACCEPTANCE) == set(STRATUM_NAMES)
    assert all(0 < rate <= SAMPLE_MODULUS for rate in STRATUM_ACCEPTANCE.values())


def test_long_molecules_are_accepted_far_more_often_than_typical_ones():
    """The whole reason for stratifying: a uniform rate would leave the
    long-molecule question answered by a handful of molecules."""
    assert STRATUM_ACCEPTANCE[STRATUM_VERY_LONG] > STRATUM_ACCEPTANCE[STRATUM_TYPICAL] * 100


# ---------------------------------------------------------------------------
# membership
# ---------------------------------------------------------------------------


def test_membership_is_bucket_below_the_stratum_rate():
    for smiles, bucket in PINNED_BUCKETS.items():
        for token_count, stratum in ((10, STRATUM_SMALL), (50, STRATUM_TYPICAL), (300, STRATUM_VERY_LONG)):
            assert is_sampled(smiles, token_count) == (bucket < STRATUM_ACCEPTANCE[stratum])


def test_the_same_molecule_can_change_stratum_and_therefore_membership():
    """Membership depends on the molecule *and* its own length, which is
    what makes the stratified rates apply per stratum rather than globally."""
    smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCC"
    assert sample_bucket(smiles) == 106
    assert is_sampled(smiles, 26)  # small: rate 150 > 106
    assert not is_sampled(smiles, 50)  # typical: rate 25 < 106
    assert is_sampled(smiles, 300)  # very_long: rate 4000 > 106


def test_membership_does_not_depend_on_other_molecules():
    before = {s: is_sampled(s, 50) for s in PINNED_BUCKETS}
    for filler in range(500):
        sample_bucket(f"N{'C' * filler}O")
    after = {s: is_sampled(s, 50) for s in reversed(list(PINNED_BUCKETS))}
    assert before == after


def test_realized_rate_is_close_to_the_nominal_rate():
    population = [f"C{'C' * index}O" for index in range(40_000)]
    sampled = sum(1 for smiles in population if is_sampled(smiles, 10))
    realized = sampled / len(population) * SAMPLE_MODULUS
    assert 0.7 * STRATUM_ACCEPTANCE[STRATUM_SMALL] < realized < 1.3 * STRATUM_ACCEPTANCE[STRATUM_SMALL]


# ---------------------------------------------------------------------------
# helpers and reporting
# ---------------------------------------------------------------------------


def test_empty_strata_covers_every_name():
    assert set(empty_strata()) == set(STRATUM_NAMES)
    assert all(values == [] for values in empty_strata().values())


def test_stratum_sizes_counts_each_bucket():
    assert stratum_sizes({STRATUM_SMALL: ["a", "b"], STRATUM_TYPICAL: ["c"]}) == {
        STRATUM_SMALL: 2,
        STRATUM_TYPICAL: 1,
        STRATUM_LARGE: 0,
        STRATUM_VERY_LONG: 0,
    }


def test_definition_records_every_pinned_choice_and_the_df_source():
    definition = sampling_definition()
    assert definition["sample_id"] == SAMPLE_ID
    assert definition["digest"] == "sha256"
    assert definition["integer_conversion"] == {"byte_order": "big", "signed": False}
    assert definition["modulus"] == SAMPLE_MODULUS
    assert len(definition["strata"]) == len(STRATUM_NAMES)
    assert definition["df_and_idf_source"] == "the full frozen corpus, never the sample"
