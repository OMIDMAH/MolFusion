from array import array

import pytest

from molfusion_backend.corpus.study.coverage import (
    PERCENTILE_CONVENTION,
    HoldoutCoverageAccumulator,
    VocabularyFamily,
    coverage_definition,
    percentile,
    summarize,
)


def family(name="A-df", orders=(1,), ranked=(("C",), ("O",)), sizes=(1, 2), **kwargs):
    return VocabularyFamily(
        name=name,
        policy=kwargs.pop("policy", "(1,1)"),
        orders=orders,
        ranking=kwargs.pop("ranking", "document_frequency"),
        protected_unigrams=kwargs.pop("protected_unigrams", False),
        sizes=sizes,
        ranked=ranked,
    )


def rows_by_dimension(accumulator):
    return {row["dimension"]: row for row in accumulator.results()}


# ---------------------------------------------------------------------------
# percentiles
# ---------------------------------------------------------------------------


def test_percentile_is_nearest_rank_on_observed_values():
    sample = list(range(10))
    assert percentile(sample, 50) == 4
    assert percentile(sample, 95) == 9
    assert percentile(sample, 99) == 9
    assert percentile(sample, 0) == 0
    assert percentile(sample, 100) == 9


def test_percentile_never_interpolates():
    """Every reported percentile is a value the study actually observed."""
    sample = [0.0, 1.0]
    assert percentile(sample, 50) in sample
    assert percentile(sample, 75) in sample


def test_percentile_of_an_empty_sample_is_none_not_zero():
    assert percentile([], 95) is None


def test_summarize_reports_counts_and_nulls_consistently():
    assert summarize(array("I", [])) == {
        "count": 0,
        "min": None,
        "mean": None,
        "median": None,
        "p95": None,
        "p99": None,
        "max": None,
    }
    summary = summarize(array("I", [1, 2, 3, 4]))
    assert summary["count"] == 4
    assert summary["min"] == 1
    assert summary["max"] == 4
    assert summary["mean"] == 2.5
    assert summary["median"] == 2  # nearest rank: ceil(0.5 * 4) = 2 -> index 1


def test_percentile_convention_is_named_in_the_report():
    assert PERCENTILE_CONVENTION == "nearest_rank"
    assert coverage_definition()["percentile_convention"] == "nearest_rank"


# ---------------------------------------------------------------------------
# family validation
# ---------------------------------------------------------------------------


def test_sizes_must_be_ascending_and_unique():
    with pytest.raises(ValueError):
        family(sizes=(2, 1))
    with pytest.raises(ValueError):
        family(sizes=(1, 1))


def test_a_size_cannot_exceed_the_ranking():
    with pytest.raises(ValueError):
        family(sizes=(1, 5))


def test_vocabulary_is_the_ranking_prefix():
    assert family().vocabulary(1) == [("C",)]
    assert family().vocabulary(2) == [("C",), ("O",)]


# ---------------------------------------------------------------------------
# coverage, OOV, all-zero and density on a pinned tiny holdout
# ---------------------------------------------------------------------------
#
# Vocabulary ranking is [("C",), ("O",)]; candidates are the 1-term and
# 2-term prefixes. Holdout is two molecules:
#
#   ("C","C","O")  -> C x2, O x1   (3 occurrences, 2 distinct)
#   ("N",)         -> N x1         (1 occurrence,  1 distinct, all OOV)


@pytest.fixture()
def scored():
    accumulator = HoldoutCoverageAccumulator([family()])
    accumulator.add_document(("C", "C", "O"))
    accumulator.add_document(("N",))
    return rows_by_dimension(accumulator)


def test_occurrence_coverage_is_aggregate_over_the_whole_holdout(scored):
    assert scored[1]["holdout_occurrences"] == 4
    assert scored[1]["holdout_occurrence_coverage"] == pytest.approx(2 / 4)
    assert scored[2]["holdout_occurrence_coverage"] == pytest.approx(3 / 4)


def test_molecule_oov_fraction_is_per_molecule_then_averaged(scored):
    # ("C","C","O") loses 1 of 3 occurrences at dimension 1; ("N",) loses all.
    assert scored[1]["molecule_oov_fraction"]["mean"] == pytest.approx((1 / 3 + 1.0) / 2)
    assert scored[2]["molecule_oov_fraction"]["mean"] == pytest.approx((0.0 + 1.0) / 2)
    assert scored[2]["molecule_oov_fraction"]["max"] == pytest.approx(1.0)
    assert scored[2]["molecule_oov_fraction"]["min"] == pytest.approx(0.0)


def test_oov_mean_differs_from_aggregate_coverage(scored):
    """Averaging per-molecule fractions is not the same as pooling
    occurrences; conflating them would flatter long molecules."""
    aggregate_oov = 1 - scored[2]["holdout_occurrence_coverage"]
    assert scored[2]["molecule_oov_fraction"]["mean"] != pytest.approx(aggregate_oov)


def test_all_zero_molecules_are_detected(scored):
    assert scored[1]["all_zero_molecules"] == 1
    assert scored[2]["all_zero_molecules"] == 1
    assert scored[2]["all_zero_fraction"] == pytest.approx(0.5)


def test_nonzero_feature_counts_and_sparsity(scored):
    assert scored[1]["nonzero_features"]["mean"] == pytest.approx(0.5)  # [1, 0]
    assert scored[1]["nonzero_features"]["max"] == 1
    assert scored[1]["sparsity_at_mean"] == pytest.approx(1 - 0.5 / 1)
    assert scored[2]["nonzero_features"]["mean"] == pytest.approx(1.0)  # [2, 0]
    assert scored[2]["sparsity_at_mean"] == pytest.approx(1 - 1.0 / 2)


def test_a_term_outside_every_candidate_is_oov_everywhere():
    narrow = family(ranked=(("C",), ("O",)), sizes=(1,))
    accumulator = HoldoutCoverageAccumulator([narrow])
    accumulator.add_document(("O", "O"))

    row = rows_by_dimension(accumulator)[1]
    assert row["holdout_occurrence_coverage"] == 0.0
    assert row["all_zero_molecules"] == 1


def test_candidates_are_nested_so_coverage_is_monotone_in_dimension():
    accumulator = HoldoutCoverageAccumulator([family()])
    for tokens in (("C", "C", "O"), ("O",), ("C", "N"), ("N", "N")):
        accumulator.add_document(tokens)

    scored = rows_by_dimension(accumulator)
    assert scored[1]["holdout_occurrence_coverage"] <= scored[2]["holdout_occurrence_coverage"]
    assert scored[1]["all_zero_molecules"] >= scored[2]["all_zero_molecules"]
    assert scored[1]["nonzero_features"]["mean"] <= scored[2]["nonzero_features"]["mean"]


def test_molecules_with_no_ngram_of_the_policy_order_are_counted_apart():
    """A two-token molecule has no trigram. It is all-zero, but it has no
    OOV *fraction* -- the denominator does not exist, and inventing one
    would bias the distribution."""
    trigrams = family(
        name="C-df",
        policy="(3,3)",
        orders=(3,),
        ranked=(("C", "C", "C"),),
        sizes=(1,),
    )
    accumulator = HoldoutCoverageAccumulator([trigrams])
    accumulator.add_document(("C", "O"))
    accumulator.add_document(("C", "C", "C"))

    row = rows_by_dimension(accumulator)[1]
    assert row["holdout_documents"] == 2
    assert row["holdout_documents_without_ngrams"] == 1
    assert row["molecule_oov_fraction"]["count"] == 1
    assert row["molecule_oov_fraction"]["mean"] == pytest.approx(0.0)
    assert row["all_zero_molecules"] == 1


def test_multiple_families_are_scored_in_one_pass_without_interfering():
    unigrams = family(name="A-df", orders=(1,), ranked=(("C",),), sizes=(1,))
    bigrams = family(
        name="D-df", policy="(2,2)", orders=(2,), ranked=(("C", "C"),), sizes=(1,)
    )
    accumulator = HoldoutCoverageAccumulator([unigrams, bigrams])
    accumulator.add_document(("C", "C", "O"))

    results = {row["family"]: row for row in accumulator.results()}
    assert results["A-df"]["holdout_occurrence_coverage"] == pytest.approx(2 / 3)
    assert results["D-df"]["holdout_occurrence_coverage"] == pytest.approx(1 / 2)


def test_results_are_ordered_deterministically():
    wide = family(sizes=(1, 2))
    accumulator = HoldoutCoverageAccumulator([wide])
    accumulator.add_document(("C", "O"))
    assert [row["dimension"] for row in accumulator.results()] == [1, 2]


def test_scoring_does_not_depend_on_holdout_document_order():
    documents = [("C", "C", "O"), ("N",), ("O", "O"), ("C",)]

    forward = HoldoutCoverageAccumulator([family()])
    for tokens in documents:
        forward.add_document(tokens)
    backward = HoldoutCoverageAccumulator([family()])
    for tokens in reversed(documents):
        backward.add_document(tokens)

    assert forward.results() == backward.results()
