import pytest

from molfusion_backend.corpus.study.ngrams import NgramEntry, NgramFrequencyAccumulator
from molfusion_backend.corpus.study.vocabulary import (
    CANDIDATE_POLICIES,
    MIN_DF_THRESHOLDS,
    POLICY_BI_TRI,
    POLICY_UNI_BI,
    POLICY_UNI_BI_TRI,
    POLICY_UNIGRAM,
    RANKING_DOCUMENT_FREQUENCY,
    RANKING_TERM_FREQUENCY,
    SCOPE_CORPUS,
    SCOPE_FIT,
    apply_min_df,
    frequency,
    min_df_prefix_size,
    rank_entries,
    ranking_definition,
    ranking_sort_key,
    rarity_histogram,
    select_orders,
    unigram_protected_ranking,
    unigram_retention,
)


def entry(ngram, document_frequency, term_frequency):
    """A minimal entry whose fit counts equal its corpus counts."""
    return NgramEntry(
        ngram=ngram,
        order=len(ngram),
        document_frequency=document_frequency,
        term_frequency=term_frequency,
        document_frequency_fit=document_frequency,
        term_frequency_fit=term_frequency,
        document_frequency_holdout=0,
        term_frequency_holdout=0,
        document_frequency_fit_bands=(document_frequency, 0, 0, 0, 0, 0),
        term_frequency_fit_bands=(term_frequency, 0, 0, 0, 0, 0),
    )


# Deliberately constructed so DF order and TF order disagree: ("N",) is in
# few molecules but repeats heavily inside them, exactly the long-molecule
# effect the study has to be able to see.
SAMPLE = [
    entry(("C",), 100, 400),
    entry(("O",), 80, 90),
    entry(("N",), 10, 900),
    entry(("F",), 10, 12),
    entry(("C", "C"), 60, 200),
    entry(("C", "O"), 60, 70),
    entry(("Cl",), 3, 3),
    entry(("C", "C", "C"), 1, 1),
]


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


def test_df_ranking_is_descending_with_lexicographic_tie_break():
    ranked = [e.ngram for e in rank_entries(SAMPLE, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)]
    assert ranked == [
        ("C",),  # df 100
        ("O",),  # df  80
        ("C", "C"),  # df 60, tie broken lexicographically...
        ("C", "O"),  # ...("C","C") < ("C","O")
        ("F",),  # df 10, tie broken lexicographically...
        ("N",),  # ...("F",) < ("N",)
        ("Cl",),  # df 3
        ("C", "C", "C"),  # df 1
    ]


def test_tf_ranking_is_pinned_separately_and_differs_from_df():
    ranked = [e.ngram for e in rank_entries(SAMPLE, RANKING_TERM_FREQUENCY, SCOPE_FIT)]
    assert ranked == [
        ("N",),  # tf 900 -- rare but repetitive
        ("C",),  # tf 400
        ("C", "C"),  # tf 200
        ("O",),  # tf  90
        ("C", "O"),  # tf  70
        ("F",),  # tf  12
        ("Cl",),  # tf   3
        ("C", "C", "C"),  # tf   1
    ]
    assert ranked[0] == ("N",)
    assert rank_entries(SAMPLE, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)[0].ngram == ("C",)


def test_sort_key_is_exactly_negative_frequency_then_tuple():
    target = entry(("C", "O"), 60, 70)
    assert ranking_sort_key(target, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT) == (-60, ("C", "O"))
    assert ranking_sort_key(target, RANKING_TERM_FREQUENCY, SCOPE_FIT) == (-70, ("C", "O"))


def test_tie_break_is_total_so_ranking_is_unique():
    """Two distinct n-grams can never tie, so no ranking position is ever
    resolved by insertion order, hash seed, or sort stability."""
    keys = [ranking_sort_key(e, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT) for e in SAMPLE]
    assert len(set(keys)) == len(keys)


def test_tuples_of_different_lengths_order_cleanly():
    entries = [entry(("C", "C"), 5, 5), entry(("C",), 5, 5)]
    assert [e.ngram for e in rank_entries(entries, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)] == [
        ("C",),
        ("C", "C"),
    ]


def test_ranking_does_not_depend_on_input_order():
    forward = rank_entries(SAMPLE, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)
    backward = rank_entries(list(reversed(SAMPLE)), RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)
    assert [e.ngram for e in forward] == [e.ngram for e in backward]


def test_unknown_metric_or_scope_is_rejected():
    with pytest.raises(ValueError):
        frequency(SAMPLE[0], "popularity", SCOPE_FIT)
    with pytest.raises(ValueError):
        frequency(SAMPLE[0], RANKING_DOCUMENT_FREQUENCY, "everything")


def test_fit_and_corpus_scope_read_different_numbers():
    mixed = NgramFrequencyAccumulator((1,))
    mixed.add_document(("C", "C"), holdout=False)
    mixed.add_document(("C",), holdout=True)
    only = mixed.entries(1)[0]

    assert frequency(only, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT) == 1
    assert frequency(only, RANKING_DOCUMENT_FREQUENCY, SCOPE_CORPUS) == 2
    assert frequency(only, RANKING_TERM_FREQUENCY, SCOPE_FIT) == 2
    assert frequency(only, RANKING_TERM_FREQUENCY, SCOPE_CORPUS) == 3


# ---------------------------------------------------------------------------
# min_df
# ---------------------------------------------------------------------------


def test_min_df_retains_exactly_the_expected_vocabulary():
    assert {e.ngram for e in apply_min_df(SAMPLE, 10, SCOPE_FIT)} == {
        ("C",),
        ("O",),
        ("N",),
        ("F",),
        ("C", "C"),
        ("C", "O"),
    }
    assert {e.ngram for e in apply_min_df(SAMPLE, 60, SCOPE_FIT)} == {
        ("C",),
        ("O",),
        ("C", "C"),
        ("C", "O"),
    }
    assert {e.ngram for e in apply_min_df(SAMPLE, 101, SCOPE_FIT)} == set()


def test_min_df_of_one_keeps_everything():
    assert len(apply_min_df(SAMPLE, 1, SCOPE_FIT)) == len(SAMPLE)


def test_min_df_boundary_is_inclusive():
    """min_df = t keeps DF == t; the threshold is ">=", not ">"."""
    assert any(e.ngram == ("Cl",) for e in apply_min_df(SAMPLE, 3, SCOPE_FIT))
    assert not any(e.ngram == ("Cl",) for e in apply_min_df(SAMPLE, 4, SCOPE_FIT))


def test_min_df_set_is_exactly_a_prefix_of_the_df_ranking():
    """The structural fact the whole candidate sweep relies on."""
    ranked = rank_entries(SAMPLE, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)
    for threshold in (1, 2, 3, 10, 60, 80, 100, 101):
        size = min_df_prefix_size(ranked, threshold, SCOPE_FIT)
        prefix = {e.ngram for e in ranked[:size]}
        assert prefix == {e.ngram for e in apply_min_df(SAMPLE, threshold, SCOPE_FIT)}


def test_min_df_prefix_size_counts_survivors():
    ranked = rank_entries(SAMPLE, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)
    assert min_df_prefix_size(ranked, 1, SCOPE_FIT) == 8
    assert min_df_prefix_size(ranked, 10, SCOPE_FIT) == 6
    assert min_df_prefix_size(ranked, 1000, SCOPE_FIT) == 0


def test_thresholds_are_absolute_counts_not_fractions():
    assert all(isinstance(t, int) and t >= 1 for t in MIN_DF_THRESHOLDS)
    assert MIN_DF_THRESHOLDS[0] == 1
    assert max(MIN_DF_THRESHOLDS) == 1000


# ---------------------------------------------------------------------------
# rarity
# ---------------------------------------------------------------------------


def test_rarity_histogram_is_cumulative():
    histogram = rarity_histogram(SAMPLE, (1, 2, 5, 10), SCOPE_CORPUS)
    assert histogram == {1: 1, 2: 1, 5: 2, 10: 4}


# ---------------------------------------------------------------------------
# top-K and unigram protection
# ---------------------------------------------------------------------------


def test_top_k_is_a_deterministic_prefix():
    ranked = rank_entries(SAMPLE, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)
    assert [e.ngram for e in ranked[:3]] == [("C",), ("O",), ("C", "C")]
    assert [e.ngram for e in ranked[:3]] == [e.ngram for e in ranked][:3]


def test_unigram_retention_reports_exclusions_at_a_cap():
    ranked = rank_entries(SAMPLE, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)
    assert unigram_retention(ranked, 3) == {
        "unigrams_total": 5,
        "unigrams_retained": 2,
        "unigrams_excluded": 3,
    }
    assert unigram_retention(ranked, len(ranked)) == {
        "unigrams_total": 5,
        "unigrams_retained": 5,
        "unigrams_excluded": 0,
    }


def test_a_global_cap_can_displace_a_unigram_with_a_bigram():
    """The exact risk section 12 asks about: ("C","C") outranks ("F",)."""
    ranked = [e.ngram for e in rank_entries(SAMPLE, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)]
    assert ranked.index(("C", "C")) < ranked.index(("F",))


def test_unigram_protection_lifts_every_unigram_without_reshuffling():
    ranked = rank_entries(SAMPLE, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)
    protected = unigram_protected_ranking(ranked)

    assert [e.ngram for e in protected[:5]] == [("C",), ("O",), ("F",), ("N",), ("Cl",)]
    assert [e.ngram for e in protected[5:]] == [("C", "C"), ("C", "O"), ("C", "C", "C")]
    # Order within each group is the original ranking order, unchanged.
    assert [e.ngram for e in protected if e.order == 1] == [
        e.ngram for e in ranked if e.order == 1
    ]
    assert unigram_retention(protected, 5)["unigrams_excluded"] == 0


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------


def test_candidate_policies_cover_the_four_required_ranges():
    assert [policy.label for policy in CANDIDATE_POLICIES] == [
        "(1,1)",
        "(1,2)",
        "(1,3)",
        "(2,3)",
    ]
    assert POLICY_UNIGRAM.orders == (1,)
    assert POLICY_UNI_BI.orders == (1, 2)
    assert POLICY_UNI_BI_TRI.orders == (1, 2, 3)
    assert POLICY_BI_TRI.orders == (2, 3)


def test_select_orders_filters_by_ngram_order():
    assert {e.ngram for e in select_orders(SAMPLE, (1,))} == {
        ("C",),
        ("O",),
        ("N",),
        ("F",),
        ("Cl",),
    }
    assert {e.ngram for e in select_orders(SAMPLE, (2, 3))} == {
        ("C", "C"),
        ("C", "O"),
        ("C", "C", "C"),
    }


def test_ranking_definition_states_the_rule_and_disclaims_sklearn():
    definition = ranking_definition()
    assert definition["sort_key"] == "(-frequency, ngram_tuple)"
    assert definition["tie_break"] == "ascending lexicographic n-gram token tuple"
    assert definition["depends_on_sklearn"] is False
    assert definition["total_order"] is True
