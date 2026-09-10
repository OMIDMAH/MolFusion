"""Candidate vocabulary construction: policies, pruning, and ranking.

MolFusion selects its own vocabulary rather than delegating to a
vectorizer's `max_features`. The reason is auditability, not distrust: a
frozen production artifact has to be rebuildable years from now, and a
rule stated here ("descending document frequency, ties broken by the
lexicographic token tuple") is a rule anyone can re-apply from the
recorded counts alone. A rule that lives inside a third-party library's
sort is reproducible only for as long as that library's internals do not
change, and its tie behaviour is not part of its public contract.

Two orthogonal knobs are modelled:

    pruning   a minimum document frequency, in absolute molecule counts
    ranking   a total order over the surviving n-grams, optionally cut to
              a fixed dimension

They interact more simply than they look. Under descending-DF ranking the
set {DF >= t} is exactly a prefix of the ranking, so every `min_df`
threshold *is* a dimension cap and vice versa; the two questions collapse
into one. Under TF ranking they genuinely differ, which is itself part of
what the study measures.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from molfusion_backend.corpus.study.ngrams import Ngram, NgramEntry

RANKING_DOCUMENT_FREQUENCY = "document_frequency"
RANKING_TERM_FREQUENCY = "term_frequency"
RANKING_METRICS = (RANKING_DOCUMENT_FREQUENCY, RANKING_TERM_FREQUENCY)

SCOPE_CORPUS = "corpus"
SCOPE_FIT = "fit"

# Absolute molecule counts, never percentages: an absolute threshold is
# reproducible against this fixed corpus by inspection, whereas a
# percentage silently changes meaning if the corpus size ever does.
MIN_DF_THRESHOLDS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)

# Rarity cut points reported per order, to expose a long tail before any
# policy is chosen rather than after.
RARITY_THRESHOLDS = (1, 2, 5, 10, 25, 50)

CANDIDATE_DIMENSIONS = (1024, 2048, 4096, 8192, 16384)


@dataclass(frozen=True)
class NgramPolicy:
    """An n-gram range, named the way the study report refers to it."""

    name: str
    orders: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"({self.orders[0]},{self.orders[-1]})"


POLICY_UNIGRAM = NgramPolicy("A", (1,))
POLICY_UNI_BI = NgramPolicy("B", (1, 2))
POLICY_UNI_BI_TRI = NgramPolicy("C", (1, 2, 3))
POLICY_BI_TRI = NgramPolicy("D", (2, 3))
CANDIDATE_POLICIES = (POLICY_UNIGRAM, POLICY_UNI_BI, POLICY_UNI_BI_TRI, POLICY_BI_TRI)


def frequency(entry: NgramEntry, metric: str, scope: str) -> int:
    """The one frequency number a ranking or threshold is reading.

    Scope matters as much as metric: a study vocabulary must be built from
    the fit subset alone, or the holdout it is scored against is not
    unseen. Corpus scope exists for the descriptive tables, which describe
    the whole frozen corpus.
    """
    if scope == SCOPE_FIT:
        if metric == RANKING_DOCUMENT_FREQUENCY:
            return entry.document_frequency_fit
        if metric == RANKING_TERM_FREQUENCY:
            return entry.term_frequency_fit
    elif scope == SCOPE_CORPUS:
        if metric == RANKING_DOCUMENT_FREQUENCY:
            return entry.document_frequency
        if metric == RANKING_TERM_FREQUENCY:
            return entry.term_frequency
    else:
        raise ValueError(f"unknown frequency scope: {scope!r}")
    raise ValueError(f"unknown ranking metric: {metric!r}")


def ranking_sort_key(entry: NgramEntry, metric: str, scope: str) -> tuple[int, Ngram]:
    """`(-frequency, ngram)` -- the frozen deterministic ranking key.

    The tie-break is the token tuple itself, compared element by element,
    so it is total (no two distinct n-grams can tie) and independent of
    insertion order, hash seed, locale, and process scheduling. Tuples of
    different lengths compare cleanly: ("C",) sorts before ("C","C").
    """
    return (-frequency(entry, metric, scope), entry.ngram)


def rank_entries(
    entries: Iterable[NgramEntry], metric: str, scope: str = SCOPE_FIT
) -> list[NgramEntry]:
    """Entries in descending frequency order with the lexicographic tie-break."""
    return sorted(entries, key=lambda entry: ranking_sort_key(entry, metric, scope))


def select_orders(entries: Iterable[NgramEntry], orders: Sequence[int]) -> list[NgramEntry]:
    wanted = set(orders)
    return [entry for entry in entries if entry.order in wanted]


def min_df_prefix_size(ranked: Sequence[NgramEntry], min_df: int, scope: str = SCOPE_FIT) -> int:
    """How many leading entries of a DF ranking satisfy `DF >= min_df`.

    Only meaningful for a ranking produced with RANKING_DOCUMENT_FREQUENCY:
    that ordering is non-increasing in DF, so the survivors are exactly a
    prefix and a count fully describes them.
    """
    size = 0
    for entry in ranked:
        if frequency(entry, RANKING_DOCUMENT_FREQUENCY, scope) < min_df:
            break
        size += 1
    return size


def apply_min_df(
    entries: Iterable[NgramEntry], min_df: int, scope: str = SCOPE_FIT
) -> list[NgramEntry]:
    return [
        entry
        for entry in entries
        if frequency(entry, RANKING_DOCUMENT_FREQUENCY, scope) >= min_df
    ]


def rarity_histogram(
    entries: Sequence[NgramEntry],
    thresholds: Sequence[int] = RARITY_THRESHOLDS,
    scope: str = SCOPE_CORPUS,
) -> dict[int, int]:
    """`{threshold: how many distinct n-grams have DF <= threshold}`.

    Cumulative rather than bucketed because the question this answers is
    "how much of the vocabulary is too rare to be worth a dimension", and
    that is inherently a running total.
    """
    return {
        threshold: sum(
            1
            for entry in entries
            if frequency(entry, RANKING_DOCUMENT_FREQUENCY, scope) <= threshold
        )
        for threshold in thresholds
    }


def unigram_protected_ranking(ranked: Sequence[NgramEntry]) -> list[NgramEntry]:
    """The same ranking, with every unigram lifted to the front.

    The alternative pruning policy of section 12: capacity is given to all
    order-1 tokens first and only the remainder is auctioned off to higher
    orders. Within each group the original ranking order is preserved, so
    the result is still fully deterministic. Whether this is *needed* is
    an empirical question -- it matters only if a global cap would
    otherwise drop a chemically real atom or bracket token.
    """
    unigrams = [entry for entry in ranked if entry.order == 1]
    higher = [entry for entry in ranked if entry.order != 1]
    return unigrams + higher


def unigram_retention(ranked: Sequence[NgramEntry], size: int) -> dict[str, int]:
    """How many order-1 tokens survive a global top-`size` cut."""
    total = sum(1 for entry in ranked if entry.order == 1)
    retained = sum(1 for entry in ranked[:size] if entry.order == 1)
    return {
        "unigrams_total": total,
        "unigrams_retained": retained,
        "unigrams_excluded": total - retained,
    }


def ranking_definition() -> dict[str, object]:
    return {
        "primary_key": "descending frequency (document or term, as named per candidate)",
        "tie_break": "ascending lexicographic n-gram token tuple",
        "sort_key": "(-frequency, ngram_tuple)",
        "total_order": True,
        "depends_on_sklearn": False,
        "note": (
            "Under document-frequency ranking, {DF >= min_df} is exactly a prefix "
            "of the ranking, so every min_df threshold is also a dimension cap."
        ),
    }


__all__ = [
    "CANDIDATE_DIMENSIONS",
    "CANDIDATE_POLICIES",
    "MIN_DF_THRESHOLDS",
    "NgramPolicy",
    "POLICY_BI_TRI",
    "POLICY_UNIGRAM",
    "POLICY_UNI_BI",
    "POLICY_UNI_BI_TRI",
    "RANKING_DOCUMENT_FREQUENCY",
    "RANKING_METRICS",
    "RANKING_TERM_FREQUENCY",
    "RARITY_THRESHOLDS",
    "SCOPE_CORPUS",
    "SCOPE_FIT",
    "apply_min_df",
    "frequency",
    "min_df_prefix_size",
    "rank_entries",
    "ranking_definition",
    "ranking_sort_key",
    "rarity_histogram",
    "select_orders",
    "unigram_protected_ranking",
    "unigram_retention",
]
