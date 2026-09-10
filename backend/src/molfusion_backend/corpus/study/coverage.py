"""Holdout coverage, OOV burden, and expected vector density.

Scoring a candidate vocabulary means asking, for every molecule the
vocabulary was not fitted on: how much of it survives? Three answers
matter, and they are not the same answer:

    occurrence coverage   how much of the molecule's n-gram *mass* is
                          representable -- the quantity a TF-IDF vector
                          actually carries;
    unique coverage       how many *distinct* motifs are representable --
                          insensitive to a common motif repeated often;
    all-zero risk         whether the molecule survives at all. A molecule
                          reduced to an all-zero vector is not "poorly
                          represented", it is indistinguishable from every
                          other all-zero molecule, so this is a hard
                          quality gate rather than one more average.

The evaluation exploits one structural fact to stay affordable. For a
fixed ranking, every candidate vocabulary is a prefix of that ranking, so
the candidates are nested. Recording which prefix an n-gram *first* enters
therefore answers every candidate dimension at once: one dictionary lookup
per n-gram per molecule, then a running total, instead of one membership
test per candidate. That is what keeps a 68-candidate sweep over ~145k
holdout molecules to a single pass.
"""

import math
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from molfusion_backend.corpus.study.ngrams import Ngram, document_ngram_counts

# Percentiles are nearest-rank on the observed sample: the p-th percentile
# is the ceil(p/100 * n)-th smallest value, an actually observed value with
# no interpolation. Stated explicitly because "the 95th percentile" is not
# a single definition, and a study report that does not pin one is not
# reproducible.
PERCENTILE_CONVENTION = "nearest_rank"
REPORTED_PERCENTILES = (50, 95, 99)


def percentile(sorted_values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile of an already-sorted sample."""
    count = len(sorted_values)
    if count == 0:
        return None
    rank = max(1, math.ceil(p / 100.0 * count))
    return sorted_values[min(rank, count) - 1]


def summarize(values: "array[Any]") -> dict[str, Any]:
    """min / mean / nearest-rank percentiles / max over a numeric sample."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "max": ordered[-1],
    }


@dataclass(frozen=True)
class VocabularyFamily:
    """A ranking plus the ascending prefix sizes to be scored against it.

    `ranked` is the full ranking; `sizes` names the prefixes of interest.
    Every size must be <= len(ranked), and sizes must be strictly
    ascending -- the nesting the accumulator relies on is only meaningful
    for an ordered, deduplicated list.
    """

    name: str
    policy: str
    orders: tuple[int, ...]
    ranking: str
    protected_unigrams: bool
    sizes: tuple[int, ...]
    ranked: tuple[Ngram, ...]

    def __post_init__(self) -> None:
        if list(self.sizes) != sorted(set(self.sizes)):
            raise ValueError(f"{self.name}: sizes must be strictly ascending and unique")
        if self.sizes and self.sizes[-1] > len(self.ranked):
            raise ValueError(
                f"{self.name}: size {self.sizes[-1]} exceeds ranking of {len(self.ranked)}"
            )

    def vocabulary(self, size: int) -> list[Ngram]:
        """The candidate vocabulary of the given dimension, in rank order."""
        return list(self.ranked[:size])


@dataclass
class _FamilyState:
    """Per-candidate running totals for one family."""

    family: VocabularyFamily
    occurrences_total: int = 0
    distinct_total: int = 0
    documents: int = 0
    documents_without_ngrams: int = 0
    bucket_of: dict[Ngram, int] = field(default_factory=dict)
    covered_occurrences: list[int] = field(default_factory=list)
    covered_distinct: list[int] = field(default_factory=list)
    all_zero: list[int] = field(default_factory=list)
    oov_sum: list[float] = field(default_factory=list)
    oov_samples: list["array[float]"] = field(default_factory=list)
    nonzero_samples: list["array[int]"] = field(default_factory=list)

    def __post_init__(self) -> None:
        sizes = self.family.sizes
        # An n-gram's bucket is the index of the smallest candidate
        # vocabulary that contains it; n-grams past the largest candidate
        # are simply absent, which makes them OOV everywhere.
        bucket = 0
        for rank, ngram in enumerate(self.family.ranked):
            while bucket < len(sizes) and rank >= sizes[bucket]:
                bucket += 1
            if bucket >= len(sizes):
                break
            self.bucket_of[ngram] = bucket
        width = len(sizes)
        self.covered_occurrences = [0] * width
        self.covered_distinct = [0] * width
        self.all_zero = [0] * width
        self.oov_sum = [0.0] * width
        # Percentile samples are kept as float32: ~145k holdout molecules
        # times ~70 candidates is 40 MB at this width and twice that at
        # float64, and a percentile reported to six decimals does not need
        # more. The headline mean is accumulated separately in float64
        # (`oov_sum`) so it is not limited by the sample's storage width.
        self.oov_samples = [array("f") for _ in range(width)]
        self.nonzero_samples = [array("I") for _ in range(width)]


class HoldoutCoverageAccumulator:
    """Scores every candidate vocabulary against the holdout in one pass."""

    def __init__(self, families: Sequence[VocabularyFamily]) -> None:
        self._states = [_FamilyState(family) for family in families]

    def add_document(self, tokens: Sequence[str]) -> None:
        # Cached per order so policies sharing an order (B, C and D all
        # use order 2) tokenize the molecule's n-grams once, not once each.
        counts_by_order: dict[int, dict[Ngram, int]] = {}

        for state in self._states:
            counts: dict[Ngram, int] = {}
            for order in state.family.orders:
                per_order = counts_by_order.get(order)
                if per_order is None:
                    per_order = counts_by_order[order] = document_ngram_counts(tokens, order)
                counts.update(per_order)

            occurrences = sum(counts.values())
            distinct = len(counts)
            state.documents += 1
            state.occurrences_total += occurrences
            state.distinct_total += distinct
            if occurrences == 0:
                # Too short to contain any n-gram of this policy's orders.
                # It is genuinely unrepresentable, so it counts as all-zero
                # for every candidate, but it has no OOV *fraction* -- the
                # denominator does not exist and inventing 0.0 or 1.0 would
                # bias the distribution either way.
                state.documents_without_ngrams += 1
                for index in range(len(state.family.sizes)):
                    state.all_zero[index] += 1
                    state.nonzero_samples[index].append(0)
                continue

            width = len(state.family.sizes)
            occurrence_buckets = [0] * width
            distinct_buckets = [0] * width
            bucket_of = state.bucket_of
            for ngram, count in counts.items():
                bucket = bucket_of.get(ngram)
                if bucket is not None:
                    occurrence_buckets[bucket] += count
                    distinct_buckets[bucket] += 1

            running_occurrences = 0
            running_distinct = 0
            for index in range(width):
                running_occurrences += occurrence_buckets[index]
                running_distinct += distinct_buckets[index]
                state.covered_occurrences[index] += running_occurrences
                state.covered_distinct[index] += running_distinct
                oov = 1.0 - running_occurrences / occurrences
                state.oov_sum[index] += oov
                state.oov_samples[index].append(oov)
                state.nonzero_samples[index].append(running_distinct)
                if running_distinct == 0:
                    state.all_zero[index] += 1

    def results(self) -> list[dict[str, Any]]:
        """One row per (family, candidate dimension), deterministically ordered."""
        rows = []
        for state in self._states:
            family = state.family
            for index, size in enumerate(family.sizes):
                scored = state.documents - state.documents_without_ngrams
                oov = summarize(state.oov_samples[index])
                nonzero = summarize(state.nonzero_samples[index])
                rows.append(
                    {
                        "family": family.name,
                        "policy": family.policy,
                        "ngram_orders": list(family.orders),
                        "ranking": family.ranking,
                        "protected_unigrams": family.protected_unigrams,
                        "dimension": size,
                        "holdout_documents": state.documents,
                        "holdout_documents_without_ngrams": state.documents_without_ngrams,
                        "holdout_occurrences": state.occurrences_total,
                        "holdout_occurrence_coverage": (
                            state.covered_occurrences[index] / state.occurrences_total
                            if state.occurrences_total
                            else None
                        ),
                        "molecule_oov_fraction": {
                            **oov,
                            # Exact float64 running sum, not the float32
                            # percentile sample, so the headline average is
                            # not limited by the sample's storage width.
                            "mean": (state.oov_sum[index] / scored) if scored else None,
                        },
                        "nonzero_features": nonzero,
                        "sparsity_at_mean": (
                            1.0 - (nonzero["mean"] / size) if nonzero["mean"] is not None else None
                        ),
                        "all_zero_molecules": state.all_zero[index],
                        "all_zero_fraction": (
                            state.all_zero[index] / state.documents if state.documents else None
                        ),
                    }
                )
        return rows


def coverage_definition() -> dict[str, Any]:
    return {
        "percentile_convention": PERCENTILE_CONVENTION,
        "percentiles_reported": list(REPORTED_PERCENTILES),
        "occurrence_coverage": (
            "retained n-gram occurrences / all n-gram occurrences, over the whole holdout"
        ),
        "molecule_oov_fraction": (
            "per molecule: 1 - retained occurrences / its own occurrences; "
            "molecules with no n-gram of the policy's orders are excluded from "
            "the distribution and counted separately"
        ),
        "nonzero_features": "distinct retained vocabulary terms per holdout molecule",
        "sparsity_at_mean": "1 - mean nonzero features / vocabulary dimension",
        "all_zero_molecules": "holdout molecules retaining no vocabulary term at all",
    }


__all__ = [
    "HoldoutCoverageAccumulator",
    "PERCENTILE_CONVENTION",
    "REPORTED_PERCENTILES",
    "VocabularyFamily",
    "coverage_definition",
    "percentile",
    "summarize",
]
