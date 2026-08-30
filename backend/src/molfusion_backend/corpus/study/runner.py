"""The Phase 5F-C study pipeline: measure, tabulate, decide nothing.

    frozen corpus -> identity check -> tokenize -> n-gram DF/TF
    -> rarity and min_df tables -> candidate rankings and vocabularies
    -> holdout coverage / OOV / density -> study report

Nothing here fits, freezes, or writes a production representation. The
output is evidence: the tables a vocabulary policy should be chosen from,
plus enough provenance that the choice can be re-derived later from the
same corpus.

The corpus is treated as immutable input and is only ever opened for
reading, in binary, twice: once to count and once to score the holdout.
Its SHA-256 is checked before either pass, because a study run against a
different corpus is not a weaker result -- it is a result about something
else, silently mislabelled.
"""

import json
from array import array
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from molfusion_backend.artifacts.checksum import sha256_file
from molfusion_backend.corpus.errors import CorpusIdentityError
from molfusion_backend.corpus.serialization import CORPUS_ENCODING
from molfusion_backend.corpus.study.coverage import (
    HoldoutCoverageAccumulator,
    VocabularyFamily,
    summarize,
)
from molfusion_backend.corpus.study.ngrams import (
    NGRAM_ORDERS,
    Ngram,
    NgramEntry,
    NgramFrequencyAccumulator,
    band_labels,
    token_count_band,
)
from molfusion_backend.corpus.study.split import is_study_holdout
from molfusion_backend.corpus.study.vocabulary import (
    CANDIDATE_DIMENSIONS,
    CANDIDATE_POLICIES,
    MIN_DF_THRESHOLDS,
    RANKING_DOCUMENT_FREQUENCY,
    RANKING_TERM_FREQUENCY,
    RARITY_THRESHOLDS,
    SCOPE_CORPUS,
    SCOPE_FIT,
    frequency,
    min_df_prefix_size,
    rank_entries,
    rarity_histogram,
    select_orders,
    unigram_protected_ranking,
    unigram_retention,
)
from molfusion_backend.smiles_tokenizer import SMILES_TOKENIZER_ID, tokenize_smiles

STUDY_SCHEMA_VERSION = 1
STUDY_ID = "chembl37_token_ngram_vocabulary_study"

# The Phase 5F-B corpus this study is defined against. Hard-coded rather
# than read from the build report next to the corpus: a report sitting
# beside a file cannot vouch for that file, and the point of the check is
# to catch a corpus that is not the frozen one.
FROZEN_FIT_CORPUS_SHA256 = "b2c4b81160df05c95f8421582bb4b1c95fdf5964a4edaff24a7c1ddd43e2a5de"
FROZEN_DOCUMENT_COUNT = 2_897_639

STUDY_REPORT_FILENAME = "study_report.json"
DF_THRESHOLDS_FILENAME = "df_thresholds.csv"
VOCABULARY_COVERAGE_FILENAME = "vocabulary_coverage.csv"
HOLDOUT_COVERAGE_FILENAME = "holdout_coverage.csv"
TOP_NGRAMS_FILENAME = "top_ngrams.csv"
RANKING_COMPARISON_FILENAME = "ranking_comparison.csv"

# Diagnostic sample size for the human-readable top-n-gram dump. Bounded on
# purpose: the point is to eyeball what the ranking promotes, which a few
# hundred rows show and a few million rows hide.
TOP_NGRAM_DIAGNOSTIC_LIMIT = 250

# The band boundary used for the long-molecule sensitivity diagnostic:
# molecules of more than 256 tokens, i.e. roughly five times the corpus
# median. Expressed as a band index so it stays aligned with the bands the
# accumulator actually recorded.
LONG_MOLECULE_BAND_EDGE = 256


def verify_corpus_identity(corpus_path: Path, expected_sha256: str) -> str:
    """Hash the corpus and refuse to continue unless it is the frozen one."""
    if not corpus_path.is_file():
        raise CorpusIdentityError(f"Corpus not found: {corpus_path}")

    actual = sha256_file(corpus_path)
    if actual != expected_sha256:
        raise CorpusIdentityError(
            "Corpus identity mismatch -- refusing to run the study against a "
            f"different corpus.\n  expected sha256: {expected_sha256}\n"
            f"  actual   sha256: {actual}\n  path: {corpus_path}"
        )
    return actual


def iter_corpus_documents(corpus_path: Path) -> Iterator[str]:
    """Yield each canonical SMILES, strictly decoded, LF-delimited.

    `newline="\\n"` disables universal-newline translation: the corpus
    contract says LF, so a stray CR would be a corpus defect to surface,
    not whitespace to absorb.
    """
    with corpus_path.open("r", encoding=CORPUS_ENCODING, errors="strict", newline="\n") as handle:
        for line in handle:
            yield line[:-1] if line.endswith("\n") else line


# ---------------------------------------------------------------------------
# pass 1 -- exact n-gram frequencies
# ---------------------------------------------------------------------------


def count_ngrams(
    corpus_path: Path,
    *,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 250_000,
) -> tuple[NgramFrequencyAccumulator, dict[str, Any]]:
    """Stream the corpus once, accumulating DF/TF for orders 1, 2 and 3."""
    accumulator = NgramFrequencyAccumulator(NGRAM_ORDERS)
    token_counts_fit: list[int] = []
    token_counts_holdout: list[int] = []
    documents = 0

    for smiles in iter_corpus_documents(corpus_path):
        tokens = tokenize_smiles(smiles)
        holdout = is_study_holdout(smiles)
        accumulator.add_document(tokens, holdout=holdout)
        (token_counts_holdout if holdout else token_counts_fit).append(len(tokens))
        documents += 1
        if progress and progress_every and documents % progress_every == 0:
            progress(documents)

    token_summary = {
        "fit": summarize_int_sample(token_counts_fit),
        "holdout": summarize_int_sample(token_counts_holdout),
    }
    return accumulator, {"documents": documents, "token_count": token_summary}


def summarize_int_sample(values: Sequence[int]) -> dict[str, Any]:
    """min/mean/percentiles over an integer sample, via the shared summary."""
    return summarize(array("Q", values))


# ---------------------------------------------------------------------------
# descriptive tables
# ---------------------------------------------------------------------------


def order_tables(accumulator: NgramFrequencyAccumulator) -> dict[str, Any]:
    """Raw vocabulary size, occurrence totals, and rarity per n-gram order."""
    tables = {}
    for order in accumulator.orders:
        entries = accumulator.entries(order)
        holdout_present = sum(1 for entry in entries if entry.document_frequency_holdout > 0)
        tables[str(order)] = {
            "distinct_ngrams_corpus": len(entries),
            "distinct_ngrams_fit": sum(
                1 for entry in entries if entry.document_frequency_fit > 0
            ),
            "distinct_ngrams_holdout": holdout_present,
            # The vocabulary-level OOV floor: motifs the holdout contains
            # that the fit subset never saw. No pruning policy can recover
            # these, so they bound what any candidate vocabulary can cover.
            "distinct_ngrams_holdout_unseen_in_fit": sum(
                1
                for entry in entries
                if entry.document_frequency_holdout > 0 and entry.document_frequency_fit == 0
            ),
            "occurrences_holdout_unseen_in_fit": sum(
                entry.term_frequency_holdout
                for entry in entries
                if entry.document_frequency_fit == 0
            ),
            "total_occurrences_corpus": sum(entry.term_frequency for entry in entries),
            "total_occurrences_fit": sum(entry.term_frequency_fit for entry in entries),
            "total_occurrences_holdout": sum(entry.term_frequency_holdout for entry in entries),
            "rarity_corpus": _rarity_block(entries, len(entries), SCOPE_CORPUS),
            "rarity_fit": _rarity_block(
                [entry for entry in entries if entry.document_frequency_fit > 0],
                sum(1 for entry in entries if entry.document_frequency_fit > 0),
                SCOPE_FIT,
            ),
            "document_frequency": summarize_int_sample(
                [entry.document_frequency for entry in entries]
            ),
            "term_frequency": summarize_int_sample([entry.term_frequency for entry in entries]),
        }
    return tables


def _rarity_block(entries: Sequence[NgramEntry], total: int, scope: str) -> dict[str, Any]:
    histogram = rarity_histogram(entries, RARITY_THRESHOLDS, scope)
    return {
        "vocabulary": total,
        "counts": {f"df_le_{threshold}": count for threshold, count in histogram.items()},
        "fractions": {
            f"df_le_{threshold}": (count / total if total else None)
            for threshold, count in histogram.items()
        },
    }


def min_df_table(
    entries_by_order: dict[int, list[NgramEntry]], scope: str
) -> list[dict[str, Any]]:
    """Vocabulary size at each absolute document-frequency threshold."""
    rows = []
    for threshold in MIN_DF_THRESHOLDS:
        sizes = {
            order: sum(
                1
                for entry in entries
                if frequency(entry, RANKING_DOCUMENT_FREQUENCY, scope) >= threshold
            )
            for order, entries in entries_by_order.items()
        }
        rows.append(
            {
                "scope": scope,
                "min_df": threshold,
                "unigrams": sizes.get(1, 0),
                "bigrams": sizes.get(2, 0),
                "trigrams": sizes.get(3, 0),
                "combined_1_1": sizes.get(1, 0),
                "combined_1_2": sizes.get(1, 0) + sizes.get(2, 0),
                "combined_1_3": sizes.get(1, 0) + sizes.get(2, 0) + sizes.get(3, 0),
                "combined_2_3": sizes.get(2, 0) + sizes.get(3, 0),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# candidate vocabularies
# ---------------------------------------------------------------------------


def _candidate_sizes(vocabulary_size: int, extra: Sequence[int] = ()) -> tuple[int, ...]:
    """Ascending, deduplicated, clamped candidate dimensions.

    The full vocabulary is always included so every capped candidate can be
    read against its own uncapped baseline rather than against a different
    policy's.
    """
    sizes = {min(dimension, vocabulary_size) for dimension in CANDIDATE_DIMENSIONS}
    sizes.update(min(value, vocabulary_size) for value in extra)
    sizes.add(vocabulary_size)
    return tuple(sorted(size for size in sizes if size > 0))


def build_families(
    entries_by_order: dict[int, list[NgramEntry]],
) -> tuple[list[VocabularyFamily], dict[str, Any]]:
    """One ranking family per (policy, ranking metric), plus the
    unigram-protected variants the measurements show are needed."""
    families: list[VocabularyFamily] = []
    policy_report: dict[str, Any] = {}

    for policy in CANDIDATE_POLICIES:
        entries = select_orders(
            [entry for order in sorted(entries_by_order) for entry in entries_by_order[order]],
            policy.orders,
        )
        # Study vocabularies are fitted on the 95% subset only, so an
        # n-gram that never occurs there is not part of the ranking at all.
        eligible = [entry for entry in entries if entry.document_frequency_fit > 0]

        ranked_df = rank_entries(eligible, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)
        ranked_tf = rank_entries(eligible, RANKING_TERM_FREQUENCY, SCOPE_FIT)
        vocabulary_size = len(eligible)

        min_df_sizes = {
            threshold: min_df_prefix_size(ranked_df, threshold, SCOPE_FIT)
            for threshold in MIN_DF_THRESHOLDS
        }
        df_sizes = _candidate_sizes(vocabulary_size, tuple(min_df_sizes.values()))
        tf_sizes = _candidate_sizes(vocabulary_size)

        families.append(
            VocabularyFamily(
                name=f"{policy.name}-df",
                policy=policy.label,
                orders=policy.orders,
                ranking=RANKING_DOCUMENT_FREQUENCY,
                protected_unigrams=False,
                sizes=df_sizes,
                ranked=tuple(entry.ngram for entry in ranked_df),
            )
        )
        families.append(
            VocabularyFamily(
                name=f"{policy.name}-tf",
                policy=policy.label,
                orders=policy.orders,
                ranking=RANKING_TERM_FREQUENCY,
                protected_unigrams=False,
                sizes=tf_sizes,
                ranked=tuple(entry.ngram for entry in ranked_tf),
            )
        )

        retention = {
            str(size): unigram_retention(ranked_df, size) for size in df_sizes
        }
        # The alternative policy of section 12 is only worth scoring where
        # a global cap would actually drop a unigram, and only where the
        # policy mixes orders -- promoting unigrams within a unigram-only
        # ranking is the identity, and would just duplicate rows.
        mixes_orders = 1 in policy.orders and len(policy.orders) > 1
        lossy_sizes = tuple(
            size for size in df_sizes if retention[str(size)]["unigrams_excluded"] > 0
        )
        if lossy_sizes and mixes_orders:
            families.append(
                VocabularyFamily(
                    name=f"{policy.name}-df-protected",
                    policy=policy.label,
                    orders=policy.orders,
                    ranking=RANKING_DOCUMENT_FREQUENCY,
                    protected_unigrams=True,
                    sizes=lossy_sizes,
                    ranked=tuple(
                        entry.ngram for entry in unigram_protected_ranking(ranked_df)
                    ),
                )
            )

        policy_report[policy.name] = {
            "label": policy.label,
            "orders": list(policy.orders),
            "vocabulary_size_fit": vocabulary_size,
            "vocabulary_size_corpus": len(entries),
            "min_df_vocabulary_size_fit": {
                str(threshold): size for threshold, size in min_df_sizes.items()
            },
            "candidate_dimensions_df": list(df_sizes),
            "candidate_dimensions_tf": list(tf_sizes),
            # The DF of the last term a cut of this width keeps. Because
            # descending-DF ranking is non-increasing, this is the
            # effective min_df that dimension imposes -- which is what
            # makes "cap at K" and "prune at min_df" the same knob here.
            "effective_min_df_at_dimension": {
                str(size): frequency(ranked_df[size - 1], RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)
                for size in df_sizes
            },
            "unigram_retention_df": retention,
            "unigram_protection_scored_at": list(lossy_sizes) if mixes_orders else [],
        }

    return families, policy_report


# ---------------------------------------------------------------------------
# pass 2 -- holdout scoring
# ---------------------------------------------------------------------------


def score_holdout(
    corpus_path: Path,
    families: Sequence[VocabularyFamily],
    *,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 250_000,
) -> list[dict[str, Any]]:
    """Second read-only pass, holdout molecules only."""
    accumulator = HoldoutCoverageAccumulator(families)
    seen = 0
    for smiles in iter_corpus_documents(corpus_path):
        if not is_study_holdout(smiles):
            continue
        accumulator.add_document(tokenize_smiles(smiles))
        seen += 1
        if progress and progress_every and seen % progress_every == 0:
            progress(seen)
    return accumulator.results()


def unique_ngram_coverage(
    families: Sequence[VocabularyFamily], entries_by_order: dict[int, list[NgramEntry]]
) -> dict[tuple[str, int], dict[str, Any]]:
    """Fraction of the holdout's *distinct* n-grams each candidate covers.

    A corpus-level quantity, not an average of per-molecule ones, so it is
    computed from the pass-1 holdout counts rather than re-derived per
    molecule. Cumulative over the ranking, which makes every candidate
    dimension of a family answerable in one sweep.
    """
    holdout_present = {
        order: {entry.ngram for entry in entries if entry.document_frequency_holdout > 0}
        for order, entries in entries_by_order.items()
    }
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for family in families:
        present = set()
        for order in family.orders:
            present |= holdout_present.get(order, set())
        total = len(present)
        covered = 0
        size_index = 0
        for rank, ngram in enumerate(family.ranked):
            while size_index < len(family.sizes) and rank == family.sizes[size_index]:
                out[(family.name, family.sizes[size_index])] = {
                    "holdout_distinct_ngrams": total,
                    "holdout_distinct_covered": covered,
                    "holdout_unique_coverage": covered / total if total else None,
                }
                size_index += 1
            if size_index >= len(family.sizes):
                break
            if ngram in present:
                covered += 1
        while size_index < len(family.sizes):
            out[(family.name, family.sizes[size_index])] = {
                "holdout_distinct_ngrams": total,
                "holdout_distinct_covered": covered,
                "holdout_unique_coverage": covered / total if total else None,
            }
            size_index += 1
    return out


# ---------------------------------------------------------------------------
# ranking diagnostics
# ---------------------------------------------------------------------------


def ranking_comparison(
    entries_by_order: dict[int, list[NgramEntry]],
    dimensions: Sequence[int] = (2048, 4096, 8192),
) -> list[dict[str, Any]]:
    """Top-K agreement between the DF and TF rankings, per policy."""
    rows = []
    all_entries = [entry for order in sorted(entries_by_order) for entry in entries_by_order[order]]
    for policy in CANDIDATE_POLICIES:
        eligible = [
            entry
            for entry in select_orders(all_entries, policy.orders)
            if entry.document_frequency_fit > 0
        ]
        ranked_df = [entry.ngram for entry in rank_entries(eligible, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)]
        ranked_tf = [entry.ngram for entry in rank_entries(eligible, RANKING_TERM_FREQUENCY, SCOPE_FIT)]
        for dimension in dimensions:
            size = min(dimension, len(eligible))
            if size <= 0:
                continue
            top_df = set(ranked_df[:size])
            top_tf = set(ranked_tf[:size])
            shared = len(top_df & top_tf)
            rows.append(
                {
                    "policy": policy.label,
                    "dimension": size,
                    "requested_dimension": dimension,
                    "vocabulary_size_fit": len(eligible),
                    "shared_terms": shared,
                    "jaccard": shared / len(top_df | top_tf) if (top_df or top_tf) else None,
                    "overlap_fraction": shared / size,
                    "df_only_terms": size - shared,
                    "tf_only_terms": size - shared,
                }
            )
    return rows


def long_molecule_sensitivity(
    accumulator: NgramFrequencyAccumulator,
    entries_by_order: dict[int, list[NgramEntry]],
    dimensions: Sequence[int] = (2048, 4096, 8192),
) -> dict[str, Any]:
    """Does dropping unusually long molecules move the TF ranking more than
    the DF ranking?

    The mechanism under test: TF counts every repetition, so one 1617-token
    molecule can contribute as much weight to a motif as hundreds of
    ordinary molecules, while DF caps every molecule's contribution at one.
    If TF ranking is the more fragile of the two, that is an argument
    against it that does not depend on taste.
    """
    band = token_count_band(LONG_MOLECULE_BAND_EDGE)
    fit_documents = accumulator.documents_by_band(holdout=False)
    long_documents = sum(fit_documents[band + 1 :])

    per_policy = []
    all_entries = [entry for order in sorted(entries_by_order) for entry in entries_by_order[order]]
    for policy in CANDIDATE_POLICIES:
        eligible = [
            entry
            for entry in select_orders(all_entries, policy.orders)
            if entry.document_frequency_fit > 0
        ]
        full_tf = [entry.ngram for entry in rank_entries(eligible, RANKING_TERM_FREQUENCY, SCOPE_FIT)]
        full_df = [entry.ngram for entry in rank_entries(eligible, RANKING_DOCUMENT_FREQUENCY, SCOPE_FIT)]
        short_tf = [
            entry.ngram
            for entry in sorted(
                eligible,
                key=lambda entry: (-entry.term_frequency_fit_up_to_band(band), entry.ngram),
            )
        ]
        short_df = [
            entry.ngram
            for entry in sorted(
                eligible,
                key=lambda entry: (-entry.document_frequency_fit_up_to_band(band), entry.ngram),
            )
        ]
        for dimension in dimensions:
            size = min(dimension, len(eligible))
            if size <= 0:
                continue
            tf_churn = size - len(set(full_tf[:size]) & set(short_tf[:size]))
            df_churn = size - len(set(full_df[:size]) & set(short_df[:size]))
            per_policy.append(
                {
                    "policy": policy.label,
                    "dimension": size,
                    "tf_terms_changed": tf_churn,
                    "tf_churn_fraction": tf_churn / size,
                    "df_terms_changed": df_churn,
                    "df_churn_fraction": df_churn / size,
                }
            )

    occurrences_total = sum(
        sum(entry.term_frequency_fit for entry in entries) for entries in entries_by_order.values()
    )
    occurrences_short = sum(
        sum(entry.term_frequency_fit_up_to_band(band) for entry in entries)
        for entries in entries_by_order.values()
    )
    return {
        "long_molecule_definition": f"more than {LONG_MOLECULE_BAND_EDGE} tokens",
        "band_index": band,
        "band_labels": band_labels(),
        "fit_documents_by_band": list(fit_documents),
        "fit_documents_long": long_documents,
        "fit_documents_long_fraction": (
            long_documents / accumulator.fit_document_count
            if accumulator.fit_document_count
            else None
        ),
        "fit_occurrence_share_long": (
            (occurrences_total - occurrences_short) / occurrences_total
            if occurrences_total
            else None
        ),
        "rank_churn_when_long_molecules_dropped": per_policy,
    }


def top_ngram_rows(
    entries_by_order: dict[int, list[NgramEntry]], limit: int = TOP_NGRAM_DIAGNOSTIC_LIMIT
) -> list[dict[str, Any]]:
    """A bounded, human-readable sample of what each ranking promotes."""
    rows = []
    for order in sorted(entries_by_order):
        entries = entries_by_order[order]
        df_rank = {
            entry.ngram: index
            for index, entry in enumerate(rank_entries(entries, RANKING_DOCUMENT_FREQUENCY, SCOPE_CORPUS))
        }
        tf_rank = {
            entry.ngram: index
            for index, entry in enumerate(rank_entries(entries, RANKING_TERM_FREQUENCY, SCOPE_CORPUS))
        }
        for entry in rank_entries(entries, RANKING_DOCUMENT_FREQUENCY, SCOPE_CORPUS)[:limit]:
            rows.append(
                {
                    "order": order,
                    "ngram": json.dumps(list(entry.ngram), ensure_ascii=False),
                    "df_rank": df_rank[entry.ngram],
                    "tf_rank": tf_rank[entry.ngram],
                    "document_frequency": entry.document_frequency,
                    "term_frequency": entry.term_frequency,
                    "document_frequency_fit": entry.document_frequency_fit,
                    "term_frequency_fit": entry.term_frequency_fit,
                    "document_frequency_holdout": entry.document_frequency_holdout,
                    "term_frequency_holdout": entry.term_frequency_holdout,
                    "occurrences_per_document": (
                        entry.term_frequency / entry.document_frequency
                        if entry.document_frequency
                        else None
                    ),
                }
            )
    return rows


__all__ = [
    "FROZEN_DOCUMENT_COUNT",
    "FROZEN_FIT_CORPUS_SHA256",
    "STUDY_ID",
    "STUDY_SCHEMA_VERSION",
    "build_families",
    "count_ngrams",
    "iter_corpus_documents",
    "long_molecule_sensitivity",
    "min_df_table",
    "order_tables",
    "ranking_comparison",
    "score_holdout",
    "top_ngram_rows",
    "unique_ngram_coverage",
    "verify_corpus_identity",
]
