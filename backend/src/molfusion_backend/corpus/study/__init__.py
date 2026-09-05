"""Phase 5F-C: the corpus-driven token n-gram and vocabulary study.

A measurement package, not a fitting package. It answers "what should the
SMILES TF-IDF vocabulary be?" with tables computed from the frozen ChEMBL
37 reference corpus, and deliberately stops short of building anything
production consumes -- no vectorizer, no IDF, no artifact, no agent.

    from molfusion_backend.corpus.study import run_study
"""

from molfusion_backend.corpus.study.coverage import (
    PERCENTILE_CONVENTION,
    HoldoutCoverageAccumulator,
    VocabularyFamily,
    percentile,
    summarize,
)
from molfusion_backend.corpus.study.ngrams import (
    NGRAM_ORDERS,
    TOKEN_COUNT_BAND_EDGES,
    NgramEntry,
    NgramFrequencyAccumulator,
    document_ngram_counts,
    iter_ngrams,
    token_count_band,
)
from molfusion_backend.corpus.study.report import (
    deterministic_study_view,
    peak_memory_bytes,
    run_study,
    study_report_bytes,
)
from molfusion_backend.corpus.study.runner import (
    FROZEN_DOCUMENT_COUNT,
    FROZEN_FIT_CORPUS_SHA256,
    STUDY_ID,
    STUDY_SCHEMA_VERSION,
    build_families,
    count_ngrams,
    verify_corpus_identity,
)
from molfusion_backend.corpus.study.split import (
    STUDY_HOLDOUT_BUCKET,
    STUDY_SPLIT_BUCKET_COUNT,
    STUDY_SPLIT_ID,
    is_study_holdout,
    split_definition,
    study_bucket,
)
from molfusion_backend.corpus.study.vocabulary import (
    CANDIDATE_DIMENSIONS,
    CANDIDATE_POLICIES,
    MIN_DF_THRESHOLDS,
    RANKING_DOCUMENT_FREQUENCY,
    RANKING_TERM_FREQUENCY,
    RARITY_THRESHOLDS,
    SCOPE_CORPUS,
    SCOPE_FIT,
    NgramPolicy,
    apply_min_df,
    min_df_prefix_size,
    rank_entries,
    ranking_sort_key,
    rarity_histogram,
    unigram_protected_ranking,
    unigram_retention,
)

__all__ = [
    "CANDIDATE_DIMENSIONS",
    "CANDIDATE_POLICIES",
    "FROZEN_DOCUMENT_COUNT",
    "FROZEN_FIT_CORPUS_SHA256",
    "HoldoutCoverageAccumulator",
    "MIN_DF_THRESHOLDS",
    "NGRAM_ORDERS",
    "NgramEntry",
    "NgramFrequencyAccumulator",
    "NgramPolicy",
    "PERCENTILE_CONVENTION",
    "RANKING_DOCUMENT_FREQUENCY",
    "RANKING_TERM_FREQUENCY",
    "RARITY_THRESHOLDS",
    "SCOPE_CORPUS",
    "SCOPE_FIT",
    "STUDY_HOLDOUT_BUCKET",
    "STUDY_ID",
    "STUDY_SCHEMA_VERSION",
    "STUDY_SPLIT_BUCKET_COUNT",
    "STUDY_SPLIT_ID",
    "TOKEN_COUNT_BAND_EDGES",
    "VocabularyFamily",
    "apply_min_df",
    "build_families",
    "count_ngrams",
    "deterministic_study_view",
    "document_ngram_counts",
    "is_study_holdout",
    "iter_ngrams",
    "min_df_prefix_size",
    "peak_memory_bytes",
    "percentile",
    "rank_entries",
    "ranking_sort_key",
    "rarity_histogram",
    "run_study",
    "split_definition",
    "study_bucket",
    "study_report_bytes",
    "summarize",
    "token_count_band",
    "unigram_protected_ranking",
    "unigram_retention",
    "verify_corpus_identity",
]
