"""Phase 5F-C.1: the TF-IDF numerical weighting contract.

Phase 5F-C froze *which* token n-grams become features. This package
freezes *how* a retained feature becomes a number -- the term-frequency
rule, the exact inverse-document-frequency formula, the normalization, the
precision policy, the vector index ordering, the lossless vocabulary
encoding, and the zero-vector and out-of-vocabulary semantics.

It is still a study package: it measures and recommends, and writes no
production vocabulary, IDF payload, artifact metadata or feature agent.
Phase 5F-D packages what this phase settles.

    from molfusion_backend.corpus.study.weighting import tfidf, run_weighting_study
"""

from molfusion_backend.corpus.study.weighting.diagnostics import (
    SparseVector,
    WeightingDiagnostics,
    idf_comparison,
    pearson,
    spearman,
    vectorize,
)
from molfusion_backend.corpus.study.weighting.payload import (
    FROZEN_DIMENSION,
    FROZEN_MIN_DF,
    FROZEN_NGRAM_ORDERS,
    INDEX_ORDER_LEXICOGRAPHIC,
    INDEX_ORDER_RANKING,
    VOCABULARY_PAYLOAD_SCHEMA,
    VocabularyTerm,
    assign_indices,
    decode_tokens,
    document_frequencies,
    encode_tokens,
    index_ordering_definition,
    payload_schema,
    select_terms,
    selection_definition,
    term_index,
)
from molfusion_backend.corpus.study.weighting.report import (
    REPORT_FILENAME,
    STUDY_ID,
    STUDY_SCHEMA_VERSION,
    corpus_pass,
    deterministic_study_view,
    run_weighting_study,
    study_report_bytes,
)
from molfusion_backend.corpus.study.weighting.sampling import (
    SAMPLE_ID,
    STRATUM_NAMES,
    is_sampled,
    sample_bucket,
    sampling_definition,
    stratum_for,
)
from molfusion_backend.corpus.study.weighting.weights import (
    IDF_MODES,
    IDF_SMOOTHED,
    IDF_UNSMOOTHED,
    NORM_L2,
    NORM_NONE,
    NORMS,
    TF_MODES,
    TF_RAW,
    TF_SUBLINEAR,
    idf_formula,
    inverse_document_frequency,
    l2_normalize,
    normalize,
    term_frequency,
    tf_formula,
    tfidf,
    weighting_definition,
)

__all__ = [
    "FROZEN_DIMENSION",
    "FROZEN_MIN_DF",
    "FROZEN_NGRAM_ORDERS",
    "IDF_MODES",
    "IDF_SMOOTHED",
    "IDF_UNSMOOTHED",
    "INDEX_ORDER_LEXICOGRAPHIC",
    "INDEX_ORDER_RANKING",
    "NORMS",
    "NORM_L2",
    "NORM_NONE",
    "REPORT_FILENAME",
    "SAMPLE_ID",
    "STRATUM_NAMES",
    "STUDY_ID",
    "STUDY_SCHEMA_VERSION",
    "SparseVector",
    "TF_MODES",
    "TF_RAW",
    "TF_SUBLINEAR",
    "VOCABULARY_PAYLOAD_SCHEMA",
    "VocabularyTerm",
    "WeightingDiagnostics",
    "assign_indices",
    "corpus_pass",
    "decode_tokens",
    "deterministic_study_view",
    "document_frequencies",
    "encode_tokens",
    "idf_comparison",
    "idf_formula",
    "index_ordering_definition",
    "inverse_document_frequency",
    "is_sampled",
    "l2_normalize",
    "normalize",
    "payload_schema",
    "pearson",
    "run_weighting_study",
    "sample_bucket",
    "sampling_definition",
    "select_terms",
    "selection_definition",
    "spearman",
    "stratum_for",
    "study_report_bytes",
    "term_frequency",
    "term_index",
    "tf_formula",
    "tfidf",
    "vectorize",
    "weighting_definition",
]
