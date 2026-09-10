"""The frozen SMILES token n-gram TF-IDF representation.

Phase boundaries, kept explicit because they answer different questions:

    5F-C    which token n-grams are features   (selection policy)
    5F-C.1  how a feature becomes a number     (weighting contract)
    5F-D    how both are materialized          (this artifact)

This package builds, loads, validates and applies the artifact. It does
not define a FeatureAgent, touch the registry, or expose an API surface.

    from molfusion_backend.tfidf import build_artifact, load_tfidf_artifact
"""

from molfusion_backend.tfidf.builder import (
    FROZEN_DOCUMENT_COUNT,
    FROZEN_FIT_CORPUS_SHA256,
    build_artifact,
    count_document_frequencies,
    deterministic_report_view,
    rebuild_and_compare,
    verify_corpus_identity,
)
from molfusion_backend.tfidf.contract import (
    ARTIFACT_ID,
    ARTIFACT_TYPE,
    ARTIFACT_VERSION,
    BUILD_REPORT_FILENAME,
    CONFIG_FILENAME,
    DIMENSION,
    IDF_FILENAME,
    INDEX_ORDER,
    MAX_FEATURES,
    MIN_DF,
    NGRAM_MAX,
    NGRAM_MIN,
    NGRAM_ORDERS,
    VOCABULARY_FILENAME,
    TfidfConfig,
    contract_mismatches,
    frozen_config,
)
from molfusion_backend.tfidf.errors import (
    TfidfArtifactError,
    TfidfArtifactExistsError,
    TfidfConfigError,
    TfidfCorpusIdentityError,
    TfidfIdfError,
    TfidfVocabularyError,
)
from molfusion_backend.tfidf.idf import (
    IDF_NPY_DESCR,
    IDF_NPY_VERSION,
    compute_idf,
    idf_bytes,
    inspect_idf_payload,
    load_idf,
    validate_idf,
    validate_idf_payload,
)
from molfusion_backend.tfidf.loader import TfidfArtifact, load_tfidf_artifact
from molfusion_backend.tfidf.ngrams import (
    Ngram,
    document_ngram_counts,
    document_ngram_counts_over_orders,
    iter_ngrams,
)
from molfusion_backend.tfidf.transform import TfidfTransformer, zero_vector
from molfusion_backend.tfidf.vocabulary import (
    Vocabulary,
    VocabularyEntry,
    composition_by_order,
    parse_vocabulary,
    select_vocabulary,
    validate_vocabulary,
    vocabulary_bytes,
)
from molfusion_backend.tfidf.weighting import (
    FROZEN_IDF_MODE,
    FROZEN_NORM,
    FROZEN_TF_MODE,
    IDF_DTYPE,
    RUNTIME_DTYPE,
    idf_formula,
    inverse_document_frequency,
    l2_normalize,
    term_frequency,
    tf_formula,
    tfidf,
)

__all__ = [
    "ARTIFACT_ID",
    "ARTIFACT_TYPE",
    "ARTIFACT_VERSION",
    "BUILD_REPORT_FILENAME",
    "CONFIG_FILENAME",
    "DIMENSION",
    "FROZEN_DOCUMENT_COUNT",
    "FROZEN_FIT_CORPUS_SHA256",
    "FROZEN_IDF_MODE",
    "FROZEN_NORM",
    "FROZEN_TF_MODE",
    "IDF_DTYPE",
    "IDF_NPY_DESCR",
    "IDF_NPY_VERSION",
    "IDF_FILENAME",
    "INDEX_ORDER",
    "MAX_FEATURES",
    "MIN_DF",
    "NGRAM_MAX",
    "NGRAM_MIN",
    "NGRAM_ORDERS",
    "RUNTIME_DTYPE",
    "Ngram",
    "TfidfArtifact",
    "TfidfArtifactError",
    "TfidfArtifactExistsError",
    "TfidfConfig",
    "TfidfConfigError",
    "TfidfCorpusIdentityError",
    "TfidfIdfError",
    "TfidfTransformer",
    "TfidfVocabularyError",
    "VOCABULARY_FILENAME",
    "Vocabulary",
    "VocabularyEntry",
    "build_artifact",
    "composition_by_order",
    "compute_idf",
    "contract_mismatches",
    "count_document_frequencies",
    "deterministic_report_view",
    "document_ngram_counts",
    "document_ngram_counts_over_orders",
    "frozen_config",
    "idf_bytes",
    "inspect_idf_payload",
    "idf_formula",
    "inverse_document_frequency",
    "iter_ngrams",
    "l2_normalize",
    "load_idf",
    "load_tfidf_artifact",
    "parse_vocabulary",
    "rebuild_and_compare",
    "select_vocabulary",
    "term_frequency",
    "tf_formula",
    "tfidf",
    "validate_idf",
    "validate_idf_payload",
    "validate_vocabulary",
    "verify_corpus_identity",
    "vocabulary_bytes",
    "zero_vector",
]
