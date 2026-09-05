"""The frozen SMILES TF-IDF representation contract.

`tfidf_config.json` is the artifact's statement of what its numbers mean.
It is written so that someone holding only the corpus, the contract IDs
and this file can reconstruct every value in `idf.npy` and every vector
the representation will ever emit -- without reading MolFusion's source,
and without a library flag standing in for a formula.

Phase separation, kept explicit because the three answer different
questions:

    5F-C    which token n-grams are features
    5F-C.1  how a retained feature becomes a number
    5F-D    how both are materialized as an immutable artifact
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from molfusion_backend.chemistry import CANONICAL_SMILES_NORMALIZATION_ID
from molfusion_backend.corpus.serialization import CORPUS_SERIALIZATION_ID
from molfusion_backend.smiles_tokenizer import SMILES_TOKENIZER_ID
from molfusion_backend.tfidf import weighting

CONFIG_SCHEMA_VERSION = 1

ARTIFACT_TYPE = "smiles_tfidf"
ARTIFACT_ID = "chembl37_token_ngrams_1_3"
ARTIFACT_VERSION = "1.0.0"

# The Phase 5F-C selection policy.
NGRAM_MIN = 1
NGRAM_MAX = 3
NGRAM_ORDERS = (1, 2, 3)
MIN_DF = 5
MAX_FEATURES = 4096
DIMENSION = MAX_FEATURES

SELECTION_RANKING = "descending full-corpus document frequency"
SELECTION_TIE_BREAK = "ascending lexicographic n-gram token tuple"
SELECTION_KEY = "(-document_frequency, ngram_tuple)"

# Index ordering is deliberately *not* the selection ranking. Selection
# decides membership; indexing decides which column a member occupies. A
# lexicographic index is stable under re-fitting: two terms swapping
# document frequency by a single document would swap columns under ranking
# order, whereas here a re-fit that keeps the same term set keeps the same
# layout.
INDEX_ORDER = "lexicographic_token_tuple_after_selection"

OOV_POLICY = (
    "an n-gram outside the frozen vocabulary contributes nothing: no UNK "
    "dimension, no vocabulary growth, no refit, and no exception. Vocabulary "
    "OOV is normal runtime behaviour, distinct from tokenization failure, "
    "which remains an error under the Phase 5F-A contract."
)
ZERO_VECTOR_POLICY = (
    "a valid molecule that tokenizes successfully but retains no vocabulary "
    "n-gram yields np.zeros(dimension, dtype=float32). This is a valid "
    "result, not a representation failure. L2 normalization leaves it "
    "exactly zero and never produces NaN or Inf."
)

# Payload filenames, fixed so the loader and the builder cannot disagree.
VOCABULARY_FILENAME = "vocabulary.json"
IDF_FILENAME = "idf.npy"
CONFIG_FILENAME = "tfidf_config.json"
BUILD_REPORT_FILENAME = "build_report.json"


class TfidfConfig(BaseModel):
    """Validated contents of `tfidf_config.json`.

    Every field is a semantic commitment, so loading is strict: a config
    that disagrees with the frozen contract is rejected rather than
    tolerated, because tolerating it would mean emitting vectors whose
    meaning differs from the metadata describing them.
    """

    schema_version: int
    normalization_id: str
    tokenizer_id: str
    serialization_id: str

    ngram_min: int = Field(ge=1)
    ngram_max: int = Field(ge=1)
    min_df: int = Field(ge=1)
    max_features: int = Field(ge=1)
    dimension: int = Field(ge=1)

    selection_ranking: str
    selection_tie_break: str
    selection_key: str
    index_order: str
    cap_is_binding: bool
    eligible_terms_at_min_df: int = Field(ge=0)

    tf_mode: str
    tf_formula: str
    use_idf: bool
    idf_mode: str
    idf_formula: str
    smooth_idf: bool
    log_base: str
    norm: str
    order_of_operations: str

    internal_arithmetic_dtype: str
    idf_dtype: str
    runtime_output_dtype: str

    fit_document_count: int = Field(ge=1)
    oov_policy: str
    zero_vector_policy: str

    @field_validator("tf_mode")
    @classmethod
    def _known_tf_mode(cls, value: str) -> str:
        if value not in weighting.TF_MODES:
            raise ValueError(f"unknown tf_mode: {value!r}")
        return value

    @field_validator("idf_mode")
    @classmethod
    def _known_idf_mode(cls, value: str) -> str:
        if value not in weighting.IDF_MODES:
            raise ValueError(f"unknown idf_mode: {value!r}")
        return value

    @field_validator("norm")
    @classmethod
    def _known_norm(cls, value: str) -> str:
        if value not in weighting.NORMS:
            raise ValueError(f"unknown norm: {value!r}")
        return value


def frozen_config(
    *,
    fit_document_count: int,
    eligible_terms_at_min_df: int,
    dimension: int = DIMENSION,
    min_df: int = MIN_DF,
    max_features: int = MAX_FEATURES,
) -> dict[str, Any]:
    """The canonical config payload for this representation.

    `min_df` and `max_features` are both recorded, and `cap_is_binding`
    states which one actually pruned. On ChEMBL 37 `min_df = 5` leaves far
    more than `max_features` eligible terms, so quoting the rarity floor
    alone would misstate the pruning by more than a factor of two.

    The two selection parameters are arguments rather than baked-in
    constants so the config always states what the build that wrote it
    actually did. A fixture build with different parameters must produce a
    config describing *those* parameters -- a payload that reports the
    frozen defaults regardless of what ran would be worse than no payload.
    """
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "normalization_id": CANONICAL_SMILES_NORMALIZATION_ID,
        "tokenizer_id": SMILES_TOKENIZER_ID,
        "serialization_id": CORPUS_SERIALIZATION_ID,
        "ngram_min": NGRAM_MIN,
        "ngram_max": NGRAM_MAX,
        "min_df": min_df,
        "max_features": max_features,
        "dimension": dimension,
        "selection_ranking": SELECTION_RANKING,
        "selection_tie_break": SELECTION_TIE_BREAK,
        "selection_key": SELECTION_KEY,
        "index_order": INDEX_ORDER,
        "cap_is_binding": eligible_terms_at_min_df > dimension,
        "eligible_terms_at_min_df": eligible_terms_at_min_df,
        "tf_mode": weighting.FROZEN_TF_MODE,
        "tf_formula": weighting.tf_formula(weighting.FROZEN_TF_MODE),
        "use_idf": True,
        "idf_mode": weighting.FROZEN_IDF_MODE,
        "idf_formula": weighting.idf_formula(weighting.FROZEN_IDF_MODE),
        "smooth_idf": weighting.FROZEN_IDF_MODE == weighting.IDF_SMOOTHED,
        "log_base": weighting.LOG_BASE,
        "norm": weighting.FROZEN_NORM,
        "order_of_operations": "tf(counts) -> multiply by idf -> normalize",
        "internal_arithmetic_dtype": "float64",
        "idf_dtype": weighting.FROZEN_IDF_DTYPE,
        "runtime_output_dtype": weighting.FROZEN_RUNTIME_DTYPE,
        "fit_document_count": fit_document_count,
        "oov_policy": OOV_POLICY,
        "zero_vector_policy": ZERO_VECTOR_POLICY,
    }


def contract_mismatches(
    config: TfidfConfig, *, min_df: int = MIN_DF, max_features: int = MAX_FEATURES
) -> list[str]:
    """Every way `config` departs from the frozen contract, named.

    Returned as a list rather than raised so a caller can report all the
    problems at once; a loader that surfaces one mismatch per run makes
    diagnosing a stale artifact needlessly slow.

    `min_df` and `max_features` are overridable so a fixture artifact can
    be checked against the parameters it was built with. Everything else --
    the contract identifiers, formulas, dtypes and policies -- is frozen
    and not negotiable.
    """
    expected = (
        ("schema_version", config.schema_version, CONFIG_SCHEMA_VERSION),
        ("normalization_id", config.normalization_id, CANONICAL_SMILES_NORMALIZATION_ID),
        ("tokenizer_id", config.tokenizer_id, SMILES_TOKENIZER_ID),
        ("serialization_id", config.serialization_id, CORPUS_SERIALIZATION_ID),
        ("ngram_min", config.ngram_min, NGRAM_MIN),
        ("ngram_max", config.ngram_max, NGRAM_MAX),
        ("min_df", config.min_df, min_df),
        ("max_features", config.max_features, max_features),
        ("index_order", config.index_order, INDEX_ORDER),
        ("tf_mode", config.tf_mode, weighting.FROZEN_TF_MODE),
        ("idf_mode", config.idf_mode, weighting.FROZEN_IDF_MODE),
        ("use_idf", config.use_idf, True),
        ("smooth_idf", config.smooth_idf, True),
        ("log_base", config.log_base, weighting.LOG_BASE),
        ("norm", config.norm, weighting.FROZEN_NORM),
        ("idf_dtype", config.idf_dtype, weighting.FROZEN_IDF_DTYPE),
        ("runtime_output_dtype", config.runtime_output_dtype, weighting.FROZEN_RUNTIME_DTYPE),
        ("idf_formula", config.idf_formula, weighting.idf_formula(weighting.FROZEN_IDF_MODE)),
        ("tf_formula", config.tf_formula, weighting.tf_formula(weighting.FROZEN_TF_MODE)),
    )
    return [
        f"{name}: artifact has {actual!r}, contract requires {wanted!r}"
        for name, actual, wanted in expected
        if actual != wanted
    ]


__all__ = [
    "ARTIFACT_ID",
    "ARTIFACT_TYPE",
    "ARTIFACT_VERSION",
    "BUILD_REPORT_FILENAME",
    "CONFIG_FILENAME",
    "CONFIG_SCHEMA_VERSION",
    "DIMENSION",
    "IDF_FILENAME",
    "INDEX_ORDER",
    "MAX_FEATURES",
    "MIN_DF",
    "NGRAM_MAX",
    "NGRAM_MIN",
    "NGRAM_ORDERS",
    "OOV_POLICY",
    "SELECTION_KEY",
    "SELECTION_RANKING",
    "SELECTION_TIE_BREAK",
    "TfidfConfig",
    "VOCABULARY_FILENAME",
    "ZERO_VECTOR_POLICY",
    "contract_mismatches",
    "frozen_config",
]
