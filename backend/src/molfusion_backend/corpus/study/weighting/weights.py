"""The TF-IDF weighting primitives this study measured.

The implementation now lives in `molfusion_backend.tfidf.weighting`: Phase
5F-C.1 chose the contract, Phase 5F-D made it production code, and the
dependency points that way rather than leaving production importing its
own definitions out of a package named `study`.

Re-exported here so the study's own modules and tests keep reading the
names they were written against, and so there is exactly one
implementation of the arithmetic rather than two that could drift.
"""

from molfusion_backend.tfidf.weighting import (
    IDF_MODES,
    IDF_SMOOTHED,
    IDF_UNSMOOTHED,
    LOG_BASE,
    NORM_L2,
    NORM_NONE,
    NORMS,
    TF_MODES,
    TF_RAW,
    TF_SUBLINEAR,
    idf_formula,
    inverse_document_frequency,
    l2_normalize,
    natural_log,
    normalize,
    term_frequency,
    tf_formula,
    tfidf,
)
from typing import Any


def weighting_definition(
    tf_mode: str, idf_mode: str, norm: str, idf_dtype: str, output_dtype: str
) -> dict[str, Any]:
    """The complete numerical contract as report-ready data."""
    return {
        "tf_mode": tf_mode,
        "tf_formula": tf_formula(tf_mode),
        "idf_mode": idf_mode,
        "idf_formula": idf_formula(idf_mode),
        "smooth_idf": idf_mode == IDF_SMOOTHED,
        "use_idf": True,
        "log_base": LOG_BASE,
        "norm": norm,
        "normalization_formula": (
            "x / ||x||_2, with all-zero vectors left unchanged at zero"
            if norm == NORM_L2
            else "none"
        ),
        "order_of_operations": "tf(counts) -> multiply by idf -> normalize",
        "internal_arithmetic_dtype": "float64",
        "idf_storage_dtype": idf_dtype,
        "runtime_output_dtype": output_dtype,
        "zero_vector": (
            "a molecule that tokenizes but retains no vocabulary term yields an "
            "all-zero vector of the full dimension; this is a valid result, not "
            "a failure, and normalization leaves it zero rather than producing NaN"
        ),
        "oov": (
            "an n-gram outside the frozen vocabulary contributes nothing: no UNK "
            "dimension, no vocabulary growth, no refit, no exception"
        ),
    }


__all__ = [
    "IDF_MODES",
    "IDF_SMOOTHED",
    "IDF_UNSMOOTHED",
    "LOG_BASE",
    "NORMS",
    "NORM_L2",
    "NORM_NONE",
    "TF_MODES",
    "TF_RAW",
    "TF_SUBLINEAR",
    "idf_formula",
    "inverse_document_frequency",
    "l2_normalize",
    "natural_log",
    "normalize",
    "tf_formula",
    "term_frequency",
    "tfidf",
    "weighting_definition",
]
