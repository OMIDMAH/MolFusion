"""The numerical comparisons that decide the weighting contract.

Four questions, each answered by measurement rather than convention:

    raw vs sublinear TF   does a repeated motif dominate a molecule's own
                          vector, and does that get worse with length?
    smoothed vs not       is the difference between the two IDF formulas
                          numerically real at this vocabulary's df range?
    norm=None vs L2       does an unnormalized vector's magnitude encode
                          molecule size?
    float64 vs float32    what does storing or emitting the smaller type
                          actually cost?

Everything operates on the sparse support of a molecule -- roughly 80 of
4,096 columns -- because the zero columns contribute nothing to any of
these statistics and materializing them would turn a cheap study into a
600 MB one.
"""

from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from molfusion_backend.corpus.study.coverage import summarize
from molfusion_backend.corpus.study.ngrams import Ngram, document_ngram_counts
from molfusion_backend.corpus.study.weighting import weights
from molfusion_backend.corpus.study.weighting.sampling import STRATUM_NAMES

# Document-frequency bands used to locate where smoothed and unsmoothed
# IDF diverge. Upper edge inclusive; open at the top.
IDF_DF_BANDS = (10, 100, 1_000, 10_000, 100_000, 1_000_000)


@dataclass(frozen=True)
class SparseVector:
    """One molecule's counts over the vocabulary, on its support only."""

    indices: np.ndarray
    counts: np.ndarray
    token_count: int
    smiles_length: int

    @property
    def nonzero(self) -> int:
        return int(self.indices.size)


def vectorize(
    tokens: Sequence[str],
    term_index: dict[Ngram, int],
    orders: Sequence[int],
    smiles_length: int,
) -> SparseVector:
    """Count a molecule's vocabulary n-grams. Out-of-vocabulary is silent.

    An n-gram the vocabulary does not contain is simply not looked up
    again: it adds no column, raises nothing, and leaves no trace in the
    vector. That is the whole of the OOV contract.
    """
    hits: dict[int, int] = {}
    for order in orders:
        for ngram, count in document_ngram_counts(tokens, order).items():
            index = term_index.get(ngram)
            if index is not None:
                hits[index] = hits.get(index, 0) + count

    if hits:
        columns = np.fromiter(sorted(hits), dtype=np.int64, count=len(hits))
        values = np.fromiter((hits[c] for c in columns), dtype=np.float64, count=len(hits))
    else:
        columns = np.empty(0, dtype=np.int64)
        values = np.empty(0, dtype=np.float64)
    return SparseVector(columns, values, len(tokens), smiles_length)


# ---------------------------------------------------------------------------
# correlation, without scipy
# ---------------------------------------------------------------------------


def _ranks_with_average_ties(values: np.ndarray) -> np.ndarray:
    """Competition-free ranks, ties sharing their mean rank.

    Averaging ties matters here: token counts are small integers with
    enormous tie groups, and an arbitrary tie order would make Spearman
    depend on sort stability rather than on the data.
    """
    order = np.argsort(values, kind="stable")
    ranked = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranked[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranked


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson product-moment correlation, or None when undefined."""
    if x.size < 2:
        return None
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
    if denominator == 0.0:
        return None
    return float(np.dot(xc, yc) / denominator)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation: Pearson on average-tied ranks."""
    if x.size < 2:
        return None
    return pearson(_ranks_with_average_ties(x), _ranks_with_average_ties(y))


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    return summarize(array("d", values))


# ---------------------------------------------------------------------------
# IDF: smoothed vs unsmoothed
# ---------------------------------------------------------------------------


def idf_comparison(document_frequency: Sequence[int], n_documents: int) -> dict[str, Any]:
    """Both IDF formulas over the actual vocabulary, and their difference."""
    frequencies = np.asarray(document_frequency, dtype=np.float64)
    smoothed = weights.inverse_document_frequency(
        frequencies, n_documents, weights.IDF_SMOOTHED
    )
    unsmoothed = weights.inverse_document_frequency(
        frequencies, n_documents, weights.IDF_UNSMOOTHED
    )
    difference = np.abs(smoothed - unsmoothed)

    bands = []
    lower = 0
    for edge in (*IDF_DF_BANDS, None):
        if edge is None:
            mask = frequencies > lower
            label = f"({lower},inf)"
        else:
            mask = (frequencies > lower) & (frequencies <= edge)
            label = f"({lower},{edge}]"
        if mask.any():
            bands.append(
                {
                    "document_frequency_band": label,
                    "terms": int(mask.sum()),
                    "mean_absolute_difference": float(difference[mask].mean()),
                    "max_absolute_difference": float(difference[mask].max()),
                    "mean_smoothed_idf": float(smoothed[mask].mean()),
                    "mean_unsmoothed_idf": float(unsmoothed[mask].mean()),
                }
            )
        if edge is not None:
            lower = edge

    return {
        "n_documents": n_documents,
        "terms": int(frequencies.size),
        "formulas": {
            "smoothed": weights.idf_formula(weights.IDF_SMOOTHED),
            "unsmoothed": weights.idf_formula(weights.IDF_UNSMOOTHED),
        },
        "smoothed": _summarize(smoothed.tolist()),
        "unsmoothed": _summarize(unsmoothed.tolist()),
        "absolute_difference": _summarize(difference.tolist()),
        "max_relative_difference": float(np.max(difference / np.abs(unsmoothed))),
        "rank_order_identical": bool(
            np.array_equal(np.argsort(smoothed, kind="stable"), np.argsort(unsmoothed, kind="stable"))
        ),
        "by_document_frequency_band": bands,
        "document_frequency": _summarize(frequencies.tolist()),
    }


# ---------------------------------------------------------------------------
# per-molecule accumulation
# ---------------------------------------------------------------------------


class WeightingDiagnostics:
    """Accumulates every per-molecule statistic in a single pass."""

    def __init__(self, idf: np.ndarray, dimension: int) -> None:
        self._idf = np.asarray(idf, dtype=np.float64)
        self._dimension = dimension
        self._strata: dict[str, dict[str, list[float]]] = {
            name: {
                "token_count": [],
                "smiles_length": [],
                "nonzero": [],
                "max_raw_tf": [],
                "raw_tf_top_share": [],
                "raw_tf_herfindahl": [],
                "sublinear_tf_top_share": [],
                "sublinear_tf_herfindahl": [],
                "raw_norm": [],
                "sublinear_norm": [],
                "float32_max_abs_diff": [],
                "float32_mean_abs_diff": [],
                "float32_l2_diff": [],
                "float32_cosine": [],
            }
            for name in STRATUM_NAMES
        }
        self.all_zero_molecules = 0
        self.documents = 0

    def add(self, vector: SparseVector, stratum: str) -> None:
        bucket = self._strata[stratum]
        self.documents += 1
        bucket["token_count"].append(float(vector.token_count))
        bucket["smiles_length"].append(float(vector.smiles_length))
        bucket["nonzero"].append(float(vector.nonzero))

        if vector.nonzero == 0:
            self.all_zero_molecules += 1
            for key in (
                "max_raw_tf",
                "raw_tf_top_share",
                "raw_tf_herfindahl",
                "sublinear_tf_top_share",
                "sublinear_tf_herfindahl",
            ):
                bucket[key].append(0.0)
            bucket["raw_norm"].append(0.0)
            bucket["sublinear_norm"].append(0.0)
            for key in ("float32_max_abs_diff", "float32_mean_abs_diff", "float32_l2_diff"):
                bucket[key].append(0.0)
            # A zero vector is identical in both dtypes; calling that a
            # cosine of 1.0 would be arithmetically undefined, so it is
            # left out of the cosine sample rather than invented.
            return

        idf = self._idf[vector.indices]
        raw_tf = weights.term_frequency(vector.counts, weights.TF_RAW)
        sublinear_tf = weights.term_frequency(vector.counts, weights.TF_SUBLINEAR)

        bucket["max_raw_tf"].append(float(raw_tf.max()))
        bucket["raw_tf_top_share"].append(float(raw_tf.max() / raw_tf.sum()))
        bucket["raw_tf_herfindahl"].append(float(np.sum((raw_tf / raw_tf.sum()) ** 2)))
        bucket["sublinear_tf_top_share"].append(float(sublinear_tf.max() / sublinear_tf.sum()))
        bucket["sublinear_tf_herfindahl"].append(
            float(np.sum((sublinear_tf / sublinear_tf.sum()) ** 2))
        )

        raw_weighted = raw_tf * idf
        sublinear_weighted = sublinear_tf * idf
        bucket["raw_norm"].append(float(np.sqrt(np.dot(raw_weighted, raw_weighted))))
        bucket["sublinear_norm"].append(
            float(np.sqrt(np.dot(sublinear_weighted, sublinear_weighted)))
        )

        # Precision: the recommended configuration (sublinear, L2) computed
        # in float64, then the same vector rounded to float32. Compared on
        # the support only -- the 4,016 zero columns are exactly zero in
        # both and would only dilute the averages.
        exact = weights.normalize(sublinear_weighted, weights.NORM_L2)
        reduced = exact.astype(np.float32).astype(np.float64)
        delta = np.abs(exact - reduced)
        bucket["float32_max_abs_diff"].append(float(delta.max()))
        bucket["float32_mean_abs_diff"].append(float(delta.mean()))
        bucket["float32_l2_diff"].append(float(np.sqrt(np.dot(delta, delta))))
        denominator = float(
            np.sqrt(np.dot(exact, exact)) * np.sqrt(np.dot(reduced, reduced))
        )
        if denominator > 0.0:
            bucket["float32_cosine"].append(float(np.dot(exact, reduced) / denominator))

    # -- read-out ----------------------------------------------------------

    def tf_report(self) -> dict[str, Any]:
        return {
            "formulas": {
                "raw": weights.tf_formula(weights.TF_RAW),
                "sublinear": weights.tf_formula(weights.TF_SUBLINEAR),
            },
            "concentration_metrics": {
                "top_share": "largest feature weight / total feature weight in that molecule",
                "herfindahl": "sum of squared weight shares; 1/nonzero is perfectly even",
            },
            "by_stratum": {
                name: {
                    "molecules": len(bucket["token_count"]),
                    "token_count": _summarize(bucket["token_count"]),
                    "nonzero_features": _summarize(bucket["nonzero"]),
                    "max_feature_raw_tf": _summarize(bucket["max_raw_tf"]),
                    "raw_top_share": _summarize(bucket["raw_tf_top_share"]),
                    "sublinear_top_share": _summarize(bucket["sublinear_tf_top_share"]),
                    "raw_herfindahl": _summarize(bucket["raw_tf_herfindahl"]),
                    "sublinear_herfindahl": _summarize(bucket["sublinear_tf_herfindahl"]),
                }
                for name, bucket in self._strata.items()
            },
        }

    def norm_report(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "question": (
                "without normalization, does vector magnitude encode molecule size?"
            ),
            "by_stratum": {},
            "pooled": {},
        }
        pooled: dict[str, list[float]] = {key: [] for key in
                                          ("token_count", "smiles_length", "raw_norm",
                                           "sublinear_norm")}
        for name, bucket in self._strata.items():
            tokens = np.asarray(bucket["token_count"])
            lengths = np.asarray(bucket["smiles_length"])
            for tf_mode, key in ((weights.TF_RAW, "raw_norm"),
                                 (weights.TF_SUBLINEAR, "sublinear_norm")):
                pooled[key].extend(bucket[key])
            pooled["token_count"].extend(bucket["token_count"])
            pooled["smiles_length"].extend(bucket["smiles_length"])

            out["by_stratum"][name] = {
                "molecules": int(tokens.size),
                "raw_tf_no_norm": {
                    "magnitude": _summarize(bucket["raw_norm"]),
                    "pearson_vs_token_count": pearson(tokens, np.asarray(bucket["raw_norm"])),
                    "spearman_vs_token_count": spearman(tokens, np.asarray(bucket["raw_norm"])),
                },
                "sublinear_tf_no_norm": {
                    "magnitude": _summarize(bucket["sublinear_norm"]),
                    "pearson_vs_token_count": pearson(
                        tokens, np.asarray(bucket["sublinear_norm"])
                    ),
                    "spearman_vs_token_count": spearman(
                        tokens, np.asarray(bucket["sublinear_norm"])
                    ),
                },
            }

        tokens = np.asarray(pooled["token_count"])
        lengths = np.asarray(pooled["smiles_length"])
        for tf_mode, key in ((weights.TF_RAW, "raw_norm"), (weights.TF_SUBLINEAR, "sublinear_norm")):
            magnitudes = np.asarray(pooled[key])
            out["pooled"][f"{tf_mode}_tf_no_norm"] = {
                "magnitude": _summarize(pooled[key]),
                "pearson_vs_token_count": pearson(tokens, magnitudes),
                "spearman_vs_token_count": spearman(tokens, magnitudes),
                "pearson_vs_smiles_length": pearson(lengths, magnitudes),
                "spearman_vs_smiles_length": spearman(lengths, magnitudes),
            }
        # After L2 every vector has magnitude exactly 1 (or exactly 0), so
        # the correlation is not merely small, it is structurally absent.
        out["pooled"]["l2_normalized"] = {
            "magnitude": "exactly 1.0 for every molecule retaining any term, 0.0 otherwise",
            "pearson_vs_token_count": 0.0,
            "note": "size information is removed by construction, not by luck",
        }
        return out

    def precision_report(self) -> dict[str, Any]:
        return {
            "configuration": "sublinear TF, smoothed IDF, L2 norm, computed in float64",
            "comparison": "float64 result vs the same result rounded to float32",
            "measured_on": "the nonzero support of each molecule",
            "by_stratum": {
                name: {
                    "molecules": len(bucket["float32_max_abs_diff"]),
                    "max_absolute_element_difference": _summarize(
                        bucket["float32_max_abs_diff"]
                    ),
                    "mean_absolute_element_difference": _summarize(
                        bucket["float32_mean_abs_diff"]
                    ),
                    "l2_vector_difference": _summarize(bucket["float32_l2_diff"]),
                    "cosine_similarity": _summarize(bucket["float32_cosine"]),
                }
                for name, bucket in self._strata.items()
            },
        }


__all__ = [
    "IDF_DF_BANDS",
    "SparseVector",
    "WeightingDiagnostics",
    "idf_comparison",
    "pearson",
    "spearman",
    "vectorize",
]
