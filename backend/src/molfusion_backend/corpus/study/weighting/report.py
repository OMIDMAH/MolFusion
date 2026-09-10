"""The Phase 5F-C.1 weighting study: run it, tabulate it, freeze nothing else.

    frozen corpus -> identity gate -> one counting pass
    -> apply the frozen 5F-C selection rule -> candidate 4,096-term vocabulary
    -> IDF from full-corpus document frequencies
    -> numerical diagnostics on a deterministic stratified sample
    -> weighting report

This phase decides how a retained feature becomes a number. It does not
write a production vocabulary payload, an IDF payload, artifact metadata,
or an agent -- those are Phase 5F-D's, and the point of separating them is
that the arithmetic should be settled before anything is packaged.

The one expensive step is the counting pass, which has to see all
2,897,639 molecules because the vocabulary is defined by full-corpus
document frequency. Its result is cached in the study directory, keyed by
the corpus digest and the selection parameters, so re-running the
diagnostics does not re-run the corpus.
"""

import csv
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import rdkit

from molfusion_backend.chemistry import CANONICAL_SMILES_NORMALIZATION_ID
from molfusion_backend.corpus.errors import CorpusIdentityError, CorpusOutputExistsError
from molfusion_backend.corpus.provenance import git_commit
from molfusion_backend.corpus.serialization import CORPUS_SERIALIZATION_ID
from molfusion_backend.corpus.study.ngrams import NgramEntry, NgramFrequencyAccumulator
from molfusion_backend.corpus.study.runner import (
    FROZEN_DOCUMENT_COUNT,
    FROZEN_FIT_CORPUS_SHA256,
    iter_corpus_documents,
    verify_corpus_identity,
)
from molfusion_backend.corpus.study.vocabulary import (
    RANKING_DOCUMENT_FREQUENCY,
    SCOPE_CORPUS,
    frequency,
)
from molfusion_backend.corpus.study.weighting import diagnostics, payload, sampling, weights
from molfusion_backend.smiles_tokenizer import SMILES_TOKENIZER_ID, tokenize_smiles

STUDY_SCHEMA_VERSION = 1
STUDY_ID = "chembl37_tfidf_weighting_contract"

REPORT_FILENAME = "weighting_report.json"
IDF_COMPARISON_FILENAME = "idf_comparison.csv"
TF_CONCENTRATION_FILENAME = "tf_concentration.csv"
NORM_FILENAME = "norm_vs_length.csv"
PRECISION_FILENAME = "precision.csv"
VOCABULARY_PREVIEW_FILENAME = "vocabulary_preview.csv"
CACHE_FILENAME = "corpus_pass_cache.json"

# The preview is a diagnostic, not a payload. It is bounded so nobody can
# mistake it for the production vocabulary file, which Phase 5F-D writes.
VOCABULARY_PREVIEW_LIMIT = 250

# The configuration the diagnostics treat as the candidate contract. The
# study still measures every alternative; this is what the precision and
# concentration comparisons are anchored to.
CANDIDATE_TF_MODE = weights.TF_SUBLINEAR
CANDIDATE_IDF_MODE = weights.IDF_SMOOTHED
CANDIDATE_NORM = weights.NORM_L2
CANDIDATE_IDF_DTYPE = "float64"
CANDIDATE_OUTPUT_DTYPE = "float32"


def corpus_pass(
    corpus_path: Path,
    *,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 250_000,
) -> tuple[NgramFrequencyAccumulator, dict[str, list[str]], int]:
    """One read-only pass: full-corpus n-gram DF plus the diagnostic sample.

    Both jobs are done together because each needs every molecule tokenized
    and tokenization is the expensive part; doing them separately would
    double the only slow step in the phase.
    """
    accumulator = NgramFrequencyAccumulator(payload.FROZEN_NGRAM_ORDERS)
    sample = sampling.empty_strata()
    documents = 0

    for smiles in iter_corpus_documents(corpus_path):
        tokens = tokenize_smiles(smiles)
        # Everything is "fit": the production vocabulary is fitted on the
        # whole corpus, so this study reads corpus-scope frequencies only.
        # The Phase 5F-C holdout was an analysis device and has no role here.
        accumulator.add_document(tokens, holdout=False)
        if sampling.is_sampled(smiles, len(tokens)):
            sample[sampling.stratum_for(len(tokens))].append(smiles)
        documents += 1
        if progress and progress_every and documents % progress_every == 0:
            progress(documents)

    return accumulator, sample, documents


def _cache_header(verified_sha256: str, min_df: int, dimension: int) -> dict[str, Any]:
    return {
        "fit_corpus_sha256": verified_sha256,
        "tokenizer_id": SMILES_TOKENIZER_ID,
        "normalization_id": CANONICAL_SMILES_NORMALIZATION_ID,
        "ngram_orders": list(payload.FROZEN_NGRAM_ORDERS),
        "min_df": min_df,
        "dimension": dimension,
        "sample_id": sampling.SAMPLE_ID,
        "sample_acceptance": dict(sampling.STRATUM_ACCEPTANCE),
    }


def _write_cache(
    path: Path,
    header: dict[str, Any],
    terms: list[payload.VocabularyTerm],
    sample: dict[str, list[str]],
    documents: int,
    eligible_terms: int,
    distinct_by_order: dict[int, int],
) -> None:
    body = {
        "header": header,
        "documents": documents,
        "eligible_terms_at_min_df": eligible_terms,
        "distinct_ngrams_by_order": {str(k): v for k, v in distinct_by_order.items()},
        "terms": [
            {
                "tokens": list(term.tokens),
                "order": term.order,
                "document_frequency": term.document_frequency,
                "selection_rank": term.selection_rank,
            }
            for term in terms
        ],
        "sample": {name: list(values) for name, values in sample.items()},
    }
    path.write_bytes((json.dumps(body, ensure_ascii=False) + "\n").encode("utf-8"))


def _read_cache(path: Path, header: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return body if body.get("header") == header else None


def run_weighting_study(
    corpus_path: Path,
    output_dir: Path,
    *,
    expected_sha256: str = FROZEN_FIT_CORPUS_SHA256,
    expected_document_count: int | None = FROZEN_DOCUMENT_COUNT,
    min_df: int = payload.FROZEN_MIN_DF,
    dimension: int = payload.FROZEN_DIMENSION,
    index_order: str = payload.INDEX_ORDER_LEXICOGRAPHIC,
    force: bool = False,
    use_cache: bool = True,
    progress: Callable[[str, int], None] | None = None,
    progress_every: int = 250_000,
) -> dict[str, Any]:
    """Run the weighting study and write its diagnostics."""
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)

    started = time.time()
    clock = time.perf_counter()
    verified_sha256 = verify_corpus_identity(corpus_path, expected_sha256)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    if report_path.exists() and not force:
        raise CorpusOutputExistsError(
            f"Study output already exists: {report_path}. Pass force to overwrite."
        )

    header = _cache_header(verified_sha256, min_df, dimension)
    cache_path = output_dir / CACHE_FILENAME
    cached = _read_cache(cache_path, header) if use_cache else None

    if cached is not None:
        documents = cached["documents"]
        eligible_terms = cached["eligible_terms_at_min_df"]
        distinct_by_order = {int(k): v for k, v in cached["distinct_ngrams_by_order"].items()}
        selected_terms = [
            payload.VocabularyTerm(
                index=0,
                tokens=tuple(record["tokens"]),
                order=record["order"],
                document_frequency=record["document_frequency"],
                selection_rank=record["selection_rank"],
            )
            for record in cached["terms"]
        ]
        # Re-derive indices from the cached selection so the ordering rule
        # is applied by code on every run, never read back from the cache.
        selected_terms.sort(key=lambda term: term.selection_rank)
        terms = _index_cached(selected_terms, index_order)
        sample = {name: list(values) for name, values in cached["sample"].items()}
        pass_seconds = 0.0
        cache_used = True
    else:
        pass_started = time.perf_counter()
        accumulator, sample, documents = corpus_pass(
            corpus_path,
            progress=(lambda n: progress("count", n)) if progress else None,
            progress_every=progress_every,
        )
        pass_seconds = time.perf_counter() - pass_started

        if expected_document_count is not None and documents != expected_document_count:
            raise CorpusIdentityError(
                "Corpus document count does not match the frozen corpus: "
                f"expected {expected_document_count:,}, read {documents:,}."
            )

        entries_by_order = {
            order: accumulator.entries(order) for order in payload.FROZEN_NGRAM_ORDERS
        }
        distinct_by_order = {
            order: len(entries) for order, entries in entries_by_order.items()
        }
        eligible_terms = sum(
            1
            for entries in entries_by_order.values()
            for entry in entries
            if frequency(entry, RANKING_DOCUMENT_FREQUENCY, SCOPE_CORPUS) >= min_df
        )
        selected = payload.select_terms(
            entries_by_order, min_df=min_df, dimension=dimension
        )
        terms = payload.assign_indices(selected, order=index_order)
        _write_cache(
            cache_path,
            header,
            payload.assign_indices(selected, order=payload.INDEX_ORDER_RANKING),
            sample,
            documents,
            eligible_terms,
            distinct_by_order,
        )
        cache_used = False

    # -- IDF from full-corpus document frequencies -------------------------
    frequencies = payload.document_frequencies(terms)
    idf_report = diagnostics.idf_comparison(frequencies, documents)
    idf = weights.inverse_document_frequency(
        np.asarray(frequencies, dtype=np.float64), documents, CANDIDATE_IDF_MODE
    )

    # -- per-molecule diagnostics on the deterministic sample --------------
    diagnostics_started = time.perf_counter()
    index = payload.term_index(terms)
    accumulator_ = diagnostics.WeightingDiagnostics(idf, len(terms))
    for stratum, molecules in sample.items():
        for smiles in molecules:
            tokens = tokenize_smiles(smiles)
            accumulator_.add(
                diagnostics.vectorize(tokens, index, payload.FROZEN_NGRAM_ORDERS, len(smiles)),
                stratum,
            )
    diagnostics_seconds = time.perf_counter() - diagnostics_started

    report = _assemble(
        corpus_path=corpus_path,
        verified_sha256=verified_sha256,
        documents=documents,
        distinct_by_order=distinct_by_order,
        eligible_terms=eligible_terms,
        min_df=min_df,
        dimension=dimension,
        index_order=index_order,
        terms=terms,
        idf_report=idf_report,
        tf_report=accumulator_.tf_report(),
        norm_report=accumulator_.norm_report(),
        precision_report=accumulator_.precision_report(),
        sample=sample,
        all_zero=accumulator_.all_zero_molecules,
        started=started,
        elapsed=time.perf_counter() - clock,
        pass_seconds=pass_seconds,
        diagnostics_seconds=diagnostics_seconds,
        cache_used=cache_used,
    )
    _write_outputs(output_dir, report, terms)
    return report


def _index_cached(
    terms: list[payload.VocabularyTerm], index_order: str
) -> list[payload.VocabularyTerm]:
    if index_order == payload.INDEX_ORDER_RANKING:
        ordered = list(terms)
    elif index_order == payload.INDEX_ORDER_LEXICOGRAPHIC:
        ordered = sorted(terms, key=lambda term: term.tokens)
    else:
        raise ValueError(f"unknown index ordering: {index_order!r}")
    return [
        payload.VocabularyTerm(
            index=position,
            tokens=term.tokens,
            order=term.order,
            document_frequency=term.document_frequency,
            selection_rank=term.selection_rank,
        )
        for position, term in enumerate(ordered)
    ]


def _assemble(**kw: Any) -> dict[str, Any]:
    terms: list[payload.VocabularyTerm] = kw["terms"]
    composition: dict[str, int] = {}
    for term in terms:
        key = str(term.order)
        composition[key] = composition.get(key, 0) + 1

    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "phase": "5F-C.1",
        "purpose": "freeze the numerical TF-IDF weighting contract",
        "produces_production_artifact": False,
        "corpus": {
            "filename": kw["corpus_path"].name,
            "verified_sha256": kw["verified_sha256"],
            "expected_sha256": FROZEN_FIT_CORPUS_SHA256,
            "identity_verified": kw["verified_sha256"] == FROZEN_FIT_CORPUS_SHA256,
            "document_count": kw["documents"],
            "normalization_id": CANONICAL_SMILES_NORMALIZATION_ID,
            "tokenizer_id": SMILES_TOKENIZER_ID,
            "serialization_id": CORPUS_SERIALIZATION_ID,
            "uses_downstream_labels": False,
        },
        "vocabulary": {
            "selection": payload.selection_definition(
                kw["min_df"], kw["dimension"], payload.FROZEN_NGRAM_ORDERS
            ),
            "distinct_ngrams_by_order": {
                str(k): v for k, v in kw["distinct_by_order"].items()
            },
            "eligible_terms_at_min_df": kw["eligible_terms"],
            "selected_terms": len(terms),
            "cap_is_binding": kw["eligible_terms"] > kw["dimension"],
            "composition_by_order": composition,
            "document_frequency_min": min(t.document_frequency for t in terms),
            "document_frequency_max": max(t.document_frequency for t in terms),
            "indexing": payload.index_ordering_definition(kw["index_order"]),
            "payload_schema": payload.payload_schema(),
        },
        "sample": {
            "definition": sampling.sampling_definition(),
            "molecules": sum(len(v) for v in kw["sample"].values()),
            "by_stratum": sampling.stratum_sizes(kw["sample"]),
            "all_zero_molecules": kw["all_zero"],
        },
        "term_frequency": kw["tf_report"],
        "inverse_document_frequency": kw["idf_report"],
        "normalization": kw["norm_report"],
        "precision": kw["precision_report"],
        "candidate_contract": weights.weighting_definition(
            CANDIDATE_TF_MODE,
            CANDIDATE_IDF_MODE,
            CANDIDATE_NORM,
            CANDIDATE_IDF_DTYPE,
            CANDIDATE_OUTPUT_DTYPE,
        ),
        "run": {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(kw["started"])),
            "elapsed_seconds": round(kw["elapsed"], 3),
            "corpus_pass_seconds": round(kw["pass_seconds"], 3),
            "diagnostics_seconds": round(kw["diagnostics_seconds"], 3),
            "corpus_pass_cached": kw["cache_used"],
            "software": {
                "python": platform.python_version(),
                "rdkit": rdkit.__version__,
                "numpy": np.__version__,
                "sklearn": _sklearn_version(),
                "molfusion_git_commit": git_commit(Path(__file__).resolve().parent),
            },
        },
    }


def _sklearn_version() -> str | None:
    try:
        import sklearn

        return sklearn.__version__
    except ImportError:
        return None


VOLATILE_STUDY_PATHS = (
    ("run", "started_at"),
    ("run", "elapsed_seconds"),
    ("run", "corpus_pass_seconds"),
    ("run", "diagnostics_seconds"),
    ("run", "corpus_pass_cached"),
)


def deterministic_study_view(report: dict[str, Any]) -> dict[str, Any]:
    trimmed = {
        name: dict(value) if isinstance(value, dict) else value
        for name, value in report.items()
    }
    for section, key in VOLATILE_STUDY_PATHS:
        if isinstance(trimmed.get(section), dict):
            trimmed[section] = {k: v for k, v in trimmed[section].items() if k != key}
    return trimmed


def study_report_bytes(report: dict[str, Any]) -> bytes:
    text = json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    output_dir: Path, report: dict[str, Any], terms: list[payload.VocabularyTerm]
) -> None:
    (output_dir / REPORT_FILENAME).write_bytes(study_report_bytes(report))

    _write_csv(
        output_dir / IDF_COMPARISON_FILENAME,
        [
            "document_frequency_band",
            "terms",
            "mean_smoothed_idf",
            "mean_unsmoothed_idf",
            "mean_absolute_difference",
            "max_absolute_difference",
        ],
        report["inverse_document_frequency"]["by_document_frequency_band"],
    )

    tf_rows = []
    for stratum, block in report["term_frequency"]["by_stratum"].items():
        tf_rows.append(
            {
                "stratum": stratum,
                "molecules": block["molecules"],
                "token_count_median": block["token_count"]["median"],
                "nonzero_mean": block["nonzero_features"]["mean"],
                "max_feature_raw_tf_mean": block["max_feature_raw_tf"]["mean"],
                "max_feature_raw_tf_p95": block["max_feature_raw_tf"]["p95"],
                "max_feature_raw_tf_p99": block["max_feature_raw_tf"]["p99"],
                "max_feature_raw_tf_max": block["max_feature_raw_tf"]["max"],
                "raw_top_share_mean": block["raw_top_share"]["mean"],
                "raw_top_share_p99": block["raw_top_share"]["p99"],
                "sublinear_top_share_mean": block["sublinear_top_share"]["mean"],
                "sublinear_top_share_p99": block["sublinear_top_share"]["p99"],
                "raw_herfindahl_mean": block["raw_herfindahl"]["mean"],
                "sublinear_herfindahl_mean": block["sublinear_herfindahl"]["mean"],
            }
        )
    _write_csv(output_dir / TF_CONCENTRATION_FILENAME, list(tf_rows[0]) if tf_rows else ["stratum"], tf_rows)

    norm_rows = []
    for stratum, block in report["normalization"]["by_stratum"].items():
        for mode in ("raw_tf_no_norm", "sublinear_tf_no_norm"):
            norm_rows.append(
                {
                    "stratum": stratum,
                    "tf_mode": mode.split("_tf_")[0],
                    "molecules": block["molecules"],
                    "magnitude_median": block[mode]["magnitude"]["median"],
                    "magnitude_max": block[mode]["magnitude"]["max"],
                    "pearson_vs_token_count": block[mode]["pearson_vs_token_count"],
                    "spearman_vs_token_count": block[mode]["spearman_vs_token_count"],
                }
            )
    for mode, block in report["normalization"]["pooled"].items():
        if mode == "l2_normalized":
            continue
        norm_rows.append(
            {
                "stratum": "pooled",
                "tf_mode": mode.split("_tf_")[0],
                "molecules": report["sample"]["molecules"],
                "magnitude_median": block["magnitude"]["median"],
                "magnitude_max": block["magnitude"]["max"],
                "pearson_vs_token_count": block["pearson_vs_token_count"],
                "spearman_vs_token_count": block["spearman_vs_token_count"],
            }
        )
    _write_csv(
        output_dir / NORM_FILENAME,
        [
            "stratum",
            "tf_mode",
            "molecules",
            "magnitude_median",
            "magnitude_max",
            "pearson_vs_token_count",
            "spearman_vs_token_count",
        ],
        norm_rows,
    )

    precision_rows = [
        {
            "stratum": stratum,
            "molecules": block["molecules"],
            "max_abs_element_diff_max": block["max_absolute_element_difference"]["max"],
            "max_abs_element_diff_mean": block["max_absolute_element_difference"]["mean"],
            "mean_abs_element_diff_mean": block["mean_absolute_element_difference"]["mean"],
            "l2_vector_diff_max": block["l2_vector_difference"]["max"],
            "l2_vector_diff_mean": block["l2_vector_difference"]["mean"],
            "cosine_min": block["cosine_similarity"]["min"],
            "cosine_mean": block["cosine_similarity"]["mean"],
        }
        for stratum, block in report["precision"]["by_stratum"].items()
    ]
    _write_csv(
        output_dir / PRECISION_FILENAME,
        [
            "stratum",
            "molecules",
            "max_abs_element_diff_max",
            "max_abs_element_diff_mean",
            "mean_abs_element_diff_mean",
            "l2_vector_diff_max",
            "l2_vector_diff_mean",
            "cosine_min",
            "cosine_mean",
        ],
        precision_rows,
    )

    # Bounded preview only. The production vocabulary payload is Phase
    # 5F-D's to write, and this file is deliberately too short to be it.
    preview = [
        {
            "index": term.index,
            "tokens": payload.encode_tokens(term.tokens),
            "order": term.order,
            "document_frequency": term.document_frequency,
            "selection_rank": term.selection_rank,
        }
        for term in terms[:VOCABULARY_PREVIEW_LIMIT]
    ]
    _write_csv(
        output_dir / VOCABULARY_PREVIEW_FILENAME,
        ["index", "tokens", "order", "document_frequency", "selection_rank"],
        preview,
    )


__all__ = [
    "CACHE_FILENAME",
    "CANDIDATE_IDF_MODE",
    "CANDIDATE_NORM",
    "CANDIDATE_TF_MODE",
    "REPORT_FILENAME",
    "STUDY_ID",
    "STUDY_SCHEMA_VERSION",
    "VOCABULARY_PREVIEW_LIMIT",
    "VOLATILE_STUDY_PATHS",
    "corpus_pass",
    "deterministic_study_view",
    "run_weighting_study",
    "study_report_bytes",
]
