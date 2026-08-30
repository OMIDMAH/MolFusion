"""Study report assembly and deterministic study-output serialization.

Everything a reader would need to re-run the study and get the same
numbers is recorded next to the numbers: the corpus digest that was
actually verified, the contract identifiers the corpus was built under,
the split rule, the thresholds swept, the ranking rule, and the software
that ran. A table without that header is a measurement of an unknown
thing.

Determinism is a property of the whole output, not just the counts. Every
container written here is either an explicitly sorted sequence or a dict
built in a fixed literal order; nothing is serialized straight out of a
set, and no CSV row order depends on dictionary iteration. The only
deliberately volatile fields are wall-clock timings and peak memory, which
`deterministic_study_view()` strips so two runs can be compared directly.
"""

import csv
import ctypes
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import rdkit

from molfusion_backend.chemistry import CANONICAL_SMILES_NORMALIZATION_ID
from molfusion_backend.corpus.errors import CorpusOutputExistsError
from molfusion_backend.corpus.provenance import git_commit
from molfusion_backend.corpus.serialization import CORPUS_ENCODING, CORPUS_SERIALIZATION_ID
from molfusion_backend.corpus.study import runner
from molfusion_backend.corpus.study.coverage import coverage_definition
from molfusion_backend.corpus.study.ngrams import NGRAM_ORDERS, NgramFrequencyAccumulator
from molfusion_backend.corpus.study.split import split_definition
from molfusion_backend.corpus.study.vocabulary import (
    CANDIDATE_DIMENSIONS,
    MIN_DF_THRESHOLDS,
    RARITY_THRESHOLDS,
    SCOPE_CORPUS,
    SCOPE_FIT,
    ranking_definition,
)
from molfusion_backend.smiles_tokenizer import SMILES_TOKENIZER_ID

# Fields excused from run-to-run determinism, as (section, key) paths.
# Kept as one explicit list so "which fields may differ" is answerable by
# reading, not by diffing two runs and guessing.
VOLATILE_STUDY_PATHS = (
    ("run", "started_at"),
    ("run", "elapsed_seconds"),
    ("run", "peak_memory_bytes"),
    ("run", "count_pass_seconds"),
    ("run", "holdout_pass_seconds"),
)


def peak_memory_bytes() -> int | None:
    """Peak working set of this process, or None where unavailable.

    Reported because section 17 asks the study to justify its counter
    implementation with a measurement rather than an assertion. Best
    effort: an unavailable number is recorded as null, never as zero,
    which would read as "measured, and it was nothing".
    """
    try:
        from ctypes import wintypes  # Windows-only module

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.PeakWorkingSetSize)
    except (AttributeError, ImportError, OSError, ValueError):
        pass

    try:  # POSIX fallback
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kibibytes, macOS reports bytes.
        return int(usage) * (1 if platform.system() == "Darwin" else 1024)
    except Exception:  # noqa: BLE001 - provenance must never fail a study
        return None


def run_study(
    corpus_path: Path,
    output_dir: Path,
    *,
    expected_sha256: str = runner.FROZEN_FIT_CORPUS_SHA256,
    expected_document_count: int | None = runner.FROZEN_DOCUMENT_COUNT,
    force: bool = False,
    progress: Callable[[str, int], None] | None = None,
    progress_every: int = 250_000,
) -> dict[str, Any]:
    """Run the full study and write its outputs. Never mutates the corpus.

    The digest check runs before anything else reads a molecule, so an
    unexpected corpus costs one hash rather than an hour of analysis
    against the wrong bytes.
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)

    started = time.time()
    clock = time.perf_counter()
    verified_sha256 = runner.verify_corpus_identity(corpus_path, expected_sha256)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / runner.STUDY_REPORT_FILENAME
    if report_path.exists() and not force:
        raise CorpusOutputExistsError(
            f"Study output already exists: {report_path}. Pass force to overwrite."
        )

    count_started = time.perf_counter()
    accumulator, corpus_observations = runner.count_ngrams(
        corpus_path,
        progress=(lambda n: progress("count", n)) if progress else None,
        progress_every=progress_every,
    )
    count_seconds = time.perf_counter() - count_started

    observed_documents = corpus_observations["documents"]
    if expected_document_count is not None and observed_documents != expected_document_count:
        raise runner.CorpusIdentityError(
            "Corpus document count does not match the frozen corpus: "
            f"expected {expected_document_count:,}, read {observed_documents:,}."
        )

    entries_by_order = {order: accumulator.entries(order) for order in NGRAM_ORDERS}
    families, policy_report = runner.build_families(entries_by_order)

    holdout_started = time.perf_counter()
    coverage_rows = runner.score_holdout(
        corpus_path,
        families,
        progress=(lambda n: progress("holdout", n)) if progress else None,
        progress_every=max(1, progress_every // 10),
    )
    holdout_seconds = time.perf_counter() - holdout_started

    unique_coverage = runner.unique_ngram_coverage(families, entries_by_order)
    for row in coverage_rows:
        row.update(unique_coverage.get((row["family"], row["dimension"]), {}))
        policy = policy_report[row["family"].split("-")[0]]
        retention = policy["unigram_retention_df"]
        detail = retention.get(str(row["dimension"]))

        # Name the pruning this candidate actually applies, so the table
        # answers "which min_df is this?" without the reader re-deriving it
        # from the ranking. Only meaningful for an unprotected DF ranking:
        # a TF cut and a unigram-protected cut are not DF prefixes.
        if row["ranking"] == runner.RANKING_DOCUMENT_FREQUENCY and not row["protected_unigrams"]:
            row["effective_min_df"] = policy["effective_min_df_at_dimension"].get(
                str(row["dimension"])
            )
            row["equivalent_min_df_thresholds"] = sorted(
                int(threshold)
                for threshold, size in policy["min_df_vocabulary_size_fit"].items()
                if size == row["dimension"]
            )
        else:
            row["effective_min_df"] = None
            row["equivalent_min_df_thresholds"] = []
        if row["protected_unigrams"]:
            # By construction every order-1 term precedes every higher
            # order term, so a protected candidate keeps them all as long
            # as it is at least as wide as the unigram vocabulary.
            total = detail["unigrams_total"] if detail else None
            row["all_unigrams_retained"] = (
                None if total is None else row["dimension"] >= total
            )
        elif 1 not in row["ngram_orders"]:
            row["all_unigrams_retained"] = None
        else:
            row["all_unigrams_retained"] = (
                None if detail is None else detail["unigrams_excluded"] == 0
            )

    coverage_rows.sort(key=lambda row: (row["family"], row["dimension"]))

    report = _assemble(
        corpus_path=corpus_path,
        verified_sha256=verified_sha256,
        accumulator=accumulator,
        corpus_observations=corpus_observations,
        entries_by_order=entries_by_order,
        policy_report=policy_report,
        coverage_rows=coverage_rows,
        started=started,
        elapsed=time.perf_counter() - clock,
        count_seconds=count_seconds,
        holdout_seconds=holdout_seconds,
    )

    _write_outputs(output_dir, report, entries_by_order)
    return report


def _assemble(
    *,
    corpus_path: Path,
    verified_sha256: str,
    accumulator: NgramFrequencyAccumulator,
    corpus_observations: dict[str, Any],
    entries_by_order: dict[int, list[Any]],
    policy_report: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
    started: float,
    elapsed: float,
    count_seconds: float,
    holdout_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": runner.STUDY_SCHEMA_VERSION,
        "study_id": runner.STUDY_ID,
        "phase": "5F-C",
        "produces_production_artifact": False,
        "corpus": {
            "filename": corpus_path.name,
            "verified_sha256": verified_sha256,
            "expected_sha256": runner.FROZEN_FIT_CORPUS_SHA256,
            "identity_verified": verified_sha256 == runner.FROZEN_FIT_CORPUS_SHA256,
            "document_count": corpus_observations["documents"],
            "normalization_id": CANONICAL_SMILES_NORMALIZATION_ID,
            "tokenizer_id": SMILES_TOKENIZER_ID,
            "serialization_id": CORPUS_SERIALIZATION_ID,
            "encoding": CORPUS_ENCODING,
            "uses_downstream_labels": False,
        },
        "split": {
            "definition": split_definition(),
            "fit_documents": accumulator.fit_document_count,
            "holdout_documents": accumulator.holdout_document_count,
            "holdout_fraction": (
                accumulator.holdout_document_count / accumulator.document_count
                if accumulator.document_count
                else None
            ),
            "fit_tokens": accumulator.token_count(holdout=False),
            "holdout_tokens": accumulator.token_count(holdout=True),
            "token_count": corpus_observations["token_count"],
        },
        "definitions": {
            "ngrams": accumulator.definition(),
            "ranking": ranking_definition(),
            "coverage": coverage_definition(),
            "thresholds": {
                "min_df": list(MIN_DF_THRESHOLDS),
                "rarity_df_le": list(RARITY_THRESHOLDS),
                "candidate_dimensions": list(CANDIDATE_DIMENSIONS),
            },
        },
        "orders": runner.order_tables(accumulator),
        "min_df_thresholds": {
            SCOPE_CORPUS: runner.min_df_table(entries_by_order, SCOPE_CORPUS),
            SCOPE_FIT: runner.min_df_table(entries_by_order, SCOPE_FIT),
        },
        "policies": policy_report,
        "holdout_coverage": coverage_rows,
        "ranking_comparison": runner.ranking_comparison(entries_by_order),
        "long_molecule_sensitivity": runner.long_molecule_sensitivity(
            accumulator, entries_by_order
        ),
        "run": {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
            "elapsed_seconds": round(elapsed, 3),
            "count_pass_seconds": round(count_seconds, 3),
            "holdout_pass_seconds": round(holdout_seconds, 3),
            "peak_memory_bytes": peak_memory_bytes(),
            "software": {
                "python": platform.python_version(),
                "rdkit": rdkit.__version__,
                "molfusion_git_commit": git_commit(Path(__file__).resolve().parent),
            },
        },
    }


def deterministic_study_view(report: dict[str, Any]) -> dict[str, Any]:
    """The report minus timings and memory, for run-to-run comparison."""
    trimmed = {
        name: dict(value) if isinstance(value, dict) else value
        for name, value in report.items()
    }
    for section, key in VOLATILE_STUDY_PATHS:
        if isinstance(trimmed.get(section), dict):
            trimmed[section] = {k: v for k, v in trimmed[section].items() if k != key}
    return trimmed


def study_report_bytes(report: dict[str, Any]) -> bytes:
    """Deterministic JSON: fixed key order, LF only, no platform newline
    translation."""
    text = json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _flatten_coverage(row: dict[str, Any]) -> dict[str, Any]:
    oov = row["molecule_oov_fraction"]
    nonzero = row["nonzero_features"]
    return {
        "family": row["family"],
        "policy": row["policy"],
        "ranking": row["ranking"],
        "protected_unigrams": row["protected_unigrams"],
        "dimension": row["dimension"],
        "effective_min_df": row["effective_min_df"],
        "all_unigrams_retained": row["all_unigrams_retained"],
        "holdout_documents": row["holdout_documents"],
        "holdout_occurrence_coverage": row["holdout_occurrence_coverage"],
        "holdout_unique_coverage": row.get("holdout_unique_coverage"),
        "oov_mean": oov["mean"],
        "oov_median": oov["median"],
        "oov_p95": oov["p95"],
        "oov_p99": oov["p99"],
        "oov_max": oov["max"],
        "nonzero_mean": nonzero["mean"],
        "nonzero_median": nonzero["median"],
        "nonzero_p95": nonzero["p95"],
        "nonzero_p99": nonzero["p99"],
        "nonzero_max": nonzero["max"],
        "sparsity_at_mean": row["sparsity_at_mean"],
        "all_zero_molecules": row["all_zero_molecules"],
        "all_zero_fraction": row["all_zero_fraction"],
    }


def _write_outputs(
    output_dir: Path, report: dict[str, Any], entries_by_order: dict[int, list[Any]]
) -> None:
    (output_dir / runner.STUDY_REPORT_FILENAME).write_bytes(study_report_bytes(report))

    threshold_rows = (
        report["min_df_thresholds"][SCOPE_CORPUS] + report["min_df_thresholds"][SCOPE_FIT]
    )
    _write_csv(
        output_dir / runner.DF_THRESHOLDS_FILENAME,
        [
            "scope",
            "min_df",
            "unigrams",
            "bigrams",
            "trigrams",
            "combined_1_1",
            "combined_1_2",
            "combined_1_3",
            "combined_2_3",
        ],
        threshold_rows,
    )

    coverage = [_flatten_coverage(row) for row in report["holdout_coverage"]]
    coverage_fields = list(coverage[0]) if coverage else ["family"]
    _write_csv(output_dir / runner.HOLDOUT_COVERAGE_FILENAME, coverage_fields, coverage)

    vocabulary_rows = []
    for name, policy in sorted(report["policies"].items()):
        for threshold, size in policy["min_df_vocabulary_size_fit"].items():
            vocabulary_rows.append(
                {
                    "policy_name": name,
                    "policy": policy["label"],
                    "min_df": int(threshold),
                    "vocabulary_size_fit": size,
                    "vocabulary_size_fit_unpruned": policy["vocabulary_size_fit"],
                    "vocabulary_size_corpus_unpruned": policy["vocabulary_size_corpus"],
                }
            )
    vocabulary_rows.sort(key=lambda row: (row["policy_name"], row["min_df"]))
    _write_csv(
        output_dir / runner.VOCABULARY_COVERAGE_FILENAME,
        [
            "policy_name",
            "policy",
            "min_df",
            "vocabulary_size_fit",
            "vocabulary_size_fit_unpruned",
            "vocabulary_size_corpus_unpruned",
        ],
        vocabulary_rows,
    )

    _write_csv(
        output_dir / runner.RANKING_COMPARISON_FILENAME,
        [
            "policy",
            "dimension",
            "requested_dimension",
            "vocabulary_size_fit",
            "shared_terms",
            "overlap_fraction",
            "jaccard",
        ],
        report["ranking_comparison"],
    )

    top_rows = runner.top_ngram_rows(entries_by_order)
    _write_csv(
        output_dir / runner.TOP_NGRAMS_FILENAME,
        [
            "order",
            "ngram",
            "df_rank",
            "tf_rank",
            "document_frequency",
            "term_frequency",
            "document_frequency_fit",
            "term_frequency_fit",
            "document_frequency_holdout",
            "term_frequency_holdout",
            "occurrences_per_document",
        ],
        top_rows,
    )


__all__ = [
    "VOLATILE_STUDY_PATHS",
    "deterministic_study_view",
    "peak_memory_bytes",
    "run_study",
    "study_report_bytes",
]
