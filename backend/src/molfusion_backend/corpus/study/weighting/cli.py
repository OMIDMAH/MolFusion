"""Command-line entry point for the Phase 5F-C.1 weighting study.

    python -m molfusion_backend.corpus.study.weighting \\
        --corpus     backend/corpus_data/chembl37/canonical_smiles.smi \\
        --output-dir backend/corpus_data/chembl37/studies/tfidf_weighting

Read-only with respect to the corpus and offline. Produces numerical
diagnostics and a recommended weighting contract; writes no production
vocabulary, no IDF payload, and no artifact.
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from molfusion_backend.corpus.errors import CorpusBuildError
from molfusion_backend.corpus.study.runner import (
    FROZEN_DOCUMENT_COUNT,
    FROZEN_FIT_CORPUS_SHA256,
)
from molfusion_backend.corpus.study.weighting.payload import (
    FROZEN_DIMENSION,
    FROZEN_MIN_DF,
    INDEX_ORDER_LEXICOGRAPHIC,
    INDEX_ORDERS,
)
from molfusion_backend.corpus.study.weighting.report import REPORT_FILENAME, run_weighting_study

DEFAULT_PROGRESS_EVERY = 250_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m molfusion_backend.corpus.study.weighting",
        description=(
            "Measure TF, IDF, normalization and precision choices for the frozen "
            "MolFusion SMILES TF-IDF vocabulary. Diagnostics only -- no artifact."
        ),
    )
    parser.add_argument("--corpus", type=Path, required=True, help="Frozen corpus, read-only.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for diagnostics. Must be outside version control.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=FROZEN_FIT_CORPUS_SHA256,
        help="Corpus digest required before analysing. Default: %(default)s",
    )
    parser.add_argument(
        "--expected-documents",
        type=int,
        default=FROZEN_DOCUMENT_COUNT,
        help="Document count required after reading; 0 disables. Default: %(default)s",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=FROZEN_MIN_DF,
        help="Frozen Phase 5F-C rarity floor. Default: %(default)s",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=FROZEN_DIMENSION,
        help="Frozen Phase 5F-C vocabulary dimension. Default: %(default)s",
    )
    parser.add_argument(
        "--index-order",
        choices=INDEX_ORDERS,
        default=INDEX_ORDER_LEXICOGRAPHIC,
        help="Vector index assignment rule. Default: %(default)s",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Re-run the full corpus counting pass even if a matching cache "
            "exists. The cache is keyed by corpus digest and selection "
            "parameters, so a stale one cannot be reused by accident."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help=f"Overwrite an existing {REPORT_FILENAME}."
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Documents between progress lines; 0 disables. Default: %(default)s",
    )
    return parser


def _report_progress(stage: str, count: int) -> None:
    print(f"  ... {stage}: {count:,} documents", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"Studying weighting over {args.corpus}", file=sys.stderr, flush=True)
    try:
        report = run_weighting_study(
            args.corpus,
            args.output_dir,
            expected_sha256=args.expected_sha256,
            expected_document_count=args.expected_documents or None,
            min_df=args.min_df,
            dimension=args.dimension,
            index_order=args.index_order,
            force=args.force,
            use_cache=not args.no_cache,
            progress=_report_progress if args.progress_every else None,
            progress_every=args.progress_every,
        )
    except CorpusBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    vocabulary = report["vocabulary"]
    idf = report["inverse_document_frequency"]
    print(
        "\n".join(
            (
                "",
                f"corpus sha256 verified   {report['corpus']['verified_sha256']}",
                f"documents                {report['corpus']['document_count']:,}",
                "",
                f"eligible at min_df={vocabulary['selection']['min_df']:<5} "
                f"{vocabulary['eligible_terms_at_min_df']:,}",
                f"selected terms           {vocabulary['selected_terms']:,}",
                f"cap is binding           {vocabulary['cap_is_binding']}",
                f"DF range in vocabulary   {vocabulary['document_frequency_min']:,}"
                f" .. {vocabulary['document_frequency_max']:,}",
                "",
                f"IDF smoothed   min/max   {idf['smoothed']['min']:.6f} / {idf['smoothed']['max']:.6f}",
                f"IDF unsmoothed min/max   {idf['unsmoothed']['min']:.6f} / {idf['unsmoothed']['max']:.6f}",
                f"max |smoothed-unsmoothed| {idf['absolute_difference']['max']:.6e}",
                "",
                f"sample molecules         {report['sample']['molecules']:,}",
                f"all-zero in sample       {report['sample']['all_zero_molecules']:,}",
                "",
                f"diagnostics -> {args.output_dir}",
            )
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
