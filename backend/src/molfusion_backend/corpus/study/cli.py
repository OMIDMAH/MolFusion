"""Command-line entry point for the Phase 5F-C vocabulary study.

    python -m molfusion_backend.corpus.study \\
        --corpus     backend/corpus_data/chembl37/canonical_smiles.smi \\
        --output-dir backend/corpus_data/chembl37/studies/ngram_vocabulary

Read-only with respect to the corpus, and offline: it never opens the
ChEMBL SQLite release, never re-canonicalizes, and never rewrites the
corpus it reads. The default expected digest is the frozen Phase 5F-B
corpus, and a mismatch aborts before any analysis runs.
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from molfusion_backend.corpus.errors import CorpusBuildError
from molfusion_backend.corpus.study.report import run_study
from molfusion_backend.corpus.study.runner import (
    FROZEN_DOCUMENT_COUNT,
    FROZEN_FIT_CORPUS_SHA256,
    STUDY_REPORT_FILENAME,
)

DEFAULT_PROGRESS_EVERY = 250_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m molfusion_backend.corpus.study",
        description=(
            "Measure token n-gram vocabulary size, rarity, holdout coverage and "
            "OOV burden over the frozen MolFusion reference corpus. Produces "
            "study tables only -- no vocabulary, no IDF, no artifact."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to the frozen canonical-SMILES corpus. Opened read-only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory to write the study report and CSV tables into. Must be "
            "outside version control (see backend/.gitignore)."
        ),
    )
    parser.add_argument(
        "--expected-sha256",
        default=FROZEN_FIT_CORPUS_SHA256,
        help=(
            "Corpus digest to require before analysing. Defaults to the frozen "
            "Phase 5F-B corpus; override only to study a different corpus "
            "deliberately. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--expected-documents",
        type=int,
        default=FROZEN_DOCUMENT_COUNT,
        help=(
            "Document count to require after reading. 0 disables the check. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Overwrite an existing {STUDY_REPORT_FILENAME} in the output directory.",
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

    print(f"Studying corpus {args.corpus}", file=sys.stderr, flush=True)
    try:
        report = run_study(
            args.corpus,
            args.output_dir,
            expected_sha256=args.expected_sha256,
            expected_document_count=args.expected_documents or None,
            force=args.force,
            progress=_report_progress if args.progress_every else None,
            progress_every=args.progress_every,
        )
    except CorpusBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    corpus = report["corpus"]
    split = report["split"]
    orders = report["orders"]
    print(
        "\n".join(
            (
                "",
                f"corpus sha256 verified   {corpus['verified_sha256']}",
                f"documents                {corpus['document_count']:,}",
                f"study fit documents      {split['fit_documents']:,}",
                f"study holdout documents  {split['holdout_documents']:,}",
                "",
                f"distinct unigrams        {orders['1']['distinct_ngrams_corpus']:,}",
                f"distinct bigrams         {orders['2']['distinct_ngrams_corpus']:,}",
                f"distinct trigrams        {orders['3']['distinct_ngrams_corpus']:,}",
                "",
                f"study outputs -> {args.output_dir}",
            )
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
