"""Command-line entry point for the ChEMBL reference-corpus builder.

    python -m molfusion_backend.corpus \\
        --source-db  E:\\chembl\\chembl_37.db \\
        --output-dir E:\\chembl\\corpus

Downloading and building are deliberately separate concerns: this command
never touches the network. Point it at an official ChEMBL SQLite release
you already have on disk, and it will run fully offline and rebuild without
re-downloading anything.
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from molfusion_backend.corpus.builder import (
    CORPUS_FILENAME,
    DEFAULT_SOURCE_NAME,
    DEFAULT_SOURCE_RELEASE,
    DEFAULT_SOURCE_URL,
    REPORT_FILENAME,
    build_corpus,
    silence_rdkit_parse_logging,
)
from molfusion_backend.corpus.errors import CorpusBuildError

DEFAULT_PROGRESS_EVERY = 250_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m molfusion_backend.corpus",
        description=(
            "Build the deterministic MolFusion reference corpus of canonical "
            "SMILES from an official ChEMBL SQLite release."
        ),
    )

    parser.add_argument(
        "--source-db",
        type=Path,
        required=True,
        help="Path to the decompressed official ChEMBL SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            f"Directory to write {CORPUS_FILENAME} and {REPORT_FILENAME} into. "
            "Must be outside version control (see backend/.gitignore)."
        ),
    )

    provenance = parser.add_argument_group("source provenance")
    provenance.add_argument(
        "--source-archive",
        type=Path,
        default=None,
        help=(
            "Path to the official archive the database was extracted from, so "
            "its exact downloaded bytes are recorded in the build report."
        ),
    )
    provenance.add_argument(
        "--source-name", default=DEFAULT_SOURCE_NAME, help="Default: %(default)s."
    )
    provenance.add_argument(
        "--source-release",
        type=int,
        default=DEFAULT_SOURCE_RELEASE,
        help="ChEMBL release number. Default: %(default)s.",
    )
    provenance.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help="Concrete versioned release URL the source came from.",
    )
    provenance.add_argument(
        "--expected-db-sha256",
        default=None,
        help=(
            "Verify the database against this digest before building. "
            "Use a checksum published by EMBL-EBI, never a third-party one."
        ),
    )
    provenance.add_argument(
        "--expected-archive-sha256",
        default=None,
        help="Verify the archive against this digest before building.",
    )
    provenance.add_argument(
        "--skip-source-checksums",
        action="store_true",
        help=(
            "Record source size and filename but skip hashing. Hashing a "
            "multi-GB release takes minutes; skip it only on a rebuild "
            "against an already-recorded asset. Cannot be combined with an "
            "--expected-*-sha256 verification."
        ),
    )

    behaviour = parser.add_argument_group("build behaviour")
    behaviour.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing corpus. Without it, a build that would "
            "replace finalized output fails instead."
        ),
    )
    behaviour.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Rows between progress lines; 0 disables. Default: %(default)s.",
    )
    behaviour.add_argument(
        "--allow-tokenizer-failures",
        action="store_true",
        help=(
            "Downgrade a Phase 5F-A lossless-tokenization violation from a "
            "build abort to an excluded, counted record. This is a "
            "contract breach, not routine noise -- use only for a documented "
            "exception. The build report always records that it was used."
        ),
    )

    return parser


def _report_progress(rows: int) -> None:
    print(f"  ... {rows:,} rows examined", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.skip_source_checksums and (
        args.expected_db_sha256 or args.expected_archive_sha256
    ):
        print(
            "error: --skip-source-checksums cannot be combined with an "
            "--expected-*-sha256 value; verification requires hashing.",
            file=sys.stderr,
        )
        return 2

    silence_rdkit_parse_logging()

    print(f"Building reference corpus from {args.source_db}", file=sys.stderr, flush=True)
    try:
        report = build_corpus(
            source_db=args.source_db,
            output_dir=args.output_dir,
            source_archive=args.source_archive,
            source_name=args.source_name,
            source_release=args.source_release,
            source_url=args.source_url,
            expected_db_sha256=args.expected_db_sha256,
            expected_archive_sha256=args.expected_archive_sha256,
            compute_source_checksums=not args.skip_source_checksums,
            force=args.force,
            allow_tokenizer_failures=args.allow_tokenizer_failures,
            progress=_report_progress,
            progress_every=args.progress_every,
        )
    except CorpusBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    counts = report["counts"]
    fit_corpus = report["fit_corpus"]
    print(
        "\n".join(
            (
                "",
                f"rows examined            {counts['rows_examined']:,}",
                f"null SMILES              {counts['null_smiles']:,}",
                f"empty SMILES             {counts['empty_smiles']:,}",
                f"RDKit parse failures     {counts['rdkit_parse_failures']:,}",
                f"zero-atom molecules      {counts['zero_atom_molecules']:,}",
                f"valid pre-dedup          {counts['valid_pre_dedup']:,}",
                f"duplicates removed       {counts['duplicate_canonical_smiles']:,}",
                f"unique canonical SMILES  {counts['unique_canonical_smiles']:,}",
                f"tokenization failures    {counts['tokenization_failures']:,}",
                "",
                f"documents                {fit_corpus['document_count']:,}",
                f"corpus bytes             {fit_corpus['size_bytes']:,}",
                f"fit_corpus_sha256        {fit_corpus['sha256']}",
                "",
                f"corpus  -> {args.output_dir / CORPUS_FILENAME}",
                f"report  -> {args.output_dir / REPORT_FILENAME}",
            )
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
