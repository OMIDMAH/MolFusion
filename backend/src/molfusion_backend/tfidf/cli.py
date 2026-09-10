"""Command-line entry point for building and auditing the TF-IDF artifact.

    python -m molfusion_backend.tfidf build \\
        --corpus backend/corpus_data/chembl37/canonical_smiles.smi

    python -m molfusion_backend.tfidf audit
    python -m molfusion_backend.tfidf verify-rebuild \\
        --corpus backend/corpus_data/chembl37/canonical_smiles.smi \\
        --scratch-root E:\\tmp\\tfidf-rebuild

Offline and read-only with respect to the corpus: it never opens the
ChEMBL release, never re-canonicalizes, and never rewrites the corpus. The
default expected digest is the frozen Phase 5F-B corpus and a mismatch
aborts before any counting.

There is no `--force`. An audited artifact version is immutable; use
`verify-rebuild` to confirm a rebuild still reproduces it.
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from molfusion_backend.artifacts.errors import ArtifactError
from molfusion_backend.artifacts.root import resolve_artifact_root
from molfusion_backend.tfidf import contract
from molfusion_backend.tfidf.builder import (
    FROZEN_DOCUMENT_COUNT,
    FROZEN_FIT_CORPUS_SHA256,
    build_artifact,
    rebuild_and_compare,
)
from molfusion_backend.tfidf.loader import load_tfidf_artifact

DEFAULT_PROGRESS_EVERY = 250_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m molfusion_backend.tfidf",
        description=(
            "Build, audit, and verify the frozen MolFusion SMILES TF-IDF artifact."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Artifact root. Defaults to the resolved MolFusion artifact root.",
    )
    parser.add_argument("--artifact-type", default=contract.ARTIFACT_TYPE)
    parser.add_argument("--artifact-id", default=contract.ARTIFACT_ID)
    parser.add_argument("--artifact-version", default=contract.ARTIFACT_VERSION)

    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Fit and finalize the artifact.")
    _add_corpus_arguments(build)
    build.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Documents between progress lines; 0 disables. Default: %(default)s",
    )

    audit = subparsers.add_parser(
        "audit", help="Load and semantically validate an existing artifact."
    )
    audit.add_argument(
        "--expect-fit-corpus-sha256",
        default=FROZEN_FIT_CORPUS_SHA256,
        help="Require the artifact to have been fitted on this corpus.",
    )

    rebuild = subparsers.add_parser(
        "verify-rebuild",
        help="Rebuild into a scratch root and compare payload digests. Never writes "
        "to the existing artifact.",
    )
    _add_corpus_arguments(rebuild)
    rebuild.add_argument(
        "--scratch-root",
        type=Path,
        required=True,
        help="Throwaway artifact root to build into. Must not already contain this version.",
    )
    rebuild.add_argument(
        "--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY
    )
    return parser


def _add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--corpus", type=Path, required=True, help="Frozen corpus, opened read-only."
    )
    parser.add_argument(
        "--expected-sha256",
        default=FROZEN_FIT_CORPUS_SHA256,
        help="Corpus digest required before fitting. Default: %(default)s",
    )
    parser.add_argument(
        "--expected-documents",
        type=int,
        default=FROZEN_DOCUMENT_COUNT,
        help="Document count required after reading; 0 disables. Default: %(default)s",
    )


def _report_progress(count: int) -> None:
    print(f"  ... {count:,} documents", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.artifact_root if args.artifact_root is not None else resolve_artifact_root()

    try:
        if args.command == "build":
            return _build(args, root)
        if args.command == "audit":
            return _audit(args, root)
        return _verify_rebuild(args, root)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build(args, root: Path) -> int:
    print(f"Fitting TF-IDF artifact from {args.corpus}", file=sys.stderr, flush=True)
    report = build_artifact(
        args.corpus,
        root=root,
        artifact_type=args.artifact_type,
        artifact_id=args.artifact_id,
        artifact_version=args.artifact_version,
        expected_sha256=args.expected_sha256,
        expected_document_count=args.expected_documents or None,
        progress=_report_progress if args.progress_every else None,
        progress_every=args.progress_every,
    )
    vocabulary = report["vocabulary"]
    boundary = vocabulary["selection_boundary"]
    print(
        "\n".join(
            (
                "",
                f"fit corpus sha256        {report['fit_corpus']['fit_corpus_sha256']}",
                f"documents                {report['fit_corpus']['document_count']:,}",
                "",
                f"distinct n-grams         {vocabulary['distinct_ngrams_total']:,}",
                f"eligible at min_df={vocabulary['min_df']:<5}  "
                f"{vocabulary['eligible_terms_at_min_df']:,}",
                f"selected dimension       {vocabulary['selected_dimension']:,}",
                f"cap is binding           {boundary['cap_is_binding']}",
                f"boundary DF              {boundary['boundary_document_frequency']:,}",
                f"tied at boundary         {boundary['terms_tied_at_boundary_df']:,} "
                f"({boundary['tied_terms_selected']} in, {boundary['tied_terms_excluded']} out)",
                f"DF range                 {vocabulary['document_frequency_min']:,}"
                f" .. {vocabulary['document_frequency_max']:,}",
                "",
                "payload sha256:",
                *(
                    f"  {name:<22} {digest}"
                    for name, digest in report["payload_sha256"].items()
                ),
                "",
                f"artifact -> {root / args.artifact_type / args.artifact_id / args.artifact_version}",
            )
        ),
        file=sys.stderr,
    )
    return 0


def _audit(args, root: Path) -> int:
    artifact = load_tfidf_artifact(
        args.artifact_id,
        args.artifact_version,
        artifact_type=args.artifact_type,
        root=root,
        expected_fit_corpus_sha256=args.expect_fit_corpus_sha256 or None,
    )
    print(
        "\n".join(
            (
                f"artifact                 {artifact.descriptor.directory}",
                f"dimension                {artifact.dimension:,}",
                f"fit corpus sha256        {artifact.fit_corpus_sha256}",
                f"fit documents            {artifact.config.fit_document_count:,}",
                f"tf/idf/norm              {artifact.config.tf_mode} / "
                f"{artifact.config.idf_mode} / {artifact.config.norm}",
                f"idf dtype                {artifact.idf.dtype}",
                f"index order              {artifact.config.index_order}",
                "",
                "checksums verified, vocabulary and IDF semantically validated.",
            )
        ),
        file=sys.stderr,
    )
    return 0


def _verify_rebuild(args, root: Path) -> int:
    print(f"Rebuilding into {args.scratch_root} for comparison", file=sys.stderr, flush=True)
    result = rebuild_and_compare(
        args.corpus,
        root,
        scratch_root=args.scratch_root,
        artifact_type=args.artifact_type,
        artifact_id=args.artifact_id,
        artifact_version=args.artifact_version,
        expected_sha256=args.expected_sha256,
        expected_document_count=args.expected_documents or None,
        progress=_report_progress if args.progress_every else None,
        progress_every=args.progress_every,
    )
    for filename, comparison in result["scientific_payloads"].items():
        status = "IDENTICAL" if comparison["identical"] else "DIFFERS"
        print(f"  {filename:<22} {status}  {comparison['rebuilt_sha256']}", file=sys.stderr)
    print(
        f"\nall scientific payloads identical: {result['all_identical']}\n"
        f"build report deterministic sections match: "
        f"{result['build_report_deterministic_sections_match']}",
        file=sys.stderr,
    )
    return 0 if result["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
