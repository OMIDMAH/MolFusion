"""The ChEMBL reference-corpus build pipeline.

    ChEMBL SQLite -> structure extraction -> MolFusion canonicalization
    -> lossless tokenizer validation -> drop unusable records
    -> deduplicate on canonical SMILES -> lexicographic sort
    -> logical corpus bytes -> SHA-256 -> build report

No fitting happens here and no fitting parameter is chosen here. The
output is an unsupervised reference corpus plus the provenance needed to
reproduce it.
"""

import json
import os
import platform
import shutil
import sqlite3
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rdkit
from rdkit import RDLogger

from molfusion_backend.artifacts.checksum import sha256_file
from molfusion_backend.chemistry import (
    CANONICAL_SMILES_NORMALIZATION_ID,
    canonical_smiles_from_mol,
    parse_smiles,
)
from molfusion_backend.corpus import chembl
from molfusion_backend.corpus.errors import (
    CorpusBuildError,
    CorpusOutputExistsError,
    TokenizerContractViolation,
)
from molfusion_backend.corpus.provenance import git_commit
from molfusion_backend.corpus.serialization import (
    CORPUS_ENCODING,
    CORPUS_HAS_FINAL_NEWLINE,
    CORPUS_NEWLINE,
    CORPUS_SERIALIZATION_ID,
    write_corpus,
)
from molfusion_backend.corpus.statistics import CorpusStatisticsAccumulator, RecordCounts
from molfusion_backend.smiles_tokenizer import SMILES_TOKENIZER_ID, tokenize_smiles

REPORT_SCHEMA_VERSION = 1
CORPUS_ID = "chembl37_canonical_smiles"
CORPUS_FILENAME = "canonical_smiles.smi"
REPORT_FILENAME = "corpus_build_report.json"

DEFAULT_SOURCE_NAME = "ChEMBL"
DEFAULT_SOURCE_RELEASE = 37
DEFAULT_SOURCE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
    "chembl_37/chembl_37_sqlite.tar.gz"
)

# The one volatile field in the report: everything else is a function of
# the source database and the frozen contracts, so two builds of the same
# source differ here and nowhere else. deterministic_report_view() strips
# it, which is what makes byte-comparison of two reports a meaningful
# determinism check.
VOLATILE_REPORT_PATH = ("build", "built_at")


@dataclass(frozen=True, slots=True)
class SourceAsset:
    """One provenance-hashed source file.

    `sha256` is provenance -- it identifies the bytes a build consumed. It
    is emphatically *not* fit_corpus_sha256, which identifies the logical
    corpus a future fit will consume. Conflating the two would let an
    archive repack or a SQLite VACUUM look like a corpus change (or vice
    versa).
    """

    role: str
    path: Path
    sha256: str | None
    size_bytes: int
    expected_sha256: str | None = None

    @property
    def checksum_verified(self) -> bool | None:
        """True/False when an expected digest was supplied and could be
        compared, None when there was nothing to verify against."""
        if self.expected_sha256 is None or self.sha256 is None:
            return None
        return self.sha256.lower() == self.expected_sha256.lower()

    def as_report(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "expected_sha256": self.expected_sha256,
            "checksum_verified": self.checksum_verified,
        }


def describe_source_asset(
    role: str,
    path: Path,
    expected_sha256: str | None = None,
    compute_checksum: bool = True,
) -> SourceAsset:
    """Hash and size a source file for the provenance record.

    `compute_checksum=False` records size and filename but leaves sha256
    null -- hashing a multi-GB release costs minutes and a rebuild against
    an unchanged asset does not need to repeat it. The report always shows
    which case applied rather than implying a hash that was never taken.
    """
    if not path.is_file():
        raise CorpusBuildError(f"Source asset for {role!r} not found: {path}")

    return SourceAsset(
        role=role,
        path=path,
        sha256=sha256_file(path) if compute_checksum else None,
        size_bytes=path.stat().st_size,
        expected_sha256=expected_sha256,
    )


def _validate_tokenization(canonical: str, record: chembl.SourceRecord) -> int:
    """Return the token count for `canonical`, or raise if it violates the
    Phase 5F-A lossless invariant.

    An explicit check, never a bare `assert` -- `python -O` strips those,
    and this validation must not be optimizable away. The caller decides
    whether to downgrade the raised violation to a counted exclusion;
    naming the source record here is why validation runs in the streaming
    pass, where the molregno and ChEMBL accession are still in hand, rather
    than after deduplication where they are not.
    """
    identity = f"molregno={record.molregno} chembl_id={record.chembl_id!r}"
    try:
        tokens = tokenize_smiles(canonical)
    except ValueError as exc:
        raise TokenizerContractViolation(
            "Canonical SMILES from the MolFusion normalizer could not be "
            f"tokenized ({identity}): {canonical!r} ({exc})"
        ) from exc

    if "".join(tokens) != canonical:
        raise TokenizerContractViolation(
            f"Tokenization was lossy ({identity}) for canonical SMILES "
            f"{canonical!r}: rejoined tokens gave {''.join(tokens)!r}."
        )

    return len(tokens)


def _collect_documents(
    records: Iterable[chembl.SourceRecord],
    counts: RecordCounts,
    allow_tokenizer_failures: bool = False,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 0,
) -> dict[str, int]:
    """Stream source rows into {canonical SMILES: token count}, counting
    every exclusion.

    One pass does canonicalization, tokenizer validation and deduplication
    together. Memory discipline (a full release is millions of rows): only
    the canonical string and its token count are retained -- each RDKit Mol
    and each token tuple goes out of scope at the end of its iteration, and
    the source table is never materialized.
    """
    documents: dict[str, int] = {}

    for record in records:
        counts.rows_examined += 1
        if (
            progress is not None
            and progress_every > 0
            and counts.rows_examined % progress_every == 0
        ):
            progress(counts.rows_examined)

        source_smiles = record.smiles
        if source_smiles is None:
            counts.null_smiles += 1
            continue
        if not source_smiles.strip():
            # Whitespace-only is empty in substance; RDKit would parse it
            # to a zero-atom molecule and it must not become a document.
            counts.empty_smiles += 1
            continue

        # parse_smiles() + canonical_smiles_from_mol() together are exactly
        # what canonicalize_smiles() does. They are used separately here
        # only so the intermediate Mol is available for the zero-atom check
        # and so a parse failure can be counted instead of raised -- no
        # normalization logic is duplicated or altered.
        mol, _error = parse_smiles(source_smiles)
        if mol is None:
            counts.rdkit_parse_failures += 1
            continue

        if mol.GetNumAtoms() == 0:
            # Defensive: with the empty/whitespace check above, the only
            # string RDKit currently parses to zero atoms is "", which is
            # already excluded. Kept as its own category so that if a
            # release or an RDKit version ever produces one, it is counted
            # rather than emitted as a meaningless empty document.
            counts.zero_atom_molecules += 1
            continue

        canonical = canonical_smiles_from_mol(mol)
        if not canonical:
            # Belt-and-braces: a non-zero-atom molecule should never
            # serialize to "". Counted in the same category rather than
            # silently emitted as an empty document.
            counts.zero_atom_molecules += 1
            continue

        # Deduplication keys on the exact canonical string, so a repeat is
        # byte-identical to one already validated and needs no re-check.
        if canonical in documents:
            counts.valid_pre_dedup += 1
            continue

        try:
            token_count = _validate_tokenization(canonical, record)
        except TokenizerContractViolation:
            if not allow_tokenizer_failures:
                raise
            counts.tokenization_failures += 1
            continue

        counts.valid_pre_dedup += 1
        documents[canonical] = token_count

    counts.unique_canonical_smiles = len(documents)
    counts.duplicate_canonical_smiles = counts.valid_pre_dedup - counts.unique_canonical_smiles
    counts.document_count = counts.unique_canonical_smiles
    return documents


def _prepare_output_paths(output_dir: Path, force: bool) -> tuple[Path, Path]:
    corpus_path = output_dir / CORPUS_FILENAME
    report_path = output_dir / REPORT_FILENAME

    if not force:
        existing = [path for path in (corpus_path, report_path) if path.exists()]
        if existing:
            raise CorpusOutputExistsError(
                "Refusing to overwrite existing corpus output: "
                f"{[str(path) for path in existing]}. Pass force=True "
                "(--force) to rebuild in place."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    return corpus_path, report_path


def build_corpus(
    source_db: Path,
    output_dir: Path,
    source_archive: Path | None = None,
    source_name: str = DEFAULT_SOURCE_NAME,
    source_release: int = DEFAULT_SOURCE_RELEASE,
    source_url: str = DEFAULT_SOURCE_URL,
    expected_db_sha256: str | None = None,
    expected_archive_sha256: str | None = None,
    compute_source_checksums: bool = True,
    force: bool = False,
    allow_tokenizer_failures: bool = False,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Build the reference corpus and return its build report.

    Both output files are written into a staging directory first and only
    moved into place once the whole build has succeeded, so a build that
    fails midway leaves any previous corpus untouched rather than replacing
    it with a truncated one.
    """
    source_db = Path(source_db)
    output_dir = Path(output_dir)
    corpus_path, report_path = _prepare_output_paths(output_dir, force)

    assets = [
        describe_source_asset(
            "database", source_db, expected_db_sha256, compute_source_checksums
        )
    ]
    if source_archive is not None:
        assets.append(
            describe_source_asset(
                "archive",
                Path(source_archive),
                expected_archive_sha256,
                compute_source_checksums,
            )
        )

    failed_verification = [asset.role for asset in assets if asset.checksum_verified is False]
    if failed_verification:
        raise CorpusBuildError(
            f"Source checksum verification failed for: {failed_verification}. "
            "The source asset does not match the expected digest; refusing to build."
        )

    counts = RecordCounts()
    accumulator = CorpusStatisticsAccumulator()

    connection = chembl.open_source_database(source_db)
    with_compound_ids = chembl.has_compound_dictionary(connection)
    query = chembl.structure_query(with_compound_ids)
    records = chembl.iter_source_records(connection)
    try:
        token_counts = _collect_documents(
            records,
            counts,
            allow_tokenizer_failures=allow_tokenizer_failures,
            progress=progress,
            progress_every=progress_every,
        )
    finally:
        # Close the generator before the connection: abandoning it mid-scan
        # (as a tokenizer-contract abort does) otherwise leaves its own
        # cursor cleanup to run at GC time, against a connection that has
        # already been closed.
        records.close()
        connection.close()

    # The single ordering decision in the pipeline. Python's str ordering
    # is by Unicode code point and is locale-independent, so the sequence
    # -- and therefore the corpus checksum -- does not vary with the host's
    # locale, filesystem, or the order rows came out of SQLite.
    documents = sorted(token_counts)
    counts.validate()

    if not documents:
        raise CorpusBuildError(
            "Refusing to finalize an empty corpus: no usable structures were "
            f"extracted from {source_db}."
        )

    # Accumulated over the sorted documents so the statistics are a
    # function of the corpus alone, never of dict insertion order.
    for canonical in documents:
        accumulator.add(canonical, token_counts[canonical])
    del token_counts

    staging_dir = output_dir / f".build-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    try:
        staged_corpus = staging_dir / CORPUS_FILENAME
        staged_report = staging_dir / REPORT_FILENAME

        fit_corpus_sha256, corpus_size = write_corpus(staged_corpus, documents)

        report = _build_report(
            corpus_id=CORPUS_ID,
            source_name=source_name,
            source_release=source_release,
            source_url=source_url,
            assets=assets,
            query=query,
            with_compound_ids=with_compound_ids,
            counts=counts,
            accumulator=accumulator,
            fit_corpus_sha256=fit_corpus_sha256,
            corpus_size=corpus_size,
            allow_tokenizer_failures=allow_tokenizer_failures,
        )
        staged_report.write_bytes(_report_bytes(report))

        os.replace(staged_corpus, corpus_path)
        os.replace(staged_report, report_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return report


def _report_bytes(report: dict[str, Any]) -> bytes:
    """Serialize the report deterministically: fixed key order, LF only, no
    platform newline translation."""
    text = json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def _build_report(
    corpus_id: str,
    source_name: str,
    source_release: int,
    source_url: str,
    assets: list[SourceAsset],
    query: str,
    with_compound_ids: bool,
    counts: RecordCounts,
    accumulator: CorpusStatisticsAccumulator,
    fit_corpus_sha256: str,
    corpus_size: int,
    allow_tokenizer_failures: bool,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "corpus_id": corpus_id,
        # Everything the corpus bytes depend on. Recorded explicitly so a
        # later phase can assert the contracts it assumes, rather than
        # trusting that they never changed.
        "contract": {
            "normalization_id": CANONICAL_SMILES_NORMALIZATION_ID,
            "tokenizer_id": SMILES_TOKENIZER_ID,
            "serialization_id": CORPUS_SERIALIZATION_ID,
            "encoding": CORPUS_ENCODING,
            "newline": CORPUS_NEWLINE,
            "final_newline": CORPUS_HAS_FINAL_NEWLINE,
            "deduplication_key": "canonical_isomeric_smiles",
            "sort": "lexicographic_unicode_codepoint",
        },
        "source": {
            "name": source_name,
            "release": source_release,
            "url": source_url,
            "query": query,
            "compound_ids_available": with_compound_ids,
            "assets": {asset.role: asset.as_report() for asset in assets},
            # Stated, not merely implied: nothing downstream of structure
            # was read, so the corpus cannot encode a supervised signal.
            "uses_downstream_labels": False,
        },
        "counts": counts.as_report(),
        "fit_corpus": {
            "filename": CORPUS_FILENAME,
            "sha256": fit_corpus_sha256,
            "size_bytes": corpus_size,
            "document_count": counts.document_count,
        },
        "statistics": accumulator.as_report(),
        "build": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "tokenizer_failures_allowed": allow_tokenizer_failures,
            "software": {
                "python": platform.python_version(),
                "rdkit": rdkit.__version__,
                "sqlite": sqlite3.sqlite_version,
                "molfusion_git_commit": git_commit(Path(__file__).resolve().parent),
            },
        },
    }


def deterministic_report_view(report: dict[str, Any]) -> dict[str, Any]:
    """The report minus its volatile fields.

    Two builds of the same source database must produce identical output
    from this function. Kept next to VOLATILE_REPORT_PATH so the set of
    fields excused from determinism stays a single explicit list rather
    than something each caller re-derives.
    """
    section, key = VOLATILE_REPORT_PATH
    trimmed = {name: dict(value) if isinstance(value, dict) else value
               for name, value in report.items()}
    if section in trimmed and isinstance(trimmed[section], dict):
        trimmed[section] = {k: v for k, v in trimmed[section].items() if k != key}
    return trimmed


def silence_rdkit_parse_logging() -> None:
    """Suppress RDKit's per-molecule parse warnings for a bulk build.

    A full release contains enough unparseable legacy records to emit tens
    of thousands of warning lines, which would bury the build's own
    progress output. The records are still counted in
    counts.rdkit_parse_failures -- only RDKit's stderr chatter is muted.
    """
    RDLogger.DisableLog("rdApp.*")


__all__ = [
    "CORPUS_FILENAME",
    "CORPUS_ID",
    "REPORT_FILENAME",
    "REPORT_SCHEMA_VERSION",
    "SourceAsset",
    "build_corpus",
    "describe_source_asset",
    "deterministic_report_view",
    "silence_rdkit_parse_logging",
]
