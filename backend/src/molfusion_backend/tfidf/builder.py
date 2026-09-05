"""Building the frozen SMILES TF-IDF artifact.

    frozen corpus -> identity gate -> stream + tokenize -> exact full-corpus DF
    -> deterministic selection -> lexicographic indexing -> IDF
    -> staged payloads -> checksums -> metadata -> validate -> atomic finalize

Three properties the implementation is arranged around:

**Nothing half-written becomes an artifact.** Every payload is written into
a staging directory beside the destination, validated there, and only then
moved into place with a single rename. A build that fails at any point
leaves no artifact directory at all, rather than one containing three of
four payloads.

**An existing version is never overwritten.** Artifact versions are
immutable once audited. There is no `--force`: to check that a rebuild
still reproduces an artifact, build into a temporary directory and compare
payload digests (`rebuild_and_compare`), which answers the question
without putting the audited bytes at risk.

**The checksum graph is a DAG, not a cycle.** The three scientific
payloads are written and hashed first; `build_report.json` records those
three digests and is hashed afterwards; `metadata.json` records all four
and is not hashed by anything. No file ever has to contain its own digest.
"""

import json
import os
import platform
import shutil
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rdkit

from molfusion_backend.artifacts.checksum import sha256_file
from molfusion_backend.chemistry import CANONICAL_SMILES_NORMALIZATION_ID
from molfusion_backend.corpus.provenance import git_commit, working_tree_is_clean
from molfusion_backend.corpus.serialization import CORPUS_ENCODING, CORPUS_SERIALIZATION_ID
from molfusion_backend.smiles_tokenizer import SMILES_TOKENIZER_ID, tokenize_smiles
from molfusion_backend.tfidf import contract, idf as idf_module, vocabulary as vocabulary_module
from molfusion_backend.tfidf.errors import (
    TfidfArtifactExistsError,
    TfidfConfigError,
    TfidfCorpusIdentityError,
)
from molfusion_backend.tfidf.ngrams import Ngram, document_ngram_counts

BUILD_REPORT_SCHEMA_VERSION = 1

# The Phase 5F-B corpus this artifact is defined against. Hard-coded rather
# than read from the build report beside the corpus: a report sitting next
# to a file cannot vouch for that file, and catching a corpus that is not
# the frozen one is the entire purpose of the check.
FROZEN_FIT_CORPUS_SHA256 = "b2c4b81160df05c95f8421582bb4b1c95fdf5964a4edaff24a7c1ddd43e2a5de"
FROZEN_DOCUMENT_COUNT = 2_897_639
FIT_CORPUS_NAME = "chembl37_canonical_smiles"
FIT_CORPUS_VERSION = "37"
FIT_CORPUS_SOURCE = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/"
    "chembl_37_sqlite.tar.gz"
)

# Volatile build-report fields, as (section, key) paths. Listed explicitly
# so "which fields may differ between two builds" is answerable by reading
# rather than by diffing.
VOLATILE_REPORT_PATHS = (("build", "built_at"),)


def verify_corpus_identity(corpus_path: Path, expected_sha256: str) -> str:
    """Hash the corpus and refuse to continue unless it is the frozen one."""
    if not corpus_path.is_file():
        raise TfidfCorpusIdentityError(f"Corpus not found: {corpus_path}")
    actual = sha256_file(corpus_path)
    if actual != expected_sha256:
        raise TfidfCorpusIdentityError(
            "Fit corpus identity mismatch -- refusing to fit against a different corpus.\n"
            f"  expected sha256: {expected_sha256}\n"
            f"  actual   sha256: {actual}\n  path: {corpus_path}"
        )
    return actual


def iter_corpus_documents(corpus_path: Path) -> Iterator[str]:
    """Yield each canonical SMILES, strictly decoded, LF-delimited.

    `newline="\\n"` disables universal-newline translation: the corpus
    contract says LF, so a stray CR is a corpus defect to surface rather
    than whitespace to absorb.
    """
    with corpus_path.open("r", encoding=CORPUS_ENCODING, errors="strict", newline="\n") as handle:
        for line in handle:
            yield line[:-1] if line.endswith("\n") else line


def count_document_frequencies(
    corpus_path: Path,
    *,
    orders: tuple[int, ...] = contract.NGRAM_ORDERS,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 250_000,
) -> tuple[dict[Ngram, int], int, dict[str, int]]:
    """Exact full-corpus document frequency per n-gram, in one streaming pass.

    Exact, never sketched: the n-gram vocabulary of canonical SMILES is
    thousands of entries, not millions, so approximation would trade
    auditability for memory the build does not need. Nothing per-molecule
    is retained -- each document's token tuple and n-gram set are consumed
    and dropped.
    """
    counts: dict[Ngram, int] = {}
    per_order: dict[str, int] = {str(order): 0 for order in orders}
    documents = 0

    for smiles in iter_corpus_documents(corpus_path):
        tokens = tokenize_smiles(smiles)
        for order in orders:
            for ngram in document_ngram_counts(tokens, order):
                # DF, not TF: one increment per document containing it,
                # however many times it occurs inside that document.
                if ngram in counts:
                    counts[ngram] += 1
                else:
                    counts[ngram] = 1
                    per_order[str(order)] += 1
        documents += 1
        if progress and progress_every and documents % progress_every == 0:
            progress(documents)

    return counts, documents, per_order


def _deterministic_json_bytes(payload: dict[str, Any]) -> bytes:
    text = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def build_artifact(
    corpus_path: Path,
    *,
    root: Path,
    artifact_type: str = contract.ARTIFACT_TYPE,
    artifact_id: str = contract.ARTIFACT_ID,
    artifact_version: str = contract.ARTIFACT_VERSION,
    expected_sha256: str = FROZEN_FIT_CORPUS_SHA256,
    expected_document_count: int | None = FROZEN_DOCUMENT_COUNT,
    min_df: int = contract.MIN_DF,
    max_features: int = contract.MAX_FEATURES,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 250_000,
) -> dict[str, Any]:
    """Build and finalize the artifact, returning its build report.

    Refuses to run if the destination version already exists: an audited
    artifact version is immutable, and a rebuild belongs in a temporary
    directory (see `rebuild_and_compare`).
    """
    corpus_path = Path(corpus_path)
    root = Path(root)
    destination = root / artifact_type / artifact_id / artifact_version
    if destination.exists():
        raise TfidfArtifactExistsError(
            f"Artifact version already exists and is immutable: {destination}. "
            "Build into a temporary root and compare payload digests instead of "
            "overwriting it."
        )

    verified_sha256 = verify_corpus_identity(corpus_path, expected_sha256)
    counts, documents, distinct_by_order = count_document_frequencies(
        corpus_path, orders=contract.NGRAM_ORDERS,
        progress=progress, progress_every=progress_every,
    )
    if expected_document_count is not None and documents != expected_document_count:
        raise TfidfCorpusIdentityError(
            "Fit corpus document count does not match the frozen corpus: "
            f"expected {expected_document_count:,}, read {documents:,}."
        )

    vocabulary, boundary = vocabulary_module.select_vocabulary(
        counts, min_df=min_df, max_features=max_features
    )
    vocabulary_module.validate_vocabulary(vocabulary, dimension=len(vocabulary.entries),
                                          min_df=min_df)
    idf_values = idf_module.compute_idf(vocabulary.document_frequencies(), documents)
    idf_module.validate_idf(
        idf_values,
        dimension=vocabulary.dimension,
        document_frequencies=vocabulary.document_frequencies(),
        n_documents=documents,
    )

    staging = destination.parent / f".staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True, exist_ok=False)
        payload_digests = _write_payloads(
            staging,
            vocabulary=vocabulary,
            idf_values=idf_values,
            documents=documents,
            boundary=boundary,
            distinct_by_order=distinct_by_order,
            verified_sha256=verified_sha256,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            min_df=min_df,
            max_features=max_features,
        )
        _validate_staged(
            staging,
            documents=documents,
            dimension=vocabulary.dimension,
            min_df=min_df,
            max_features=max_features,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return json.loads((destination / contract.BUILD_REPORT_FILENAME).read_text(encoding="utf-8"))


def _write_payloads(
    staging: Path,
    *,
    vocabulary: vocabulary_module.Vocabulary,
    idf_values: np.ndarray,
    documents: int,
    boundary: dict[str, Any],
    distinct_by_order: dict[str, int],
    verified_sha256: str,
    artifact_type: str,
    artifact_id: str,
    artifact_version: str,
    min_df: int,
    max_features: int,
) -> dict[str, str]:
    """Write payloads in dependency order, returning filename -> digest.

    The order is the checksum DAG: the three scientific payloads first,
    then the build report that records their digests, then the metadata
    that records all four. Nothing hashes itself.
    """
    digests: dict[str, str] = {}

    (staging / contract.VOCABULARY_FILENAME).write_bytes(
        vocabulary_module.vocabulary_bytes(vocabulary)
    )
    (staging / contract.IDF_FILENAME).write_bytes(idf_module.idf_bytes(idf_values))
    config = contract.frozen_config(
        fit_document_count=documents,
        eligible_terms_at_min_df=boundary["eligible_terms_at_min_df"],
        dimension=vocabulary.dimension,
        min_df=min_df,
        max_features=max_features,
    )
    (staging / contract.CONFIG_FILENAME).write_bytes(_deterministic_json_bytes(config))

    for filename in (
        contract.VOCABULARY_FILENAME,
        contract.IDF_FILENAME,
        contract.CONFIG_FILENAME,
    ):
        digests[filename] = sha256_file(staging / filename)

    report = _build_report(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        verified_sha256=verified_sha256,
        documents=documents,
        distinct_by_order=distinct_by_order,
        boundary=boundary,
        vocabulary=vocabulary,
        min_df=min_df,
        max_features=max_features,
        scientific_digests=dict(digests),
    )
    (staging / contract.BUILD_REPORT_FILENAME).write_bytes(_deterministic_json_bytes(report))
    digests[contract.BUILD_REPORT_FILENAME] = sha256_file(
        staging / contract.BUILD_REPORT_FILENAME
    )

    metadata = _metadata(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        verified_sha256=verified_sha256,
        documents=documents,
        digests=digests,
        config=config,
    )
    (staging / "metadata.json").write_bytes(_deterministic_json_bytes(metadata))
    return digests


def _build_report(**kw: Any) -> dict[str, Any]:
    vocabulary: vocabulary_module.Vocabulary = kw["vocabulary"]
    boundary = dict(kw["boundary"])
    return {
        "schema_version": BUILD_REPORT_SCHEMA_VERSION,
        "artifact": {
            "artifact_type": kw["artifact_type"],
            "artifact_id": kw["artifact_id"],
            "artifact_version": kw["artifact_version"],
        },
        "fit_corpus": {
            "name": FIT_CORPUS_NAME,
            "version": FIT_CORPUS_VERSION,
            "chembl_release": 37,
            "fit_corpus_sha256": kw["verified_sha256"],
            "document_count": kw["documents"],
            "normalization_id": CANONICAL_SMILES_NORMALIZATION_ID,
            "tokenizer_id": SMILES_TOKENIZER_ID,
            "serialization_id": CORPUS_SERIALIZATION_ID,
            "source": FIT_CORPUS_SOURCE,
            "uses_downstream_labels": False,
        },
        "vocabulary": {
            "ngram_orders": list(contract.NGRAM_ORDERS),
            "distinct_ngrams_by_order": kw["distinct_by_order"],
            "distinct_ngrams_total": boundary.pop("distinct_ngrams_total"),
            "eligible_terms_at_min_df": boundary.pop("eligible_terms_at_min_df"),
            "min_df": kw["min_df"],
            "max_features": kw["max_features"],
            "selected_dimension": vocabulary.dimension,
            "composition_by_order": vocabulary_module.composition_by_order(vocabulary),
            "document_frequency_min": min(vocabulary.document_frequencies()),
            "document_frequency_max": max(vocabulary.document_frequencies()),
            "selection_key": contract.SELECTION_KEY,
            "index_order": contract.INDEX_ORDER,
            "selection_boundary": boundary,
        },
        # The three scientific payloads only. This report is hashed after
        # it is written, and metadata.json records that digest -- so no
        # file is ever asked to contain its own checksum.
        "payload_sha256": kw["scientific_digests"],
        "build": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python": platform.python_version(),
                "rdkit": rdkit.__version__,
                "numpy": np.__version__,
                "sklearn": _sklearn_version(),
                "molfusion_git_commit": git_commit(Path(__file__).resolve().parent),
                # False means the named commit alone will not reproduce
                # this artifact, because the tree carried uncommitted
                # changes when it was built.
                "molfusion_git_working_tree_clean": working_tree_is_clean(
                    Path(__file__).resolve().parent
                ),
            },
        },
    }


def _sklearn_version() -> str | None:
    """Recorded for provenance only. sklearn is a test-time reference
    implementation and never participates in building or transforming."""
    try:
        import sklearn

        return sklearn.__version__
    except ImportError:
        return None


def _metadata(**kw: Any) -> dict[str, Any]:
    config = kw["config"]
    return {
        "artifact_id": kw["artifact_id"],
        "artifact_version": kw["artifact_version"],
        "artifact_type": kw["artifact_type"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "library_versions": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
        },
        # A pointer to the full contract, not a second copy of it:
        # tfidf_config.json is the authority and is checksummed.
        "configuration": {
            "ngram_range": [config["ngram_min"], config["ngram_max"]],
            "min_df": config["min_df"],
            "max_features": config["max_features"],
            "dimension": config["dimension"],
            "tf_mode": config["tf_mode"],
            "idf_mode": config["idf_mode"],
            "norm": config["norm"],
            "index_order": config["index_order"],
            "runtime_output_dtype": config["runtime_output_dtype"],
            "config_payload": contract.CONFIG_FILENAME,
        },
        "payload_files": [
            {"filename": filename, "sha256": digest}
            for filename, digest in sorted(kw["digests"].items())
        ],
        # The frozen logical corpus, not the ChEMBL archive or database:
        # this checksum is the scientific identity of what was fitted.
        "fit_corpus": {
            "name": FIT_CORPUS_NAME,
            "version": FIT_CORPUS_VERSION,
            "checksum": kw["verified_sha256"],
            "record_count": kw["documents"],
            "source": FIT_CORPUS_SOURCE,
        },
        # No randomness anywhere in this build: selection is a total order
        # and every formula is deterministic.
        "random_seed": None,
        "description": (
            "Frozen SMILES token n-gram TF-IDF representation fitted on the "
            "ChEMBL 37 reference corpus. Vocabulary selected by MolFusion, not "
            "by a vectorizer; sublinear TF, smoothed IDF, L2 norm."
        ),
    }


def _validate_staged(
    staging: Path, *, documents: int, dimension: int, min_df: int, max_features: int
) -> None:
    """Re-read everything from disk and validate it before finalizing.

    Deliberately reads the files rather than trusting the in-memory objects
    that produced them: the thing being shipped is the bytes, so the bytes
    are what gets checked.
    """
    payload = json.loads((staging / contract.VOCABULARY_FILENAME).read_text(encoding="utf-8"))
    vocabulary = vocabulary_module.parse_vocabulary(payload)
    vocabulary_module.validate_vocabulary(vocabulary, dimension=dimension, min_df=min_df)

    idf_module.validate_idf_payload(staging / contract.IDF_FILENAME, dimension=dimension)
    idf_values = idf_module.load_idf(staging / contract.IDF_FILENAME)
    idf_module.validate_idf(
        idf_values,
        dimension=dimension,
        document_frequencies=vocabulary.document_frequencies(),
        n_documents=documents,
    )

    config = contract.TfidfConfig.model_validate(
        json.loads((staging / contract.CONFIG_FILENAME).read_text(encoding="utf-8"))
    )
    mismatches = contract.contract_mismatches(
        config, min_df=min_df, max_features=max_features
    )
    if mismatches:
        raise TfidfConfigError(
            "staged config does not match the frozen contract: " + "; ".join(mismatches)
        )


def deterministic_report_view(report: dict[str, Any]) -> dict[str, Any]:
    """The build report minus its volatile fields, for comparing two builds."""
    trimmed = {
        name: dict(value) if isinstance(value, dict) else value
        for name, value in report.items()
    }
    for section, key in VOLATILE_REPORT_PATHS:
        if isinstance(trimmed.get(section), dict):
            trimmed[section] = {k: v for k, v in trimmed[section].items() if k != key}
    return trimmed


def rebuild_and_compare(
    corpus_path: Path,
    existing_root: Path,
    *,
    scratch_root: Path,
    artifact_type: str = contract.ARTIFACT_TYPE,
    artifact_id: str = contract.ARTIFACT_ID,
    artifact_version: str = contract.ARTIFACT_VERSION,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 250_000,
    **build_kwargs: Any,
) -> dict[str, Any]:
    """Rebuild into a scratch root and compare against an existing artifact.

    The safe alternative to a `--force` rebuild: the audited artifact is
    only ever read. Returns per-payload digest comparisons for the three
    scientific payloads, which are the ones required to be byte-identical;
    `build_report.json` and `metadata.json` carry timestamps and are
    expected to differ.
    """
    existing = Path(existing_root) / artifact_type / artifact_id / artifact_version
    if not existing.is_dir():
        raise TfidfArtifactExistsError(f"No existing artifact to compare against: {existing}")

    build_artifact(
        corpus_path,
        root=scratch_root,
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        progress=progress,
        progress_every=progress_every,
        **build_kwargs,
    )
    rebuilt = Path(scratch_root) / artifact_type / artifact_id / artifact_version

    comparisons = {}
    for filename in (
        contract.VOCABULARY_FILENAME,
        contract.IDF_FILENAME,
        contract.CONFIG_FILENAME,
    ):
        original_digest = sha256_file(existing / filename)
        rebuilt_digest = sha256_file(rebuilt / filename)
        comparisons[filename] = {
            "existing_sha256": original_digest,
            "rebuilt_sha256": rebuilt_digest,
            "identical": original_digest == rebuilt_digest,
        }

    existing_report = json.loads(
        (existing / contract.BUILD_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    rebuilt_report = json.loads(
        (rebuilt / contract.BUILD_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    return {
        "scientific_payloads": comparisons,
        "all_identical": all(entry["identical"] for entry in comparisons.values()),
        "build_report_deterministic_sections_match": (
            deterministic_report_view(existing_report) == deterministic_report_view(rebuilt_report)
        ),
    }


__all__ = [
    "BUILD_REPORT_SCHEMA_VERSION",
    "FROZEN_DOCUMENT_COUNT",
    "FROZEN_FIT_CORPUS_SHA256",
    "VOLATILE_REPORT_PATHS",
    "build_artifact",
    "count_document_frequencies",
    "deterministic_report_view",
    "iter_corpus_documents",
    "rebuild_and_compare",
    "verify_corpus_identity",
]
