import json

import numpy as np
import pytest

from molfusion_backend.artifacts import load_artifact, sha256_file
from molfusion_backend.artifacts.errors import ArtifactChecksumError
from molfusion_backend.corpus.serialization import write_corpus
from molfusion_backend.tfidf import builder, contract
from molfusion_backend.tfidf.errors import (
    TfidfArtifactExistsError,
    TfidfConfigError,
    TfidfCorpusIdentityError,
    TfidfIdfError,
    TfidfVocabularyError,
)
from molfusion_backend.tfidf.loader import load_tfidf_artifact

FIXTURE_SMILES = sorted(
    {"CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "N", "O"}
    | {f"C{'C' * index}O" for index in range(1, 300)}
    | {f"c1ccccc1{'C' * index}" for index in range(1, 200)}
)
FIXTURE_MIN_DF = 5
FIXTURE_MAX_FEATURES = 64


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "canonical_smiles.smi"
    sha256, _ = write_corpus(path, FIXTURE_SMILES)
    return path, sha256


def build(corpus, root, **kwargs):
    path, sha256 = corpus
    kwargs.setdefault("expected_sha256", sha256)
    kwargs.setdefault("expected_document_count", len(FIXTURE_SMILES))
    kwargs.setdefault("min_df", FIXTURE_MIN_DF)
    kwargs.setdefault("max_features", FIXTURE_MAX_FEATURES)
    kwargs.setdefault("progress_every", 0)
    return builder.build_artifact(path, root=root, **kwargs)


def load(root, **kwargs):
    kwargs.setdefault("root", root)
    kwargs.setdefault("expected_max_features", FIXTURE_MAX_FEATURES)
    return load_tfidf_artifact(**kwargs)


def artifact_dir(root):
    return root / contract.ARTIFACT_TYPE / contract.ARTIFACT_ID / contract.ARTIFACT_VERSION


# ---------------------------------------------------------------------------
# corpus identity
# ---------------------------------------------------------------------------


def test_a_wrong_corpus_digest_aborts_before_any_counting(corpus, tmp_path):
    path, _ = corpus
    root = tmp_path / "artifacts"
    with pytest.raises(TfidfCorpusIdentityError, match="identity mismatch"):
        builder.build_artifact(
            path, root=root, expected_sha256="0" * 64, progress_every=0
        )
    assert not root.exists()


def test_a_missing_corpus_is_reported_as_an_identity_failure(tmp_path):
    with pytest.raises(TfidfCorpusIdentityError, match="not found"):
        builder.build_artifact(
            tmp_path / "absent.smi", root=tmp_path / "artifacts", progress_every=0
        )


def test_a_wrong_document_count_aborts(corpus, tmp_path):
    with pytest.raises(TfidfCorpusIdentityError, match="document count"):
        build(corpus, tmp_path / "artifacts", expected_document_count=len(FIXTURE_SMILES) + 1)


def test_the_frozen_corpus_identity_is_pinned():
    assert (
        builder.FROZEN_FIT_CORPUS_SHA256
        == "b2c4b81160df05c95f8421582bb4b1c95fdf5964a4edaff24a7c1ddd43e2a5de"
    )
    assert builder.FROZEN_DOCUMENT_COUNT == 2_897_639


def test_the_build_never_modifies_the_corpus(corpus, tmp_path):
    path, sha256 = corpus
    before = path.read_bytes()
    build(corpus, tmp_path / "artifacts")
    assert path.read_bytes() == before
    assert sha256_file(path) == sha256


# ---------------------------------------------------------------------------
# atomicity and immutability
# ---------------------------------------------------------------------------


def test_an_existing_version_is_never_overwritten(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    before = sha256_file(artifact_dir(root) / contract.VOCABULARY_FILENAME)
    with pytest.raises(TfidfArtifactExistsError, match="immutable"):
        build(corpus, root)
    assert sha256_file(artifact_dir(root) / contract.VOCABULARY_FILENAME) == before


def test_a_failed_build_leaves_no_artifact_and_no_staging(corpus, tmp_path):
    """The corpus is fine but the document count is wrong, so the failure
    happens after the identity gate and before any payload is written."""
    root = tmp_path / "artifacts"
    with pytest.raises(TfidfCorpusIdentityError):
        build(corpus, root, expected_document_count=1)
    assert not artifact_dir(root).exists()
    if root.exists():
        assert not list(root.rglob(".staging-*"))


def test_no_staging_directory_survives_a_successful_build(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    assert not list(root.rglob(".staging-*"))


# ---------------------------------------------------------------------------
# payloads and the checksum DAG
# ---------------------------------------------------------------------------


def test_all_payloads_are_written(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    assert {p.name for p in artifact_dir(root).iterdir()} == {
        "metadata.json",
        contract.VOCABULARY_FILENAME,
        contract.IDF_FILENAME,
        contract.CONFIG_FILENAME,
        contract.BUILD_REPORT_FILENAME,
    }


def test_metadata_checksums_every_payload_and_they_all_verify(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    directory = artifact_dir(root)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    declared = {entry["filename"]: entry["sha256"] for entry in metadata["payload_files"]}
    assert set(declared) == {
        contract.VOCABULARY_FILENAME,
        contract.IDF_FILENAME,
        contract.CONFIG_FILENAME,
        contract.BUILD_REPORT_FILENAME,
    }
    for filename, digest in declared.items():
        assert sha256_file(directory / filename) == digest


def test_the_build_report_does_not_contain_its_own_checksum(corpus, tmp_path):
    """The cycle the brief warns about: build_report.json records the three
    scientific payloads and is itself hashed afterwards by metadata.json."""
    root = tmp_path / "artifacts"
    report = build(corpus, root)
    assert set(report["payload_sha256"]) == {
        contract.VOCABULARY_FILENAME,
        contract.IDF_FILENAME,
        contract.CONFIG_FILENAME,
    }
    assert contract.BUILD_REPORT_FILENAME not in report["payload_sha256"]

    directory = artifact_dir(root)
    for filename, digest in report["payload_sha256"].items():
        assert sha256_file(directory / filename) == digest


def test_metadata_is_not_checksummed_by_anything(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    metadata = json.loads((artifact_dir(root) / "metadata.json").read_text(encoding="utf-8"))
    assert "metadata.json" not in {e["filename"] for e in metadata["payload_files"]}


def test_fit_corpus_records_the_logical_corpus_not_the_archive(corpus, tmp_path):
    _, sha256 = corpus
    root = tmp_path / "artifacts"
    build(corpus, root)
    metadata = json.loads((artifact_dir(root) / "metadata.json").read_text(encoding="utf-8"))
    fit_corpus = metadata["fit_corpus"]
    assert fit_corpus["checksum"] == sha256
    assert fit_corpus["record_count"] == len(FIXTURE_SMILES)
    assert fit_corpus["name"] == builder.FIT_CORPUS_NAME


def test_payload_json_is_utf8_lf_with_a_final_newline(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    for filename in (
        "metadata.json",
        contract.VOCABULARY_FILENAME,
        contract.CONFIG_FILENAME,
        contract.BUILD_REPORT_FILENAME,
    ):
        raw = (artifact_dir(root) / filename).read_bytes()
        assert raw.endswith(b"\n")
        assert b"\r\n" not in raw


def test_build_report_records_selection_boundary_evidence(corpus, tmp_path):
    report = build(corpus, tmp_path / "artifacts")
    boundary = report["vocabulary"]["selection_boundary"]
    assert "boundary_document_frequency" in boundary
    assert "terms_tied_at_boundary_df" in boundary
    assert "tie_resolution" in boundary
    assert report["vocabulary"]["min_df"] == FIXTURE_MIN_DF
    assert report["vocabulary"]["max_features"] == FIXTURE_MAX_FEATURES
    assert report["build"]["software"]["numpy"]


def test_config_records_the_parameters_the_build_actually_used(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    config = json.loads((artifact_dir(root) / contract.CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert config["min_df"] == FIXTURE_MIN_DF
    assert config["max_features"] == FIXTURE_MAX_FEATURES
    assert config["tf_mode"] == "sublinear"
    assert config["idf_mode"] == "smoothed"
    assert config["smooth_idf"] is True
    assert config["norm"] == "l2"
    assert config["idf_dtype"] == "float64"
    assert config["runtime_output_dtype"] == "float32"
    assert config["index_order"] == contract.INDEX_ORDER
    assert "UNK" in config["oov_policy"]
    assert "not a representation failure" in config["zero_vector_policy"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_the_generic_loader_accepts_the_artifact(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    descriptor = load_artifact(
        contract.ARTIFACT_TYPE, contract.ARTIFACT_ID, contract.ARTIFACT_VERSION, root=root
    )
    assert set(descriptor.payload_paths) == {
        contract.VOCABULARY_FILENAME,
        contract.IDF_FILENAME,
        contract.CONFIG_FILENAME,
        contract.BUILD_REPORT_FILENAME,
    }
    assert descriptor.metadata.fit_corpus is not None


def test_the_tfidf_loader_validates_semantics(corpus, tmp_path):
    _, sha256 = corpus
    root = tmp_path / "artifacts"
    build(corpus, root)
    artifact = load(root, expected_fit_corpus_sha256=sha256)

    assert artifact.dimension == artifact.config.dimension
    assert artifact.idf.shape == (artifact.dimension,)
    assert artifact.idf.dtype == np.float64
    assert artifact.fit_corpus_sha256 == sha256
    assert len(artifact.feature_names()) == artifact.dimension


def test_the_loader_produces_a_working_transformer(corpus, tmp_path):
    from molfusion_backend.smiles_tokenizer import tokenize_smiles

    root = tmp_path / "artifacts"
    build(corpus, root)
    transformer = load(root).transformer()

    vector = transformer.transform(tokenize_smiles("CCO"))
    assert vector.shape == (transformer.dimension,)
    assert vector.dtype == np.float32
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, rel=1e-6)


def test_the_loader_rejects_a_corpus_it_was_not_fitted_on(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    with pytest.raises(TfidfCorpusIdentityError, match="different corpus"):
        load(root, expected_fit_corpus_sha256="0" * 64)


def test_the_loader_rejects_a_config_that_departs_from_the_contract(corpus, tmp_path):
    """Loading with the production `max_features` must refuse a fixture
    artifact -- the contract check is live, not decorative."""
    root = tmp_path / "artifacts"
    build(corpus, root)
    with pytest.raises(TfidfConfigError, match="frozen contract"):
        load_tfidf_artifact(root=root)


# ---------------------------------------------------------------------------
# corruption is caught
# ---------------------------------------------------------------------------


def _rechecksum(directory, filename):
    """Update metadata so a tampered payload passes the checksum gate.

    Needed to prove the *semantic* validation catches things a checksum
    cannot: a payload rebuilt with a wrong formula has a perfectly valid
    digest.
    """
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for entry in metadata["payload_files"]:
        if entry["filename"] == filename:
            entry["sha256"] = sha256_file(directory / filename)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def test_a_tampered_payload_fails_the_checksum_gate(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    path = artifact_dir(root) / contract.VOCABULARY_FILENAME
    path.write_bytes(path.read_bytes().replace(b'"index": 0', b'"index": 1', 1))
    with pytest.raises(ArtifactChecksumError):
        load(root)


def test_a_corrupt_vocabulary_is_rejected_semantically(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    directory = artifact_dir(root)
    path = directory / contract.VOCABULARY_FILENAME

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["tokens"] = ["zzz"]  # breaks lexicographic order
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _rechecksum(directory, contract.VOCABULARY_FILENAME)

    with pytest.raises(TfidfVocabularyError, match="lexicographic"):
        load(root)


def test_a_vocabulary_dimension_mismatch_is_rejected(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    directory = artifact_dir(root)
    path = directory / contract.VOCABULARY_FILENAME

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"] = payload["entries"][:-1]
    payload["dimension"] = len(payload["entries"])
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _rechecksum(directory, contract.VOCABULARY_FILENAME)

    with pytest.raises(TfidfVocabularyError, match="entries, expected"):
        load(root)


def test_a_corrupt_idf_is_rejected_semantically(corpus, tmp_path):
    """Checksum-valid but formula-wrong: only re-deriving from the recorded
    document frequencies catches this."""
    root = tmp_path / "artifacts"
    build(corpus, root)
    directory = artifact_dir(root)

    from molfusion_backend.tfidf.idf import idf_bytes, load_idf

    values = load_idf(directory / contract.IDF_FILENAME)
    values[0] += 0.5
    (directory / contract.IDF_FILENAME).write_bytes(idf_bytes(values))
    _rechecksum(directory, contract.IDF_FILENAME)

    with pytest.raises(TfidfIdfError, match="does not reproduce"):
        load(root)


def test_an_idf_of_the_wrong_dtype_is_rejected(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    directory = artifact_dir(root)

    from io import BytesIO

    from molfusion_backend.tfidf.idf import load_idf

    values = load_idf(directory / contract.IDF_FILENAME).astype(np.float32)
    buffer = BytesIO()
    np.save(buffer, values, allow_pickle=False)
    (directory / contract.IDF_FILENAME).write_bytes(buffer.getvalue())
    _rechecksum(directory, contract.IDF_FILENAME)

    with pytest.raises(TfidfIdfError, match="dtype"):
        load(root)


def test_an_idf_of_the_wrong_length_is_rejected(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    directory = artifact_dir(root)

    from molfusion_backend.tfidf.idf import idf_bytes, load_idf

    values = load_idf(directory / contract.IDF_FILENAME)[:-1]
    (directory / contract.IDF_FILENAME).write_bytes(idf_bytes(values))
    _rechecksum(directory, contract.IDF_FILENAME)

    with pytest.raises(TfidfIdfError, match="shape"):
        load(root)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_two_builds_produce_byte_identical_scientific_payloads(corpus, tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    build(corpus, first)
    build(corpus, second)

    for filename in (
        contract.VOCABULARY_FILENAME,
        contract.IDF_FILENAME,
        contract.CONFIG_FILENAME,
    ):
        assert (artifact_dir(first) / filename).read_bytes() == (
            artifact_dir(second) / filename
        ).read_bytes()
        assert sha256_file(artifact_dir(first) / filename) == sha256_file(
            artifact_dir(second) / filename
        )


def test_build_reports_agree_apart_from_the_timestamp(corpus, tmp_path):
    first = build(corpus, tmp_path / "one")
    second = build(corpus, tmp_path / "two")

    # Every scientific section is identical; only `built_at` may differ,
    # and it is present in both rather than quietly absent.
    assert builder.deterministic_report_view(first) == builder.deterministic_report_view(second)
    assert first["build"]["built_at"]
    assert second["build"]["built_at"]
    assert first["payload_sha256"] == second["payload_sha256"]
    assert first["vocabulary"] == second["vocabulary"]


def test_only_the_timestamp_is_excused_from_determinism():
    assert builder.VOLATILE_REPORT_PATHS == (("build", "built_at"),)


def test_rebuild_and_compare_confirms_reproducibility(corpus, tmp_path):
    root = tmp_path / "artifacts"
    build(corpus, root)
    path, sha256 = corpus
    result = builder.rebuild_and_compare(
        path,
        root,
        scratch_root=tmp_path / "scratch",
        expected_sha256=sha256,
        expected_document_count=len(FIXTURE_SMILES),
        min_df=FIXTURE_MIN_DF,
        max_features=FIXTURE_MAX_FEATURES,
        progress_every=0,
    )
    assert result["all_identical"] is True
    assert result["build_report_deterministic_sections_match"] is True
    assert set(result["scientific_payloads"]) == {
        contract.VOCABULARY_FILENAME,
        contract.IDF_FILENAME,
        contract.CONFIG_FILENAME,
    }
    # The audited artifact was only read.
    assert artifact_dir(root).is_dir()


def test_rebuild_and_compare_needs_something_to_compare_against(corpus, tmp_path):
    path, _ = corpus
    with pytest.raises(TfidfArtifactExistsError, match="No existing artifact"):
        builder.rebuild_and_compare(
            path, tmp_path / "empty", scratch_root=tmp_path / "scratch", progress_every=0
        )


def test_build_report_records_whether_the_tree_was_clean(corpus, tmp_path):
    """A commit alone is not provenance if the tree was dirty when the
    artifact was built: the named revision would not reproduce it, and
    nothing in the artifact would say so."""
    report = build(corpus, tmp_path / "artifacts")
    software = report["build"]["software"]
    assert "molfusion_git_working_tree_clean" in software
    assert software["molfusion_git_working_tree_clean"] in (True, False, None)
    assert "molfusion_git_commit" in software


def test_the_build_report_never_records_the_metadata_digest(corpus, tmp_path):
    """The second half of the acyclicity rule.

    metadata.json owns the checksums of all four payloads, including the
    build report's. If the build report also recorded metadata.json's
    digest the two would depend on each other and neither could be written
    second. The metadata digest is therefore never computed at all, and
    the report does not mention the file anywhere -- checked over the whole
    body rather than just the digest map, so a future field cannot
    reintroduce the cycle somewhere else in the document.
    """
    root = tmp_path / "artifacts"
    report = build(corpus, root)
    directory = artifact_dir(root)

    assert "metadata.json" not in report["payload_sha256"]
    assert "build_report.json" not in report["payload_sha256"]

    body = (directory / contract.BUILD_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "metadata.json" not in body
    assert sha256_file(directory / "metadata.json") not in body
    assert sha256_file(directory / contract.BUILD_REPORT_FILENAME) not in body


def test_metadata_owns_the_checksum_of_every_payload_including_the_report(corpus, tmp_path):
    """The first half: metadata is the single owner of payload integrity."""
    root = tmp_path / "artifacts"
    build(corpus, root)
    directory = artifact_dir(root)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    owned = {entry["filename"]: entry["sha256"] for entry in metadata["payload_files"]}

    assert set(owned) == {
        contract.VOCABULARY_FILENAME,
        contract.IDF_FILENAME,
        contract.CONFIG_FILENAME,
        contract.BUILD_REPORT_FILENAME,
    }
    for filename, digest in owned.items():
        assert sha256_file(directory / filename) == digest
