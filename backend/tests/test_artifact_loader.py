import hashlib
import json

import pytest

from molfusion_backend.artifacts import (
    ArtifactChecksumError,
    ArtifactMetadataError,
    ArtifactNotFoundError,
    load_artifact,
)

ARTIFACT_TYPE = "tfidf"
ARTIFACT_ID = "pubchem_smiles_tfidf"
ARTIFACT_VERSION = "1.0.0"


def _write_artifact(root, artifact_type, artifact_id, artifact_version, payloads, metadata_overrides=None):
    """Write a valid artifact directory (metadata.json + payload files) under `root`.

    `payloads` maps filename -> raw bytes; each payload's checksum is
    computed from those exact bytes, so the artifact is valid unless a
    test corrupts something afterward. `metadata_overrides` replaces
    entries in the otherwise-consistent metadata.json (used to deliberately
    create an identity mismatch between the directory and its metadata).
    """
    directory = root / artifact_type / artifact_id / artifact_version
    directory.mkdir(parents=True)

    payload_files = []
    for filename, content in payloads.items():
        (directory / filename).write_bytes(content)
        payload_files.append({"filename": filename, "sha256": hashlib.sha256(content).hexdigest()})

    metadata = {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "artifact_type": artifact_type,
        "created_at": "2026-01-01T00:00:00Z",
        "payload_files": payload_files,
        **(metadata_overrides or {}),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def test_valid_artifact_loads_successfully(tmp_path):
    _write_artifact(
        tmp_path, ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, {"vectorizer.joblib": b"fake fitted model bytes"}
    )

    descriptor = load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)

    assert descriptor.artifact_type == ARTIFACT_TYPE
    assert descriptor.artifact_id == ARTIFACT_ID
    assert descriptor.artifact_version == ARTIFACT_VERSION
    assert descriptor.metadata.artifact_id == ARTIFACT_ID
    assert descriptor.directory == (tmp_path / ARTIFACT_TYPE / ARTIFACT_ID / ARTIFACT_VERSION).resolve()
    assert set(descriptor.payload_paths) == {"vectorizer.joblib"}
    assert descriptor.payload_paths["vectorizer.joblib"].read_bytes() == b"fake fitted model bytes"


def test_checksum_failure_after_payload_is_modified_raises(tmp_path):
    directory = _write_artifact(
        tmp_path, ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, {"vectorizer.joblib": b"original bytes"}
    )
    (directory / "vectorizer.joblib").write_bytes(b"tampered bytes, different length even")

    with pytest.raises(ArtifactChecksumError):
        load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)


def test_missing_payload_file_raises_not_found(tmp_path):
    directory = _write_artifact(
        tmp_path, ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, {"vectorizer.joblib": b"bytes"}
    )
    (directory / "vectorizer.joblib").unlink()

    with pytest.raises(ArtifactNotFoundError):
        load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)


def test_missing_artifact_directory_raises_not_found(tmp_path):
    with pytest.raises(ArtifactNotFoundError):
        load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)


def test_missing_metadata_file_raises_metadata_error(tmp_path):
    directory = tmp_path / ARTIFACT_TYPE / ARTIFACT_ID / ARTIFACT_VERSION
    directory.mkdir(parents=True)
    (directory / "vectorizer.joblib").write_bytes(b"bytes, but no metadata.json alongside it")

    with pytest.raises(ArtifactMetadataError, match="Missing metadata.json"):
        load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)


def test_malformed_json_metadata_raises_metadata_error(tmp_path):
    directory = tmp_path / ARTIFACT_TYPE / ARTIFACT_ID / ARTIFACT_VERSION
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ArtifactMetadataError, match="not valid JSON"):
        load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)


def test_metadata_failing_schema_validation_raises_metadata_error(tmp_path):
    directory = tmp_path / ARTIFACT_TYPE / ARTIFACT_ID / ARTIFACT_VERSION
    directory.mkdir(parents=True)
    # Valid JSON, but missing every required field.
    (directory / "metadata.json").write_text(json.dumps({"description": "incomplete"}), encoding="utf-8")

    with pytest.raises(ArtifactMetadataError, match="schema validation"):
        load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)


def test_artifact_id_mismatch_between_request_and_metadata_raises(tmp_path):
    _write_artifact(
        tmp_path,
        ARTIFACT_TYPE,
        ARTIFACT_ID,
        ARTIFACT_VERSION,
        {"vectorizer.joblib": b"bytes"},
        metadata_overrides={"artifact_id": "a_different_id"},
    )

    with pytest.raises(ArtifactMetadataError, match="artifact_id"):
        load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)


def test_artifact_version_mismatch_between_request_and_metadata_raises(tmp_path):
    _write_artifact(
        tmp_path,
        ARTIFACT_TYPE,
        ARTIFACT_ID,
        ARTIFACT_VERSION,
        {"vectorizer.joblib": b"bytes"},
        metadata_overrides={"artifact_version": "9.9.9"},
    )

    with pytest.raises(ArtifactMetadataError, match="artifact_version"):
        load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)


def test_artifact_type_mismatch_between_request_and_metadata_raises(tmp_path):
    _write_artifact(
        tmp_path,
        ARTIFACT_TYPE,
        ARTIFACT_ID,
        ARTIFACT_VERSION,
        {"vectorizer.joblib": b"bytes"},
        metadata_overrides={"artifact_type": "a_different_type"},
    )

    with pytest.raises(ArtifactMetadataError, match="artifact_type"):
        load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)


@pytest.mark.parametrize(
    "artifact_type,artifact_id,artifact_version",
    [
        ("../escape", ARTIFACT_ID, ARTIFACT_VERSION),
        (ARTIFACT_TYPE, "../../escape", ARTIFACT_VERSION),
        (ARTIFACT_TYPE, ARTIFACT_ID, "../escape"),
        (ARTIFACT_TYPE, "a/b", ARTIFACT_VERSION),
        (ARTIFACT_TYPE, "a\\b", ARTIFACT_VERSION),
    ],
)
def test_path_traversal_attempts_cannot_escape_artifact_root(
    tmp_path, artifact_type, artifact_id, artifact_version
):
    # A real artifact exists outside the root, at a location a traversal
    # attempt might target -- the loader must never resolve to it.
    outside_target = tmp_path.parent / "escape_target_metadata.json"
    outside_target.write_text("{}", encoding="utf-8")
    root = tmp_path / "artifact_root"
    root.mkdir()

    try:
        with pytest.raises(ArtifactNotFoundError):
            load_artifact(artifact_type, artifact_id, artifact_version, root=root)
    finally:
        outside_target.unlink()


def test_multiple_payloads_are_all_checksum_verified(tmp_path):
    _write_artifact(
        tmp_path,
        ARTIFACT_TYPE,
        ARTIFACT_ID,
        ARTIFACT_VERSION,
        {"vectorizer.joblib": b"vectorizer bytes", "vocabulary.json": b'{"a": 0, "b": 1}'},
    )

    descriptor = load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)

    assert set(descriptor.payload_paths) == {"vectorizer.joblib", "vocabulary.json"}


def test_multiple_payloads_any_single_corruption_is_caught(tmp_path):
    directory = _write_artifact(
        tmp_path,
        ARTIFACT_TYPE,
        ARTIFACT_ID,
        ARTIFACT_VERSION,
        {"vectorizer.joblib": b"vectorizer bytes", "vocabulary.json": b'{"a": 0, "b": 1}'},
    )
    (directory / "vocabulary.json").write_bytes(b'{"a": 0, "b": 999}')

    with pytest.raises(ArtifactChecksumError, match="vocabulary.json"):
        load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)


def test_empty_payload_file_checksums_correctly(tmp_path):
    _write_artifact(tmp_path, ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, {"empty.bin": b""})

    descriptor = load_artifact(ARTIFACT_TYPE, ARTIFACT_ID, ARTIFACT_VERSION, root=tmp_path)

    assert descriptor.payload_paths["empty.bin"].read_bytes() == b""


def test_loader_uses_pathlib_semantics_not_os_specific_strings(tmp_path):
    # Exercises the loader with nested pathlib.Path segments only -- no
    # manual string concatenation of path separators anywhere in the call.
    nested_id = "vendor_a"
    _write_artifact(tmp_path, ARTIFACT_TYPE, nested_id, ARTIFACT_VERSION, {"payload.bin": b"data"})

    descriptor = load_artifact(ARTIFACT_TYPE, nested_id, ARTIFACT_VERSION, root=tmp_path)

    expected_directory = (tmp_path / ARTIFACT_TYPE / nested_id / ARTIFACT_VERSION).resolve()
    assert descriptor.directory == expected_directory
