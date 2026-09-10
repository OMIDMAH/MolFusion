import hashlib
import re

import pytest

from molfusion_backend.artifacts import (
    ArtifactChecksumError,
    ArtifactNotFoundError,
    sha256_file,
    verify_payload_checksum,
)


def test_sha256_file_matches_hashlib_reference(tmp_path):
    content = b"the quick brown fox jumps over the lazy dog"
    path = tmp_path / "payload.bin"
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_sha256_file_is_binary_safe(tmp_path):
    content = bytes(range(256)) * 100
    path = tmp_path / "binary.bin"
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_sha256_file_handles_content_larger_than_one_chunk(tmp_path):
    # Larger than the loader's internal chunk size, to exercise the
    # streaming/chunked read path rather than a single read() call.
    content = b"x" * (3 * 1024 * 1024 + 17)
    path = tmp_path / "large.bin"
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_sha256_file_of_empty_file_matches_hashlib_empty_digest(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()


def test_sha256_file_missing_raises_not_found(tmp_path):
    with pytest.raises(ArtifactNotFoundError, match="missing"):
        sha256_file(tmp_path / "does_not_exist.bin")


def test_verify_payload_checksum_passes_for_matching_checksum(tmp_path):
    content = b"payload contents"
    path = tmp_path / "payload.bin"
    path.write_bytes(content)

    verify_payload_checksum(path, hashlib.sha256(content).hexdigest())


def test_verify_payload_checksum_is_case_insensitive(tmp_path):
    content = b"payload contents"
    path = tmp_path / "payload.bin"
    path.write_bytes(content)

    verify_payload_checksum(path, hashlib.sha256(content).hexdigest().upper())


def test_verify_payload_checksum_raises_clear_mismatch_error(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"actual contents")
    wrong_checksum = hashlib.sha256(b"different contents").hexdigest()

    with pytest.raises(ArtifactChecksumError, match=re.escape(str(path))):
        verify_payload_checksum(path, wrong_checksum)


def test_verify_payload_checksum_raises_clear_missing_file_error(tmp_path):
    missing = tmp_path / "missing.bin"

    with pytest.raises(ArtifactNotFoundError, match=re.escape(str(missing))):
        verify_payload_checksum(missing, "0" * 64)
