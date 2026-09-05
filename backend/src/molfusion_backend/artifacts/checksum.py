import hashlib
from pathlib import Path

from molfusion_backend.artifacts.errors import ArtifactChecksumError, ArtifactNotFoundError

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of `path`, read in chunks.

    Binary-safe and deterministic: reads raw bytes only, in fixed-size
    chunks, so it works identically for text and binary payloads of any size.
    """
    if not path.is_file():
        raise ArtifactNotFoundError(f"Cannot checksum missing file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_payload_checksum(path: Path, expected_sha256: str) -> None:
    """Raise unless `path` exists and its SHA-256 matches `expected_sha256`."""
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ArtifactChecksumError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
