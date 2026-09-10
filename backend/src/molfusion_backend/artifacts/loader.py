import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from molfusion_backend.artifacts.checksum import verify_payload_checksum
from molfusion_backend.artifacts.errors import ArtifactMetadataError, ArtifactNotFoundError
from molfusion_backend.artifacts.metadata import ArtifactMetadata
from molfusion_backend.artifacts.root import resolve_artifact_root, validate_path_component

_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class ArtifactDescriptor:
    """A resolved, validated, checksum-verified artifact.

    Hands a representation-specific consumer everything it needs to
    deserialize the payload itself (e.g. `joblib.load(payload_paths["vectorizer.joblib"])`)
    -- this infrastructure never interprets payload contents.
    """

    artifact_type: str
    artifact_id: str
    artifact_version: str
    directory: Path
    metadata: ArtifactMetadata
    payload_paths: dict[str, Path]


def load_artifact(
    artifact_type: str,
    artifact_id: str,
    artifact_version: str,
    *,
    root: Path | None = None,
) -> ArtifactDescriptor:
    """Resolve, validate, and checksum-verify one artifact, or raise.

    Load-or-fail: never returns a partially validated artifact. Any problem
    (missing directory, missing/malformed metadata, identity mismatch,
    missing payload file, checksum mismatch) raises before returning.
    """
    resolved_root = (root if root is not None else resolve_artifact_root()).resolve()

    directory = _resolve_artifact_directory(resolved_root, artifact_type, artifact_id, artifact_version)
    if not directory.is_dir():
        raise ArtifactNotFoundError(
            f"No artifact at {artifact_type}/{artifact_id}/{artifact_version} "
            f"under root {resolved_root}"
        )

    metadata = _load_metadata(directory)
    _check_identity_matches(metadata, artifact_type, artifact_id, artifact_version)

    payload_paths: dict[str, Path] = {}
    for payload in metadata.payload_files:
        payload_path = directory / payload.filename
        if not payload_path.is_file():
            raise ArtifactNotFoundError(
                f"Payload file declared in metadata.json is missing: {payload_path}"
            )
        verify_payload_checksum(payload_path, payload.sha256)
        payload_paths[payload.filename] = payload_path

    return ArtifactDescriptor(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        directory=directory,
        metadata=metadata,
        payload_paths=payload_paths,
    )


def _resolve_artifact_directory(
    root: Path, artifact_type: str, artifact_id: str, artifact_version: str
) -> Path:
    # Reject unsafe components (path separators, "..", etc.) before they
    # ever touch the filesystem, as ArtifactNotFoundError rather than a
    # distinct "bad request" error: from the caller's perspective, an
    # unsafe/malicious location and a genuinely absent one both simply
    # don't resolve to a valid artifact.
    try:
        for value, field_name in (
            (artifact_type, "artifact_type"),
            (artifact_id, "artifact_id"),
            (artifact_version, "artifact_version"),
        ):
            validate_path_component(value, field_name)
    except ValueError as exc:
        raise ArtifactNotFoundError(str(exc)) from exc

    candidate = (root / artifact_type / artifact_id / artifact_version).resolve()
    # Defense in depth: even if a future component check is loosened,
    # never resolve to a path outside the artifact root.
    if not candidate.is_relative_to(root):
        raise ArtifactNotFoundError(f"Resolved artifact path escapes the artifact root: {candidate}")
    return candidate


def _load_metadata(directory: Path) -> ArtifactMetadata:
    metadata_path = directory / _METADATA_FILENAME
    if not metadata_path.is_file():
        raise ArtifactMetadataError(f"Missing {_METADATA_FILENAME} in {directory}")

    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ArtifactMetadataError(f"Could not read {metadata_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactMetadataError(f"{metadata_path} is not valid JSON: {exc}") from exc

    try:
        return ArtifactMetadata.model_validate(raw)
    except ValidationError as exc:
        raise ArtifactMetadataError(f"{metadata_path} failed schema validation: {exc}") from exc


def _check_identity_matches(
    metadata: ArtifactMetadata, artifact_type: str, artifact_id: str, artifact_version: str
) -> None:
    mismatches = []
    if metadata.artifact_type != artifact_type:
        mismatches.append(
            f"artifact_type: requested {artifact_type!r}, metadata has {metadata.artifact_type!r}"
        )
    if metadata.artifact_id != artifact_id:
        mismatches.append(
            f"artifact_id: requested {artifact_id!r}, metadata has {metadata.artifact_id!r}"
        )
    if metadata.artifact_version != artifact_version:
        mismatches.append(
            f"artifact_version: requested {artifact_version!r}, metadata has {metadata.artifact_version!r}"
        )
    if mismatches:
        raise ArtifactMetadataError(
            "Requested artifact identity does not match metadata.json: " + "; ".join(mismatches)
        )
