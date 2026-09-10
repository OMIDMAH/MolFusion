from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from molfusion_backend.artifacts.root import validate_path_component

_SHA256_HEX_LENGTH = 64


class PayloadFile(BaseModel):
    """One artifact payload file and the checksum it must match.

    Filename and checksum travel together on the same entry (rather than as
    two parallel lists) so they can never drift out of sync with each other.
    """

    filename: str
    sha256: str

    @field_validator("filename")
    @classmethod
    def _filename_is_safe(cls, value: str) -> str:
        return validate_path_component(value, "payload_files[].filename")

    @field_validator("sha256")
    @classmethod
    def _sha256_is_well_formed(cls, value: str) -> str:
        lowered = value.lower()
        if len(lowered) != _SHA256_HEX_LENGTH or any(c not in "0123456789abcdef" for c in lowered):
            raise ValueError(f"sha256 must be {_SHA256_HEX_LENGTH} hex characters, got {value!r}")
        return lowered


class FitCorpus(BaseModel):
    """Provenance of the corpus a fitted representation was trained on.

    Only relevant for fitted/learned artifacts (e.g. a TF-IDF vectorizer);
    absent (fit_corpus=None on the parent) for artifacts with no fitting
    step. `checksum` and `record_count` are best-effort provenance, not
    re-verified by this infrastructure -- corpora are not artifact payloads.
    """

    name: str
    version: str
    checksum: str | None = None
    record_count: int | None = Field(default=None, ge=0)
    source: str | None = None


class ArtifactMetadata(BaseModel):
    """Validated contents of an artifact's `metadata.json`.

    Deliberately generic: this schema knows nothing about sklearn, PyTorch,
    Hugging Face, TF-IDF, or SELFIES. Deserializing `payload_files` into an
    actual object is a representation-specific consumer's responsibility,
    not this infrastructure's.
    """

    artifact_id: str
    artifact_version: str
    artifact_type: str
    created_at: datetime

    library_versions: dict[str, str] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)
    payload_files: list[PayloadFile] = Field(min_length=1)

    # Present only for fitted/learned representations; nullable for
    # artifacts with no fitting step.
    fit_corpus: FitCorpus | None = None
    # Nullable when no randomness was involved in producing the artifact.
    random_seed: int | None = None
    description: str | None = None

    @field_validator("artifact_id")
    @classmethod
    def _artifact_id_is_safe(cls, value: str) -> str:
        return validate_path_component(value, "artifact_id")

    @field_validator("artifact_version")
    @classmethod
    def _artifact_version_is_safe(cls, value: str) -> str:
        return validate_path_component(value, "artifact_version")

    @field_validator("artifact_type")
    @classmethod
    def _artifact_type_is_safe(cls, value: str) -> str:
        return validate_path_component(value, "artifact_type")

    @model_validator(mode="after")
    def _payload_filenames_are_unique(self) -> "ArtifactMetadata":
        filenames = [payload.filename for payload in self.payload_files]
        if len(filenames) != len(set(filenames)):
            raise ValueError(f"payload_files contains duplicate filenames: {filenames}")
        return self
