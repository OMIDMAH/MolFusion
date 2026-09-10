class ArtifactError(Exception):
    """Base class for all artifact-infrastructure errors."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when an artifact directory or a declared payload file is missing."""


class ArtifactMetadataError(ArtifactError):
    """Raised when metadata.json is missing, malformed, fails schema
    validation, or does not match the requested artifact_type/id/version."""


class ArtifactChecksumError(ArtifactError):
    """Raised when a payload file's SHA-256 does not match its declared checksum."""
