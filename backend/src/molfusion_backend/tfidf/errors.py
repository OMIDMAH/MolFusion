"""Failures specific to the SMILES TF-IDF representation.

These subclass the generic `ArtifactError` so a caller that already
catches artifact problems keeps working, but they live here rather than in
`molfusion_backend.artifacts`: the generic infrastructure must stay
representation-agnostic, and "the IDF vector has the wrong dtype" is not a
sentence it should be able to say.

Each failure names one specific broken invariant. A single
`TfidfArtifactError("invalid artifact")` would be useless at three in the
morning, and a bare `Exception` worse still.
"""

from molfusion_backend.artifacts.errors import ArtifactError


class TfidfArtifactError(ArtifactError):
    """Base class for TF-IDF representation errors."""


class TfidfCorpusIdentityError(TfidfArtifactError):
    """The fit corpus is not the frozen corpus this artifact is defined against.

    A hard stop rather than a warning: an artifact fitted on different
    bytes is not a worse artifact, it is an artifact of something else
    wearing the same identity.
    """


class TfidfVocabularyError(TfidfArtifactError):
    """The vocabulary payload is malformed or violates a frozen invariant.

    Covers: wrong dimension, missing/duplicate/non-contiguous indices,
    duplicate n-grams, a term below min_df, a token array that is not a
    list of strings, an `order` that disagrees with `len(tokens)`, and
    index order that is not the frozen lexicographic ordering.
    """


class TfidfIdfError(TfidfArtifactError):
    """The IDF payload is malformed or inconsistent with the vocabulary.

    Covers: wrong shape, wrong dtype, non-finite or non-positive values,
    length that disagrees with the vocabulary, and values that do not
    reproduce the frozen IDF formula from the recorded document
    frequencies.
    """


class TfidfConfigError(TfidfArtifactError):
    """The configuration payload is malformed or does not match the frozen
    contract (schema, contract identifiers, formulas, dtypes, policies)."""


class TfidfArtifactExistsError(TfidfArtifactError):
    """A build would overwrite an existing, already-finalized artifact version.

    Artifact versions are immutable once audited. Rebuilding is done into a
    temporary directory and compared, never in place.
    """


__all__ = [
    "TfidfArtifactError",
    "TfidfArtifactExistsError",
    "TfidfConfigError",
    "TfidfCorpusIdentityError",
    "TfidfIdfError",
    "TfidfVocabularyError",
]
