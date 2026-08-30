class CorpusBuildError(Exception):
    """Base class for all reference-corpus build errors."""


class CorpusSourceError(CorpusBuildError):
    """Raised when the source database is missing, unreadable, or does not
    expose the structural schema the builder requires."""


class CorpusOutputExistsError(CorpusBuildError):
    """Raised when a finalized output file already exists and no explicit
    overwrite was requested."""


class TokenizerContractViolation(CorpusBuildError):
    """Raised when an RDKit-generated canonical SMILES fails the Phase 5F-A
    lossless-tokenization invariant.

    This is never routine input noise: the input to the tokenizer here is
    output from our own normalizer, so a violation means the normalization
    and tokenization contracts have drifted apart and every downstream
    vocabulary/document-frequency derived from this corpus would be
    suspect. The build aborts rather than quietly dropping the record.
    """


class CorpusIdentityError(CorpusBuildError):
    """Raised when a corpus on disk is not the frozen corpus a consumer was
    defined against.

    A study, a fit, or a vocabulary freeze is only meaningful relative to
    an exact corpus. Running one against different bytes does not weaken
    the result, it silently produces a result about something else, so the
    digest check is a hard stop rather than a warning.
    """
