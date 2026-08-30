"""Loading and semantically validating a frozen TF-IDF artifact.

Two layers, deliberately separate:

    load_artifact()      generic: resolves the directory, validates
                         metadata.json against a schema that knows nothing
                         about TF-IDF, and verifies every payload checksum
        |
    load_tfidf_artifact()  representation-specific: parses the vocabulary,
                           loads the IDF array, validates the config
                           against the frozen contract, and checks that all
                           three agree with each other

The generic layer stays unaware of NumPy and TF-IDF -- it hands back
verified paths and nothing more. Everything that knows what an IDF vector
is lives here.

Checksums answer "are these the bytes that were built?". They cannot
answer "do these bytes mean what the artifact claims?" -- a payload
rebuilt with a different formula would have a perfectly valid checksum. So
the semantic validation below re-derives the IDF from the recorded
document frequencies rather than trusting that it was computed correctly.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from molfusion_backend.artifacts.loader import ArtifactDescriptor, load_artifact
from molfusion_backend.tfidf import contract, idf as idf_module, vocabulary as vocabulary_module
from molfusion_backend.tfidf.errors import (
    TfidfConfigError,
    TfidfCorpusIdentityError,
    TfidfVocabularyError,
)
from molfusion_backend.tfidf.transform import TfidfTransformer


@dataclass(frozen=True)
class TfidfArtifact:
    """A loaded, checksum-verified, semantically validated TF-IDF artifact."""

    descriptor: ArtifactDescriptor
    config: contract.TfidfConfig
    vocabulary: vocabulary_module.Vocabulary
    idf: np.ndarray

    @property
    def dimension(self) -> int:
        return self.vocabulary.dimension

    @property
    def fit_corpus_sha256(self) -> str | None:
        fit_corpus = self.descriptor.metadata.fit_corpus
        return fit_corpus.checksum if fit_corpus else None

    def feature_names(self) -> list[str]:
        return self.vocabulary.feature_names()

    def transformer(self) -> TfidfTransformer:
        """The frozen transformation, ready to use.

        Not a FeatureAgent: no registry entry, no agent interface, no API
        surface. Just the vocabulary and IDF bound to the frozen formulas.
        """
        return TfidfTransformer(
            index_map=self.vocabulary.index_map(),
            idf=self.idf,
            dimension=self.dimension,
            orders=tuple(range(self.config.ngram_min, self.config.ngram_max + 1)),
        )


def load_tfidf_artifact(
    artifact_id: str = contract.ARTIFACT_ID,
    artifact_version: str = contract.ARTIFACT_VERSION,
    *,
    artifact_type: str = contract.ARTIFACT_TYPE,
    root: Path | None = None,
    expected_fit_corpus_sha256: str | None = None,
    enforce_contract: bool = True,
    expected_min_df: int = contract.MIN_DF,
    expected_max_features: int = contract.MAX_FEATURES,
) -> TfidfArtifact:
    """Load-or-fail. Any inconsistency raises before a vector can be produced.

    `expected_min_df` and `expected_max_features` default to the frozen
    production policy. They are overridable so a test fixture built with a
    smaller vocabulary can still be held to the *rest* of the contract --
    the formulas, dtypes, index ordering and contract identifiers -- rather
    than having to switch all validation off to accommodate two numbers.
    """
    descriptor = load_artifact(artifact_type, artifact_id, artifact_version, root=root)

    config = _load_config(descriptor)
    if enforce_contract:
        mismatches = contract.contract_mismatches(
            config, min_df=expected_min_df, max_features=expected_max_features
        )
        if mismatches:
            raise TfidfConfigError(
                f"{descriptor.directory / contract.CONFIG_FILENAME} does not match the "
                "frozen contract: " + "; ".join(mismatches)
            )

    vocabulary = _load_vocabulary(descriptor)
    vocabulary_module.validate_vocabulary(
        vocabulary,
        dimension=config.dimension,
        min_df=config.min_df,
        orders=range(config.ngram_min, config.ngram_max + 1),
    )

    idf_values = idf_module.load_idf(_payload_path(descriptor, contract.IDF_FILENAME))
    idf_module.validate_idf(
        idf_values,
        dimension=vocabulary.dimension,
        document_frequencies=vocabulary.document_frequencies(),
        n_documents=config.fit_document_count,
    )

    if expected_fit_corpus_sha256 is not None:
        actual = descriptor.metadata.fit_corpus.checksum if descriptor.metadata.fit_corpus else None
        if actual != expected_fit_corpus_sha256:
            raise TfidfCorpusIdentityError(
                "Artifact was fitted on a different corpus: metadata records "
                f"{actual!r}, caller requires {expected_fit_corpus_sha256!r}"
            )

    return TfidfArtifact(
        descriptor=descriptor, config=config, vocabulary=vocabulary, idf=idf_values
    )


def _payload_path(descriptor: ArtifactDescriptor, filename: str) -> Path:
    path = descriptor.payload_paths.get(filename)
    if path is None:
        raise TfidfConfigError(
            f"artifact does not declare required payload {filename!r}; "
            f"declared: {sorted(descriptor.payload_paths)}"
        )
    return path


def _load_config(descriptor: ArtifactDescriptor) -> contract.TfidfConfig:
    path = _payload_path(descriptor, contract.CONFIG_FILENAME)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise TfidfConfigError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TfidfConfigError(f"{path} is not valid JSON: {exc}") from exc
    try:
        return contract.TfidfConfig.model_validate(raw)
    except ValidationError as exc:
        raise TfidfConfigError(f"{path} failed schema validation: {exc}") from exc


def _load_vocabulary(descriptor: ArtifactDescriptor) -> vocabulary_module.Vocabulary:
    path = _payload_path(descriptor, contract.VOCABULARY_FILENAME)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise TfidfVocabularyError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TfidfVocabularyError(f"{path} is not valid JSON: {exc}") from exc
    return vocabulary_module.parse_vocabulary(raw)


__all__ = ["TfidfArtifact", "load_tfidf_artifact"]
