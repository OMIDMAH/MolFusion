import pytest
from pydantic import ValidationError

from molfusion_backend.artifacts.metadata import ArtifactMetadata, FitCorpus, PayloadFile

VALID_SHA256 = "a" * 64


def _minimal_metadata_kwargs(**overrides):
    kwargs = {
        "artifact_id": "pubchem_smiles_tfidf",
        "artifact_version": "1.0.0",
        "artifact_type": "tfidf",
        "created_at": "2026-01-01T00:00:00Z",
        "payload_files": [{"filename": "vectorizer.joblib", "sha256": VALID_SHA256}],
    }
    kwargs.update(overrides)
    return kwargs


def test_minimal_valid_metadata_parses_with_defaults():
    metadata = ArtifactMetadata.model_validate(_minimal_metadata_kwargs())
    assert metadata.library_versions == {}
    assert metadata.configuration == {}
    assert metadata.fit_corpus is None
    assert metadata.random_seed is None
    assert metadata.description is None


def test_full_metadata_with_fit_corpus_and_seed_parses():
    metadata = ArtifactMetadata.model_validate(
        _minimal_metadata_kwargs(
            library_versions={"scikit-learn": "1.5.0"},
            configuration={"ngram_range": [1, 2], "max_features": 32},
            fit_corpus={
                "name": "pubchem_sample",
                "version": "2026-01",
                "checksum": "b" * 64,
                "record_count": 1_000_000,
                "source": "https://pubchem.ncbi.nlm.nih.gov/",
            },
            random_seed=42,
            description="Reference TF-IDF vectorizer for SMILES strings.",
        )
    )
    assert metadata.fit_corpus == FitCorpus(
        name="pubchem_sample",
        version="2026-01",
        checksum="b" * 64,
        record_count=1_000_000,
        source="https://pubchem.ncbi.nlm.nih.gov/",
    )
    assert metadata.random_seed == 42


def test_random_seed_nullable_for_non_stochastic_artifacts():
    metadata = ArtifactMetadata.model_validate(_minimal_metadata_kwargs(random_seed=None))
    assert metadata.random_seed is None


def test_payload_files_requires_at_least_one():
    with pytest.raises(ValidationError):
        ArtifactMetadata.model_validate(_minimal_metadata_kwargs(payload_files=[]))


def test_duplicate_payload_filenames_are_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        ArtifactMetadata.model_validate(
            _minimal_metadata_kwargs(
                payload_files=[
                    {"filename": "vectorizer.joblib", "sha256": VALID_SHA256},
                    {"filename": "vectorizer.joblib", "sha256": "c" * 64},
                ]
            )
        )


@pytest.mark.parametrize("bad_filename", ["../escape.joblib", "sub/dir.joblib", "sub\\dir.joblib", ""])
def test_payload_filename_rejects_path_traversal_and_separators(bad_filename):
    with pytest.raises(ValidationError):
        PayloadFile(filename=bad_filename, sha256=VALID_SHA256)


@pytest.mark.parametrize("bad_sha256", ["short", "z" * 64, "A" * 63, ""])
def test_payload_sha256_must_be_64_hex_characters(bad_sha256):
    with pytest.raises(ValidationError):
        PayloadFile(filename="vectorizer.joblib", sha256=bad_sha256)


def test_payload_sha256_is_normalized_to_lowercase():
    payload = PayloadFile(filename="vectorizer.joblib", sha256="A" * 64)
    assert payload.sha256 == "a" * 64


@pytest.mark.parametrize("field", ["artifact_id", "artifact_version", "artifact_type"])
def test_identity_fields_reject_path_traversal(field):
    with pytest.raises(ValidationError):
        ArtifactMetadata.model_validate(_minimal_metadata_kwargs(**{field: "../escape"}))


@pytest.mark.parametrize("field", ["artifact_id", "artifact_version", "artifact_type"])
def test_identity_fields_reject_empty_string(field):
    with pytest.raises(ValidationError):
        ArtifactMetadata.model_validate(_minimal_metadata_kwargs(**{field: ""}))
