import pytest

from molfusion_backend.artifacts.root import (
    default_artifact_root,
    resolve_artifact_root,
    validate_path_component,
)


def test_default_artifact_root_is_named_artifacts_under_backend():
    root = default_artifact_root()
    assert root.name == "artifacts"
    assert root.parent.name == "backend"


def test_default_artifact_root_does_not_depend_on_cwd(monkeypatch, tmp_path):
    before = default_artifact_root()
    monkeypatch.chdir(tmp_path)
    after = default_artifact_root()
    assert before == after


def test_resolve_artifact_root_defaults_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("MOLFUSION_ARTIFACT_ROOT", raising=False)
    assert resolve_artifact_root() == default_artifact_root()


def test_resolve_artifact_root_defaults_when_env_var_empty(monkeypatch):
    monkeypatch.setenv("MOLFUSION_ARTIFACT_ROOT", "")
    assert resolve_artifact_root() == default_artifact_root()


def test_resolve_artifact_root_honors_env_var_override(monkeypatch, tmp_path):
    override = tmp_path / "custom_artifacts"
    monkeypatch.setenv("MOLFUSION_ARTIFACT_ROOT", str(override))
    assert resolve_artifact_root() == override.resolve()


@pytest.mark.parametrize("value", ["1.0.0", "pubchem_smiles_tfidf", "tfidf", "a.b.c-d_e"])
def test_validate_path_component_accepts_normal_values(value):
    assert validate_path_component(value, "field") == value


@pytest.mark.parametrize(
    "value",
    ["", " ", "  padded  ", ".", "..", "a/b", "a\\b", "../escape", "..\\escape", "/abs", "a/../b"],
)
def test_validate_path_component_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        validate_path_component(value, "field")
