"""Per-agent error isolation: one representation failing must not discard
another's successful output.

Uses real failures wherever one exists -- hypervalent iodine genuinely
defeats SELFIES' pinned constraints, and a missing or corrupt artifact
genuinely defeats the TF-IDF agent -- so these are end-to-end regressions
rather than assertions about mocks.
"""

import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

from molfusion_backend.agents import registry as agent_registry
from molfusion_backend.agents.base import FeatureAgent
from molfusion_backend.agents.smiles_tfidf import SmilesTfidfAgent
from molfusion_backend.artifacts import sha256_file
from molfusion_backend.artifacts.root import default_artifact_root
from molfusion_backend.main import app
from molfusion_backend.tfidf import contract
from molfusion_backend.tfidf.idf import idf_bytes, load_idf
from tests.tfidf_artifact_fixture import build_fixture_artifact, fixture_dimension

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
INVALID = "not-a-molecule"
# RDKit parses this; SELFIES' pinned default constraints reject it because
# iodine is capped at valence 1. Verified directly in test_api.py.
HYPERVALENT_IODINE = "Cl[I](Cl)Cl"

MORGAN = "morgan_ecfp4_1024"
MACCS = "maccs_keys_167"
SELFIES = "selfies_sequence"
TFIDF = "smiles_tfidf_4096"

PRODUCTION_DIRECTORY = (
    default_artifact_root()
    / contract.ARTIFACT_TYPE
    / contract.ARTIFACT_ID
    / contract.ARTIFACT_VERSION
)
production_artifact = pytest.mark.skipif(
    not (PRODUCTION_DIRECTORY / contract.VOCABULARY_FILENAME).is_file(),
    reason="the frozen production TF-IDF artifact is not present in this checkout",
)


@pytest.fixture()
def client():
    return TestClient(app)


def compute(client, smiles, agent_ids):
    response = client.post(
        "/features/compute", json={"smiles": smiles, "agent_ids": agent_ids}
    )
    assert response.status_code == 200
    return response.json()["results"]


class _AlwaysFails(FeatureAgent):
    """A vector agent that always signals a representation failure."""

    def __init__(self, agent_id: str) -> None:
        self.id = agent_id
        self.name = agent_id
        self.version = "9.9.9"
        self.output_dim = 4
        self.requires_3d = False
        self.value_type = "continuous"
        self.output_structure = "vector"

    def compute(self, mol):
        raise ValueError(f"{self.id}: synthetic representation failure")


class _AlwaysSucceeds(FeatureAgent):
    def __init__(self, agent_id: str) -> None:
        self.id = agent_id
        self.name = agent_id
        self.version = "1.2.3"
        self.output_dim = 3
        self.requires_3d = False
        self.value_type = "continuous"
        self.output_structure = "vector"

    def compute(self, mol):
        return np.array([1.0, 2.0, 3.0], dtype=np.float64)


class _Explodes(FeatureAgent):
    """Raises a programming error, not a representation failure."""

    id = "explodes"
    name = "explodes"
    version = "0.0.1"
    output_dim = 1
    requires_3d = False
    value_type = "continuous"
    output_structure = "vector"

    def compute(self, mol):
        raise TypeError("this is a bug, not a representation outcome")


def patch_agents(monkeypatch, agents):
    """Route registry lookups to a fixed set of agents by id."""
    by_id = {agent.id: agent for agent in agents}
    real_get = agent_registry.get
    monkeypatch.setattr(
        agent_registry, "get", lambda agent_id: by_id.get(agent_id) or real_get(agent_id)
    )


# ---------------------------------------------------------------------------
# molecule validity is unchanged
# ---------------------------------------------------------------------------


def test_invalid_smiles_stays_a_molecule_level_failure(client):
    result = compute(client, [INVALID], [MORGAN, MACCS])[0]
    assert result["valid"] is False
    assert result["error"] is not None
    assert result["features"] == []
    assert result["feature_errors"] == []


def test_invalid_smiles_attempts_no_representation(client, monkeypatch):
    """Nothing should be computed at all -- proven by an agent that would
    have raised loudly if it had been called."""
    patch_agents(monkeypatch, [_Explodes()])
    result = compute(client, [INVALID], [_Explodes.id])[0]
    assert result["valid"] is False
    assert result["features"] == []
    assert result["feature_errors"] == []


# ---------------------------------------------------------------------------
# all success
# ---------------------------------------------------------------------------


@production_artifact
def test_all_agents_succeeding_reports_no_errors(client):
    result = compute(client, [ETHANOL], [MORGAN, MACCS, SELFIES, TFIDF])[0]
    assert result["valid"] is True
    assert result["error"] is None
    assert result["feature_errors"] == []
    assert [feature["agent_id"] for feature in result["features"]] == [
        MORGAN,
        MACCS,
        SELFIES,
        TFIDF,
    ]


def test_successful_output_is_unchanged_by_the_new_contract(client):
    """Phase 5H must not alter any representation's numbers."""
    result = compute(client, [ETHANOL], [MORGAN])[0]
    feature = result["features"][0]
    assert feature["dim"] == 1024
    assert len(feature["values"]) == 1024
    assert set(feature["values"]) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# partial success
# ---------------------------------------------------------------------------


def test_one_failure_does_not_discard_the_other_agents_output(client):
    """The core of Phase 5H, with a real SELFIES failure."""
    result = compute(client, [HYPERVALENT_IODINE], [MORGAN, SELFIES])[0]

    assert result["valid"] is True
    assert result["error"] is None
    assert [feature["agent_id"] for feature in result["features"]] == [MORGAN]
    assert result["features"][0]["dim"] == 1024
    assert len(result["feature_errors"]) == 1
    assert result["feature_errors"][0]["agent_id"] == SELFIES
    assert "SELFIES" in result["feature_errors"][0]["error"]


def test_a_failed_agent_is_absent_from_features(client):
    result = compute(client, [HYPERVALENT_IODINE], [MORGAN, SELFIES, MACCS])[0]
    assert SELFIES not in {feature["agent_id"] for feature in result["features"]}
    assert {feature["agent_id"] for feature in result["features"]} == {MORGAN, MACCS}


def test_the_failure_record_identifies_the_agent_and_version(client):
    result = compute(client, [HYPERVALENT_IODINE], [SELFIES])[0]
    failure = result["feature_errors"][0]
    assert set(failure) == {"agent_id", "agent_version", "error"}
    assert failure["agent_id"] == SELFIES
    assert failure["agent_version"]
    assert failure["error"]


def test_the_failure_record_carries_no_traceback(client):
    result = compute(client, [HYPERVALENT_IODINE], [SELFIES])[0]
    message = result["feature_errors"][0]["error"]
    assert "Traceback" not in message
    assert "File \"" not in message
    assert "\n" not in message


def test_multiple_failures_are_each_recorded(client, monkeypatch):
    failing_a = _AlwaysFails("failing_a")
    failing_b = _AlwaysFails("failing_b")
    patch_agents(monkeypatch, [failing_a, failing_b])

    result = compute(client, [ETHANOL], ["failing_a", MORGAN, "failing_b"])[0]
    assert result["valid"] is True
    assert result["error"] is None
    assert [feature["agent_id"] for feature in result["features"]] == [MORGAN]
    assert [failure["agent_id"] for failure in result["feature_errors"]] == [
        "failing_a",
        "failing_b",
    ]


def test_all_selected_agents_failing_keeps_the_molecule_valid(client, monkeypatch):
    patch_agents(monkeypatch, [_AlwaysFails("failing_a"), _AlwaysFails("failing_b")])

    result = compute(client, [ETHANOL], ["failing_a", "failing_b"])[0]
    assert result["valid"] is True
    assert result["error"] is None
    assert result["features"] == []
    assert len(result["feature_errors"]) == 2


def test_a_failure_does_not_affect_other_molecules_in_the_batch(client):
    results = compute(client, [ETHANOL, HYPERVALENT_IODINE, BENZENE], [MORGAN, SELFIES])
    ethanol, iodine, benzene = results

    for good in (ethanol, benzene):
        assert good["valid"] is True
        assert good["feature_errors"] == []
        assert len(good["features"]) == 2

    assert iodine["valid"] is True
    assert len(iodine["features"]) == 1
    assert len(iodine["feature_errors"]) == 1


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------


def test_features_follow_the_requested_agent_order(client):
    result = compute(client, [ETHANOL], [MACCS, MORGAN])[0]
    assert [feature["agent_id"] for feature in result["features"]] == [MACCS, MORGAN]

    reversed_result = compute(client, [ETHANOL], [MORGAN, MACCS])[0]
    assert [feature["agent_id"] for feature in reversed_result["features"]] == [
        MORGAN,
        MACCS,
    ]


def test_failures_follow_the_requested_agent_order(client, monkeypatch):
    patch_agents(
        monkeypatch,
        [_AlwaysFails("zzz_fails"), _AlwaysFails("aaa_fails"), _AlwaysSucceeds("mid_ok")],
    )
    result = compute(client, [ETHANOL], ["zzz_fails", "mid_ok", "aaa_fails"])[0]

    # Requested order, not alphabetical and not failure order.
    assert [failure["agent_id"] for failure in result["feature_errors"]] == [
        "zzz_fails",
        "aaa_fails",
    ]
    assert [feature["agent_id"] for feature in result["features"]] == ["mid_ok"]


def test_ordering_is_stable_across_repeated_identical_requests(client):
    first = compute(client, [HYPERVALENT_IODINE], [MORGAN, SELFIES, MACCS])[0]
    for _ in range(3):
        again = compute(client, [HYPERVALENT_IODINE], [MORGAN, SELFIES, MACCS])[0]
        assert [f["agent_id"] for f in again["features"]] == [
            f["agent_id"] for f in first["features"]
        ]
        assert [e["agent_id"] for e in again["feature_errors"]] == [
            e["agent_id"] for e in first["feature_errors"]
        ]


# ---------------------------------------------------------------------------
# vector / sequence discrimination under partial success
# ---------------------------------------------------------------------------


@production_artifact
def test_vector_and_sequence_outputs_stay_correctly_discriminated(client):
    result = compute(client, [ETHANOL], [MORGAN, SELFIES, TFIDF])[0]
    by_agent = {feature["agent_id"]: feature for feature in result["features"]}

    assert by_agent[MORGAN]["output_structure"] == "vector"
    assert by_agent[TFIDF]["output_structure"] == "vector"
    assert by_agent[TFIDF]["dim"] == 4096
    assert by_agent[SELFIES]["output_structure"] == "sequence"
    assert "values" not in by_agent[SELFIES]
    assert by_agent[SELFIES]["length"] == len(by_agent[SELFIES]["tokens"])


@production_artifact
def test_a_sequence_failure_leaves_the_vector_outputs_intact(client):
    result = compute(client, [HYPERVALENT_IODINE], [MORGAN, SELFIES, TFIDF])[0]
    by_agent = {feature["agent_id"]: feature for feature in result["features"]}

    assert set(by_agent) == {MORGAN, TFIDF}
    assert by_agent[TFIDF]["dim"] == 4096
    assert [failure["agent_id"] for failure in result["feature_errors"]] == [SELFIES]


# ---------------------------------------------------------------------------
# programming errors are not laundered into representation failures
# ---------------------------------------------------------------------------


def test_a_programming_error_is_not_reported_as_a_feature_failure(client, monkeypatch):
    """A TypeError is a bug. Converting it into a routine per-agent failure
    would hide it exactly where it most needs to be seen."""
    patch_agents(monkeypatch, [_Explodes()])
    with pytest.raises(TypeError):
        client.post(
            "/features/compute",
            json={"smiles": [ETHANOL], "agent_ids": [_Explodes.id]},
        )


# ---------------------------------------------------------------------------
# unknown agent ids remain a request error
# ---------------------------------------------------------------------------


def test_an_unknown_agent_id_is_still_a_400(client):
    """A configuration mistake, not a per-molecule representation failure."""
    response = client.post(
        "/features/compute",
        json={"smiles": [ETHANOL], "agent_ids": [MORGAN, "no_such_agent"]},
    )
    assert response.status_code == 400
    assert "no_such_agent" in response.json()["detail"]


# ---------------------------------------------------------------------------
# TF-IDF artifact failures
# ---------------------------------------------------------------------------


def _tfidf_agent_at(root, dimension=None):
    return SmilesTfidfAgent(
        artifact_root=root, output_dim=dimension if dimension is not None else 4096
    )


def test_a_missing_tfidf_artifact_preserves_the_other_agents_output(
    client, monkeypatch, tmp_path
):
    patch_agents(monkeypatch, [_tfidf_agent_at(tmp_path / "nothing-here")])

    result = compute(client, [ETHANOL], [MORGAN, TFIDF])[0]
    assert result["valid"] is True
    assert result["error"] is None
    assert [feature["agent_id"] for feature in result["features"]] == [MORGAN]
    assert result["features"][0]["dim"] == 1024
    assert len(result["feature_errors"]) == 1
    assert result["feature_errors"][0]["agent_id"] == TFIDF
    assert "artifact could not be loaded" in result["feature_errors"][0]["error"]


def test_a_corrupt_tfidf_artifact_preserves_the_other_agents_output(
    client, monkeypatch, tmp_path
):
    good = tmp_path / "good"
    build_fixture_artifact(good, corpus_dir=tmp_path / "corpus")
    broken = tmp_path / "broken"
    shutil.copytree(good, broken)
    directory = broken / contract.ARTIFACT_TYPE / contract.ARTIFACT_ID / contract.ARTIFACT_VERSION
    path = directory / contract.VOCABULARY_FILENAME
    path.write_bytes(path.read_bytes().replace(b'"index": 0', b'"index": 1', 1))

    patch_agents(monkeypatch, [_tfidf_agent_at(broken, fixture_dimension(broken))])

    result = compute(client, [ETHANOL], [MORGAN, TFIDF])[0]
    assert result["valid"] is True
    assert result["error"] is None
    assert [feature["agent_id"] for feature in result["features"]] == [MORGAN]
    assert len(result["feature_errors"]) == 1
    assert result["feature_errors"][0]["agent_id"] == TFIDF


def test_a_semantically_corrupt_tfidf_artifact_preserves_other_output(
    client, monkeypatch, tmp_path
):
    """Checksum-valid but formula-wrong -- caught by semantic validation,
    and still isolated to its own agent."""
    good = tmp_path / "good"
    build_fixture_artifact(good, corpus_dir=tmp_path / "corpus")
    broken = tmp_path / "broken"
    shutil.copytree(good, broken)
    directory = broken / contract.ARTIFACT_TYPE / contract.ARTIFACT_ID / contract.ARTIFACT_VERSION

    values = load_idf(directory / contract.IDF_FILENAME)
    values[0] += 0.5
    (directory / contract.IDF_FILENAME).write_bytes(idf_bytes(values))
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for record in metadata["payload_files"]:
        if record["filename"] == contract.IDF_FILENAME:
            record["sha256"] = sha256_file(directory / contract.IDF_FILENAME)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    patch_agents(monkeypatch, [_tfidf_agent_at(broken, fixture_dimension(broken))])

    result = compute(client, [ETHANOL], [MACCS, TFIDF])[0]
    assert [feature["agent_id"] for feature in result["features"]] == [MACCS]
    assert [failure["agent_id"] for failure in result["feature_errors"]] == [TFIDF]


def test_a_working_fixture_artifact_produces_no_feature_error(client, monkeypatch, tmp_path):
    """Control for the two corruption tests above: the same wiring with an
    intact artifact must succeed, so those failures are attributable to the
    corruption and not to the fixture setup."""
    root = tmp_path / "good"
    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")
    patch_agents(monkeypatch, [_tfidf_agent_at(root, fixture_dimension(root))])

    result = compute(client, [ETHANOL], [MORGAN, TFIDF])[0]
    assert result["feature_errors"] == []
    assert [feature["agent_id"] for feature in result["features"]] == [MORGAN, TFIDF]
    assert result["features"][1]["dim"] == fixture_dimension(root)


# ---------------------------------------------------------------------------
# the all-OOV zero vector is a success, not a failure
# ---------------------------------------------------------------------------


@production_artifact
def test_an_all_oov_molecule_is_a_successful_zero_vector_not_a_failure(client):
    """Case C: the representation computed correctly and the answer is a
    zero vector. It must appear in `features`, not in `feature_errors`."""
    result = compute(client, ["[Xe]"], [TFIDF])[0]

    assert result["valid"] is True
    assert result["error"] is None
    assert result["feature_errors"] == []
    assert len(result["features"]) == 1

    feature = result["features"][0]
    assert feature["agent_id"] == TFIDF
    assert feature["dim"] == 4096
    assert len(feature["values"]) == 4096
    assert all(value == 0.0 for value in feature["values"])


@production_artifact
def test_an_all_oov_zero_vector_survives_a_sibling_agent_failure(client):
    """The zero vector must not be mistaken for the failure when another
    agent in the same request genuinely fails."""
    result = compute(client, ["[Xe]"], [TFIDF, SELFIES])[0]
    by_agent = {feature["agent_id"]: feature for feature in result["features"]}

    if TFIDF in by_agent:
        assert all(value == 0.0 for value in by_agent[TFIDF]["values"])
        assert TFIDF not in {f["agent_id"] for f in result["feature_errors"]}
