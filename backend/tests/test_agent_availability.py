"""Agent health, availability and request preflight.

Availability answers "can this agent run here at all?". Phase 5H's
`feature_errors` answers "did this molecule work?". These tests exist
largely to keep those two apart.
"""

import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

from molfusion_backend.agents import registry as agent_registry
from molfusion_backend.agents.availability import (
    AVAILABILITY_CODES,
    AVAILABLE,
    AgentAvailability,
    CODE_ARTIFACT_CHECKSUM_ERROR,
    CODE_ARTIFACT_MISSING,
    CODE_ARTIFACT_SEMANTIC_ERROR,
    CODE_CONFIGURATION_ERROR,
    CODE_HEALTH_CHECK_ERROR,
    check_agent,
    check_agents,
    unavailable,
)
from molfusion_backend.agents.base import FeatureAgent
from molfusion_backend.agents.smiles_tfidf import SmilesTfidfAgent
from molfusion_backend.artifacts import sha256_file
from molfusion_backend.artifacts.root import default_artifact_root
from molfusion_backend.main import app
from molfusion_backend.tfidf import contract
from molfusion_backend.tfidf.idf import idf_bytes, load_idf
from tests.tfidf_artifact_fixture import build_fixture_artifact, fixture_dimension

ETHANOL = "CCO"
HYPERVALENT_IODINE = "Cl[I](Cl)Cl"
INVALID = "not-a-molecule"
MORGAN = "morgan_ecfp4_1024"
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


def patch_agents(monkeypatch, agents):
    """Route registry lookups and listings to a fixed set of agents."""
    by_id = {agent.id: agent for agent in agents}
    real_get = agent_registry.get
    real_agents = agent_registry.agents
    monkeypatch.setattr(
        agent_registry, "get", lambda agent_id: by_id.get(agent_id) or real_get(agent_id)
    )
    monkeypatch.setattr(
        agent_registry,
        "agents",
        lambda: [by_id.get(agent.id, agent) for agent in real_agents()],
    )


def corrupt_checksum(root):
    directory = root / contract.ARTIFACT_TYPE / contract.ARTIFACT_ID / contract.ARTIFACT_VERSION
    path = directory / contract.VOCABULARY_FILENAME
    path.write_bytes(path.read_bytes().replace(b'"index": 0', b'"index": 1', 1))


def corrupt_semantics(root):
    """Checksum-valid but formula-wrong: only semantic validation catches it."""
    directory = root / contract.ARTIFACT_TYPE / contract.ARTIFACT_ID / contract.ARTIFACT_VERSION
    values = load_idf(directory / contract.IDF_FILENAME)
    values[0] += 0.5
    (directory / contract.IDF_FILENAME).write_bytes(idf_bytes(values))
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for record in metadata["payload_files"]:
        if record["filename"] == contract.IDF_FILENAME:
            record["sha256"] = sha256_file(directory / contract.IDF_FILENAME)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# the model itself
# ---------------------------------------------------------------------------


def test_available_carries_no_code_or_message():
    assert AVAILABLE.available is True
    assert AVAILABLE.code is None
    assert AVAILABLE.message is None


def test_unavailable_carries_a_stable_code_and_a_message():
    result = unavailable(CODE_ARTIFACT_MISSING, "gone")
    assert result.available is False
    assert result.code == CODE_ARTIFACT_MISSING
    assert result.message == "gone"


def test_codes_are_generic_not_representation_specific():
    """A client should branch on the kind of problem, not on which
    representation happens to have it."""
    for code in AVAILABILITY_CODES:
        assert "tfidf" not in code
        assert "selfies" not in code
        assert "morgan" not in code


def test_availability_is_immutable():
    with pytest.raises(Exception):
        AVAILABLE.available = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# stateless agents
# ---------------------------------------------------------------------------


STATELESS_IDS = [
    MORGAN,
    "maccs_keys_167",
    "rdkit_physchem_descriptors",
    "avalon_1024",
    "erg_reduced_graph_315",
    "rdkit_fragment_descriptors",
    SELFIES,
]


@pytest.mark.parametrize("agent_id", STATELESS_IDS)
def test_stateless_production_agents_report_available(agent_id):
    assert agent_registry.get(agent_id).check_availability() == AVAILABLE


def test_the_default_health_check_needs_no_boilerplate():
    """A simple agent inherits availability without implementing anything."""

    class _Plain(FeatureAgent):
        id = "plain"
        name = "plain"
        version = "1.0.0"
        output_dim = 1
        requires_3d = False
        value_type = "continuous"
        output_structure = "vector"

        def compute(self, mol):
            return np.zeros(1)

    assert _Plain().check_availability().available is True


def test_a_health_check_never_calls_compute(monkeypatch):
    """The rule that keeps health separate from one molecule's luck."""
    called = []

    class _Tripwire(FeatureAgent):
        id = "tripwire"
        name = "tripwire"
        version = "1.0.0"
        output_dim = 1
        requires_3d = False
        value_type = "continuous"
        output_structure = "vector"

        def compute(self, mol):
            called.append(1)
            return np.zeros(1)

    assert _Tripwire().check_availability().available is True
    assert called == []


# ---------------------------------------------------------------------------
# health-check isolation
# ---------------------------------------------------------------------------


class _BrokenHealth(FeatureAgent):
    id = "broken_health"
    name = "broken health"
    version = "1.0.0"
    output_dim = 1
    requires_3d = False
    value_type = "continuous"
    output_structure = "vector"

    def check_availability(self):
        raise RuntimeError("health check itself is broken")

    def compute(self, mol):
        return np.zeros(1)


def test_an_exploding_health_check_is_isolated_and_flagged(caplog):
    """Reported unavailable with a distinct code rather than assumed fine,
    and logged so the bug stays visible."""
    result = check_agent(_BrokenHealth())
    assert result.available is False
    assert result.code == CODE_HEALTH_CHECK_ERROR
    assert "health check itself is broken" not in (result.message or "")


def test_one_broken_health_check_does_not_affect_the_others():
    results = check_agents([_BrokenHealth(), agent_registry.get(MORGAN)])
    assert results["broken_health"].available is False
    assert results[MORGAN].available is True


# ---------------------------------------------------------------------------
# TF-IDF availability
# ---------------------------------------------------------------------------


def test_a_healthy_fixture_artifact_is_available(tmp_path):
    root = tmp_path / "artifacts"
    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")
    agent = SmilesTfidfAgent(artifact_root=root, output_dim=fixture_dimension(root))
    assert agent.check_availability() == AVAILABLE


def test_a_missing_artifact_is_unavailable(tmp_path):
    agent = SmilesTfidfAgent(artifact_root=tmp_path / "nothing-here")
    result = agent.check_availability()
    assert result.available is False
    assert result.code == CODE_ARTIFACT_MISSING
    assert result.message


def test_a_checksum_corrupt_artifact_is_unavailable(tmp_path):
    root = tmp_path / "artifacts"
    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")
    corrupt_checksum(root)
    agent = SmilesTfidfAgent(artifact_root=root, output_dim=fixture_dimension(root))
    result = agent.check_availability()
    assert result.available is False
    assert result.code == CODE_ARTIFACT_CHECKSUM_ERROR


def test_a_semantically_corrupt_artifact_is_unavailable(tmp_path):
    root = tmp_path / "artifacts"
    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")
    corrupt_semantics(root)
    agent = SmilesTfidfAgent(artifact_root=root, output_dim=fixture_dimension(root))
    result = agent.check_availability()
    assert result.available is False
    assert result.code == CODE_ARTIFACT_SEMANTIC_ERROR


def test_a_dimension_mismatch_is_a_configuration_problem(tmp_path):
    root = tmp_path / "artifacts"
    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")
    agent = SmilesTfidfAgent(artifact_root=root, output_dim=99999)
    result = agent.check_availability()
    assert result.available is False
    assert result.code == CODE_CONFIGURATION_ERROR


def test_the_health_message_leaks_no_path_or_traceback(tmp_path):
    agent = SmilesTfidfAgent(artifact_root=tmp_path / "secret-place")
    message = agent.check_availability().message or ""
    assert "secret-place" not in message
    assert "Traceback" not in message
    assert str(tmp_path) not in message
    assert "\n" not in message


def test_the_health_check_does_not_transform_a_molecule(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")
    agent = SmilesTfidfAgent(artifact_root=root, output_dim=fixture_dimension(root))

    calls = []
    monkeypatch.setattr(
        SmilesTfidfAgent, "compute", lambda self, mol: calls.append(1)
    )
    assert agent.check_availability().available is True
    assert calls == []


def test_an_all_oov_molecule_does_not_make_the_agent_unavailable(tmp_path):
    """A zero vector is a successful computation, not a health problem."""
    root = tmp_path / "artifacts"
    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")
    agent = SmilesTfidfAgent(artifact_root=root, output_dim=fixture_dimension(root))

    vector = agent.compute(Chem.MolFromSmiles("[Xe]"))
    assert not vector.any()
    assert agent.check_availability().available is True


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


def test_a_negative_result_is_not_cached_for_the_process_lifetime(tmp_path):
    """Restoring the artifact must be observed without a restart."""
    root = tmp_path / "artifacts"
    agent = SmilesTfidfAgent(artifact_root=root, output_dim=4096)

    assert agent.check_availability().code == CODE_ARTIFACT_MISSING
    assert agent.check_availability().code == CODE_ARTIFACT_MISSING

    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")
    agent.output_dim = fixture_dimension(root)
    assert agent.check_availability().available is True


def test_recovery_also_restores_compute(tmp_path):
    root = tmp_path / "artifacts"
    agent = SmilesTfidfAgent(artifact_root=root, output_dim=4096)
    assert agent.check_availability().available is False

    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")
    agent.output_dim = fixture_dimension(root)
    vector = agent.compute(Chem.MolFromSmiles(ETHANOL))
    assert vector.shape == (fixture_dimension(root),)


def test_a_successful_check_does_not_reload_the_artifact(tmp_path, monkeypatch):
    """Repeated polling of /agents must not re-read the payloads."""
    from molfusion_backend.agents import smiles_tfidf as agent_module

    root = tmp_path / "artifacts"
    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")

    calls = []
    real_loader = agent_module.load_tfidf_artifact
    monkeypatch.setattr(
        agent_module,
        "load_tfidf_artifact",
        lambda *a, **k: (calls.append(1), real_loader(*a, **k))[1],
    )

    agent = SmilesTfidfAgent(artifact_root=root, output_dim=fixture_dimension(root))
    for _ in range(5):
        assert agent.check_availability().available is True
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# GET /agents
# ---------------------------------------------------------------------------


def test_agents_endpoint_reports_availability_for_every_agent(client):
    payload = client.get("/agents").json()
    assert len(payload) == 8
    for entry in payload:
        assert "availability" in entry
        assert isinstance(entry["availability"]["available"], bool)


def test_an_unavailable_agent_stays_listed(client, monkeypatch, tmp_path):
    patch_agents(monkeypatch, [SmilesTfidfAgent(artifact_root=tmp_path / "nothing-here")])

    payload = client.get("/agents").json()
    ids = [entry["id"] for entry in payload]
    assert len(ids) == 8
    assert TFIDF in ids

    entry = next(item for item in payload if item["id"] == TFIDF)
    assert entry["availability"]["available"] is False
    assert entry["availability"]["code"] == CODE_ARTIFACT_MISSING


def test_one_unavailable_agent_does_not_break_the_others(client, monkeypatch, tmp_path):
    patch_agents(monkeypatch, [SmilesTfidfAgent(artifact_root=tmp_path / "nothing-here")])

    payload = client.get("/agents").json()
    by_id = {entry["id"]: entry for entry in payload}
    for agent_id in STATELESS_IDS:
        assert by_id[agent_id]["availability"]["available"] is True
    assert by_id[MORGAN]["output_dim"] == 1024


def test_a_broken_health_check_does_not_break_the_listing(client, monkeypatch):
    patch_agents(monkeypatch, [])
    real_agents = agent_registry.agents
    monkeypatch.setattr(
        agent_registry, "agents", lambda: [*real_agents(), _BrokenHealth()]
    )
    real_list = agent_registry.list_agents
    monkeypatch.setattr(
        agent_registry,
        "list_agents",
        lambda: [
            *real_list(),
            {
                "id": "broken_health",
                "name": "broken health",
                "version": "1.0.0",
                "output_dim": 1,
                "requires_3d": False,
                "value_type": "continuous",
                "output_structure": "vector",
                "feature_names": None,
            },
        ],
    )

    response = client.get("/agents")
    assert response.status_code == 200
    by_id = {entry["id"]: entry for entry in response.json()}
    assert by_id["broken_health"]["availability"]["code"] == CODE_HEALTH_CHECK_ERROR
    assert by_id[MORGAN]["availability"]["available"] is True


def test_agent_ordering_is_preserved(client):
    first = [entry["id"] for entry in client.get("/agents").json()]
    second = [entry["id"] for entry in client.get("/agents").json()]
    assert first == second
    assert first[0] == MORGAN


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


class _Tripwire(FeatureAgent):
    """Records every compute call, so a test can prove the loop never ran."""

    id = "tripwire"
    name = "tripwire"
    version = "1.0.0"
    output_dim = 2
    requires_3d = False
    value_type = "continuous"
    output_structure = "vector"

    def __init__(self):
        self.calls = []

    def compute(self, mol):
        self.calls.append(1)
        return np.zeros(2)


def test_an_unavailable_selected_agent_rejects_the_request(client, monkeypatch, tmp_path):
    patch_agents(monkeypatch, [SmilesTfidfAgent(artifact_root=tmp_path / "nothing-here")])

    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": [MORGAN, TFIDF]}
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["message"]
    assert [entry["agent_id"] for entry in detail["agents"]] == [TFIDF]
    assert detail["agents"][0]["code"] == CODE_ARTIFACT_MISSING
    assert detail["agents"][0]["message"]


def test_preflight_runs_before_the_molecule_loop(client, monkeypatch, tmp_path):
    """The point of the phase: no molecule is touched when a selected agent
    cannot run."""
    tripwire = _Tripwire()
    patch_agents(
        monkeypatch,
        [tripwire, SmilesTfidfAgent(artifact_root=tmp_path / "nothing-here")],
    )

    response = client.post(
        "/features/compute",
        json={"smiles": [ETHANOL] * 50, "agent_ids": [tripwire.id, TFIDF]},
    )
    assert response.status_code == 409
    assert tripwire.calls == []


def test_a_rejected_request_produces_no_per_molecule_errors(client, monkeypatch, tmp_path):
    """A systemic fault must not be reported once per molecule."""
    patch_agents(monkeypatch, [SmilesTfidfAgent(artifact_root=tmp_path / "nothing-here")])

    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL] * 100, "agent_ids": [TFIDF]}
    )
    assert response.status_code == 409
    body = response.json()
    assert "results" not in body
    assert len(body["detail"]["agents"]) == 1


def test_preflight_checks_each_selected_agent_exactly_once(client, monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    build_fixture_artifact(root, corpus_dir=tmp_path / "corpus")
    agent = SmilesTfidfAgent(artifact_root=root, output_dim=fixture_dimension(root))

    checks = []
    original = SmilesTfidfAgent.check_availability
    monkeypatch.setattr(
        SmilesTfidfAgent,
        "check_availability",
        lambda self: (checks.append(1), original(self))[1],
    )
    patch_agents(monkeypatch, [agent])

    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL] * 20, "agent_ids": [TFIDF]}
    )
    assert response.status_code == 200
    assert len(checks) == 1


def test_several_unavailable_agents_are_reported_together(client, monkeypatch, tmp_path):
    class _Down(FeatureAgent):
        def __init__(self, agent_id):
            self.id = agent_id
            self.name = agent_id
            self.version = "1.0.0"
            self.output_dim = 1
            self.requires_3d = False
            self.value_type = "continuous"
            self.output_structure = "vector"

        def check_availability(self):
            return unavailable(CODE_CONFIGURATION_ERROR, f"{self.id} is misconfigured")

        def compute(self, mol):
            return np.zeros(1)

    patch_agents(monkeypatch, [_Down("down_b"), _Down("down_a")])

    response = client.post(
        "/features/compute",
        json={"smiles": [ETHANOL], "agent_ids": ["down_b", MORGAN, "down_a"]},
    )
    assert response.status_code == 409
    # Requested order, not alphabetical.
    assert [entry["agent_id"] for entry in response.json()["detail"]["agents"]] == [
        "down_b",
        "down_a",
    ]


def test_an_agent_whose_health_check_explodes_blocks_the_request(client, monkeypatch):
    patch_agents(monkeypatch, [_BrokenHealth()])
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["broken_health"]}
    )
    assert response.status_code == 409
    assert response.json()["detail"]["agents"][0]["code"] == CODE_HEALTH_CHECK_ERROR


def test_an_unknown_agent_is_still_a_400_not_a_409(client):
    """Unknown and unavailable are different: one is a request mistake, the
    other an environment problem."""
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["no_such_agent"]}
    )
    assert response.status_code == 400


def test_unknown_is_checked_before_availability(client, monkeypatch, tmp_path):
    patch_agents(monkeypatch, [SmilesTfidfAgent(artifact_root=tmp_path / "nothing-here")])
    response = client.post(
        "/features/compute",
        json={"smiles": [ETHANOL], "agent_ids": [TFIDF, "no_such_agent"]},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# healthy preflight leaves Phase 5H behaviour intact
# ---------------------------------------------------------------------------


def test_a_healthy_request_proceeds_normally(client):
    result = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": [MORGAN, SELFIES]}
    ).json()["results"][0]
    assert result["valid"] is True
    assert result["error"] is None
    assert result["feature_errors"] == []
    assert len(result["features"]) == 2


def test_a_molecule_specific_selfies_failure_is_not_a_health_failure(client):
    """The Phase 5H regression: SELFIES stays available, the molecule is
    valid, Morgan is retained, and the failure is per molecule."""
    assert agent_registry.get(SELFIES).check_availability().available is True

    response = client.post(
        "/features/compute",
        json={"smiles": [HYPERVALENT_IODINE], "agent_ids": [MORGAN, SELFIES]},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert [feature["agent_id"] for feature in result["features"]] == [MORGAN]
    assert [failure["agent_id"] for failure in result["feature_errors"]] == [SELFIES]

    # And the agent is still healthy afterwards.
    assert agent_registry.get(SELFIES).check_availability().available is True


def test_an_invalid_molecule_still_behaves_as_before(client):
    result = client.post(
        "/features/compute", json={"smiles": [INVALID], "agent_ids": [MORGAN]}
    ).json()["results"][0]
    assert result["valid"] is False
    assert result["error"]
    assert result["features"] == []
    assert result["feature_errors"] == []


@production_artifact
def test_an_all_oov_molecule_still_succeeds_through_preflight(client):
    response = client.post(
        "/features/compute", json={"smiles": ["[Xe]"], "agent_ids": [TFIDF]}
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["feature_errors"] == []
    feature = result["features"][0]
    assert feature["dim"] == 4096
    assert all(value == 0.0 for value in feature["values"])


# ---------------------------------------------------------------------------
# production artifact smoke test (read-only)
# ---------------------------------------------------------------------------


@production_artifact
def test_the_production_tfidf_agent_is_available():
    assert SmilesTfidfAgent().check_availability() == AVAILABLE
    assert agent_registry.get(TFIDF).check_availability().available is True


@production_artifact
def test_the_health_smoke_test_does_not_touch_the_artifact():
    before = {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(PRODUCTION_DIRECTORY.iterdir())
    }
    for _ in range(3):
        assert SmilesTfidfAgent().check_availability().available is True
    after = {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(PRODUCTION_DIRECTORY.iterdir())
    }
    assert before == after


@production_artifact
def test_every_production_agent_reports_available(client):
    payload = client.get("/agents").json()
    assert all(entry["availability"]["available"] for entry in payload)
    assert all(entry["availability"]["code"] is None for entry in payload)
