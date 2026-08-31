"""Registry, API and real-production-artifact integration for the TF-IDF agent."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from molfusion_backend.agents import registry
from molfusion_backend.agents.smiles_tfidf import AGENT_ID, SmilesTfidfAgent
from molfusion_backend.artifacts.root import default_artifact_root
from molfusion_backend.main import app
from molfusion_backend.tfidf import contract

# The real frozen artifact, used read-only by the smoke tests below. They
# skip rather than fail where it is absent: a developer clone without the
# built artifact should still get a green suite.
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


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_the_agent_is_registered_exactly_once():
    ids = [entry["id"] for entry in registry.list_agents()]
    assert ids.count(AGENT_ID) == 1


def test_registry_metadata_describes_a_continuous_4096_vector():
    entry = next(item for item in registry.list_agents() if item["id"] == AGENT_ID)
    assert entry["value_type"] == "continuous"
    assert entry["output_structure"] == "vector"
    assert entry["output_dim"] == 4096
    assert entry["requires_3d"] is False


def test_the_agent_sits_alongside_the_other_vector_agents():
    entries = registry.list_agents()
    vector_ids = {item["id"] for item in entries if item["output_structure"] == "vector"}
    assert AGENT_ID in vector_ids
    # Registration did not disturb the existing agents.
    assert {"morgan_ecfp4_1024", "maccs_keys_167", "rdkit_physchem_descriptors"} <= vector_ids


def test_registry_lookup_returns_the_agent():
    assert isinstance(registry.get(AGENT_ID), SmilesTfidfAgent)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_agents_endpoint_lists_the_agent(client):
    payload = client.get("/agents").json()
    entry = next(item for item in payload if item["id"] == AGENT_ID)
    assert entry["value_type"] == "continuous"
    assert entry["output_structure"] == "vector"
    assert entry["output_dim"] == 4096


@production_artifact
def test_agents_endpoint_exposes_4096_unique_feature_names(client):
    payload = client.get("/agents").json()
    entry = next(item for item in payload if item["id"] == AGENT_ID)
    names = entry["feature_names"]
    assert names is not None
    assert len(names) == 4096
    assert len(set(names)) == 4096


@production_artifact
def test_compute_returns_a_4096_dimensional_vector(client):
    response = client.post(
        "/features/compute",
        json={"smiles": ["CC(=O)Oc1ccccc1C(=O)O"], "agent_ids": [AGENT_ID]},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert result["error"] is None

    feature = result["features"][0]
    assert feature["agent_id"] == AGENT_ID
    assert feature["output_structure"] == "vector"
    assert feature["dim"] == 4096
    assert len(feature["values"]) == 4096
    assert all(isinstance(value, float) for value in feature["values"][:16])


@production_artifact
def test_an_all_oov_molecule_returns_4096_zeros_and_no_error(client):
    """valid=True, no error, and a full-length zero vector -- distinct from
    a representation failure, which returns features=[] with an error."""
    response = client.post(
        "/features/compute", json={"smiles": ["[Xe]"], "agent_ids": [AGENT_ID]}
    )
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert result["error"] is None

    feature = result["features"][0]
    assert feature["dim"] == 4096
    assert len(feature["values"]) == 4096
    assert all(value == 0.0 for value in feature["values"])


def test_an_invalid_smiles_is_still_reported_as_invalid(client):
    response = client.post(
        "/features/compute", json={"smiles": ["not-a-molecule"], "agent_ids": [AGENT_ID]}
    )
    result = response.json()["results"][0]
    assert result["valid"] is False
    assert result["error"]
    assert result["features"] == []


@production_artifact
def test_the_agent_composes_with_other_agents_in_one_request(client):
    response = client.post(
        "/features/compute",
        json={
            "smiles": ["CCO"],
            "agent_ids": ["morgan_ecfp4_1024", AGENT_ID, "selfies_sequence"],
        },
    )
    result = response.json()["results"][0]
    assert result["valid"] is True
    by_agent = {feature["agent_id"]: feature for feature in result["features"]}
    assert by_agent["morgan_ecfp4_1024"]["dim"] == 1024
    assert by_agent[AGENT_ID]["dim"] == 4096
    assert by_agent["selfies_sequence"]["output_structure"] == "sequence"


# ---------------------------------------------------------------------------
# real production artifact -- read-only smoke test
# ---------------------------------------------------------------------------


SMOKE_MOLECULES = [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
    ("stereochemical (L-alanine)", "C[C@H](N)C(=O)O"),
    ("disconnected (sodium acetate)", "CC(=O)[O-].[Na+]"),
]


@production_artifact
@pytest.mark.parametrize(("label", "smiles"), SMOKE_MOLECULES)
def test_production_artifact_smoke(label, smiles):
    from rdkit import Chem

    agent = SmilesTfidfAgent()
    vector = agent.compute(Chem.MolFromSmiles(smiles))

    assert vector.shape == (4096,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))
    assert vector.any(), f"{label} unexpectedly produced a zero vector"
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, rel=1e-6)


@production_artifact
def test_production_artifact_is_only_read(tmp_path):
    """The agent must never write to, or touch the mtimes of, the frozen
    artifact it consumes."""
    from rdkit import Chem

    before = {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(PRODUCTION_DIRECTORY.iterdir())
    }
    agent = SmilesTfidfAgent()
    for _, smiles in SMOKE_MOLECULES:
        agent.compute(Chem.MolFromSmiles(smiles))
    after = {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(PRODUCTION_DIRECTORY.iterdir())
    }
    assert before == after


@production_artifact
def test_production_artifact_load_is_amortized_not_per_molecule():
    """Guards against an implementation that reloads or rescans per call:
    the first compute pays for the load, later ones must be far cheaper."""
    from rdkit import Chem

    agent = SmilesTfidfAgent()
    molecules = [Chem.MolFromSmiles(smiles) for _, smiles in SMOKE_MOLECULES] * 8

    started = time.perf_counter()
    agent.load()
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for molecule in molecules:
        agent.compute(molecule)
    per_molecule = (time.perf_counter() - started) / len(molecules)

    # Deliberately loose: this is a "did we accidentally reload the
    # artifact every time" check, not a benchmark.
    assert per_molecule < load_seconds / 4
    assert per_molecule < 0.05
