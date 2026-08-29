import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

from molfusion_backend.agents import registry as agent_registry
from molfusion_backend.main import app

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
INVALID = "INVALID"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_allows_cross_origin_requests_from_local_dev_frontend(client):
    """The Vite dev server (a different origin/port) must be able to call this API."""
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status(client):
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /agents
# ---------------------------------------------------------------------------


def test_agents_returns_200(client):
    response = client.get("/agents")
    assert response.status_code == 200


def test_agents_contains_exactly_the_three_production_agents(client):
    response = client.get("/agents")
    ids = {agent["id"] for agent in response.json()}
    assert ids == {"morgan_ecfp4_1024", "maccs_keys_167", "rdkit_physchem_descriptors"}


def test_agents_dimensions(client):
    response = client.get("/agents")
    dims_by_id = {agent["id"]: agent["output_dim"] for agent in response.json()}
    assert dims_by_id["morgan_ecfp4_1024"] == 1024
    assert dims_by_id["maccs_keys_167"] == 167
    assert dims_by_id["rdkit_physchem_descriptors"] == 217


def test_agents_descriptor_feature_names_are_exposed(client):
    response = client.get("/agents")
    agents_by_id = {agent["id"]: agent for agent in response.json()}

    descriptor_agent = agents_by_id["rdkit_physchem_descriptors"]
    assert descriptor_agent["feature_names"] is not None
    assert len(descriptor_agent["feature_names"]) == 217
    assert "MolWt" in descriptor_agent["feature_names"]

    assert agents_by_id["morgan_ecfp4_1024"]["feature_names"] is None
    assert agents_by_id["maccs_keys_167"]["feature_names"] is None


def test_agents_metadata_matches_live_registry(client):
    response = client.get("/agents")
    api_agents = {agent["id"]: agent for agent in response.json()}
    for meta in agent_registry.list_agents():
        api_agent = api_agents[meta["id"]]
        assert api_agent["name"] == meta["name"]
        assert api_agent["version"] == meta["version"]
        assert api_agent["output_dim"] == meta["output_dim"]
        assert api_agent["requires_3d"] == meta["requires_3d"]


# ---------------------------------------------------------------------------
# /molecules/validate
# ---------------------------------------------------------------------------


def test_validate_ethanol_is_valid(client):
    response = client.post("/molecules/validate", json={"smiles": [ETHANOL]})
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert result["error"] is None


def test_validate_benzene_is_valid(client):
    response = client.post("/molecules/validate", json={"smiles": [BENZENE]})
    result = response.json()["results"][0]
    assert result["valid"] is True


def test_validate_invalid_smiles_returns_false_with_error(client):
    response = client.post("/molecules/validate", json={"smiles": [INVALID]})
    result = response.json()["results"][0]
    assert result["valid"] is False
    assert result["error"]


def test_validate_batch_preserves_input_order(client):
    smiles = [ETHANOL, INVALID, BENZENE]
    response = client.post("/molecules/validate", json={"smiles": smiles})
    results = response.json()["results"]
    assert [r["smiles"] for r in results] == smiles
    assert [r["valid"] for r in results] == [True, False, True]


def test_validate_empty_smiles_list_is_rejected(client):
    response = client.post("/molecules/validate", json={"smiles": []})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /features/compute
# ---------------------------------------------------------------------------


ALL_AGENT_IDS = ["morgan_ecfp4_1024", "maccs_keys_167", "rdkit_physchem_descriptors"]


def test_compute_ethanol_with_all_agents_returns_three_feature_vectors(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ALL_AGENT_IDS}
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert len(result["features"]) == 3


def test_compute_dimensions_match_actual_outputs(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ALL_AGENT_IDS}
    )
    features = {f["agent_id"]: f for f in response.json()["results"][0]["features"]}
    assert features["morgan_ecfp4_1024"]["dim"] == 1024
    assert features["maccs_keys_167"]["dim"] == 167
    assert features["rdkit_physchem_descriptors"]["dim"] == 217


def test_compute_values_length_equals_dim(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ALL_AGENT_IDS}
    )
    for feature in response.json()["results"][0]["features"]:
        assert len(feature["values"]) == feature["dim"]


def test_compute_agent_versions_are_present(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ALL_AGENT_IDS}
    )
    for feature in response.json()["results"][0]["features"]:
        assert feature["agent_version"]


def test_compute_morgan_output_is_binary(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["morgan_ecfp4_1024"]}
    )
    values = response.json()["results"][0]["features"][0]["values"]
    assert set(values).issubset({0.0, 1.0})


def test_compute_maccs_output_is_binary(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["maccs_keys_167"]}
    )
    values = response.json()["results"][0]["features"][0]["values"]
    assert set(values).issubset({0.0, 1.0})


def test_compute_descriptor_output_contains_ethanol_molecular_weight(client):
    response = client.post(
        "/features/compute",
        json={"smiles": [ETHANOL], "agent_ids": ["rdkit_physchem_descriptors"]},
    )
    values = response.json()["results"][0]["features"][0]["values"]

    agent = agent_registry.get("rdkit_physchem_descriptors")
    mol_wt_index = agent.descriptor_names.index("MolWt")
    assert values[mol_wt_index] == pytest.approx(46.069, abs=0.01)


def test_compute_matches_direct_agent_computation(client):
    """Cross-check the API output for ethanol against direct agent.compute()."""
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["morgan_ecfp4_1024"]}
    )
    api_values = response.json()["results"][0]["features"][0]["values"]

    mol = Chem.MolFromSmiles(ETHANOL)
    agent = agent_registry.get("morgan_ecfp4_1024")
    direct_values = [float(v) for v in agent.compute(mol).tolist()]

    assert api_values == direct_values


def test_compute_invalid_molecule_has_no_features(client):
    response = client.post(
        "/features/compute", json={"smiles": [INVALID], "agent_ids": ALL_AGENT_IDS}
    )
    result = response.json()["results"][0]
    assert result["valid"] is False
    assert result["error"]
    assert result["features"] == []


def test_compute_invalid_molecule_does_not_crash_batch(client):
    response = client.post(
        "/features/compute",
        json={"smiles": [ETHANOL, INVALID, BENZENE], "agent_ids": ["maccs_keys_167"]},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["valid"] for r in results] == [True, False, True]
    assert len(results[0]["features"]) == 1
    assert results[1]["features"] == []
    assert len(results[2]["features"]) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_compute_unknown_agent_id_returns_400(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["not_a_real_agent"]}
    )
    assert response.status_code == 400
    assert "not_a_real_agent" in response.json()["detail"]


def test_compute_empty_smiles_list_is_rejected(client):
    response = client.post(
        "/features/compute", json={"smiles": [], "agent_ids": ALL_AGENT_IDS}
    )
    assert response.status_code == 422


def test_compute_empty_agent_ids_is_rejected(client):
    response = client.post("/features/compute", json={"smiles": [ETHANOL], "agent_ids": []})
    assert response.status_code == 422
