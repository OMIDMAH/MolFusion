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


def test_agents_contains_exactly_the_six_production_agents(client):
    response = client.get("/agents")
    ids = {agent["id"] for agent in response.json()}
    assert ids == {
        "morgan_ecfp4_1024",
        "maccs_keys_167",
        "rdkit_physchem_descriptors",
        "avalon_1024",
        "erg_reduced_graph_315",
        "rdkit_fragment_descriptors",
    }


def test_agents_dimensions(client):
    response = client.get("/agents")
    dims_by_id = {agent["id"]: agent["output_dim"] for agent in response.json()}
    assert dims_by_id["morgan_ecfp4_1024"] == 1024
    assert dims_by_id["maccs_keys_167"] == 167
    assert dims_by_id["rdkit_physchem_descriptors"] == 217
    assert dims_by_id["avalon_1024"] == 1024
    assert dims_by_id["erg_reduced_graph_315"] == agent_registry.get(
        "erg_reduced_graph_315"
    ).output_dim
    assert dims_by_id["rdkit_fragment_descriptors"] == agent_registry.get(
        "rdkit_fragment_descriptors"
    ).output_dim


def test_agents_value_type_distinguishes_binary_count_and_continuous(client):
    response = client.get("/agents")
    types_by_id = {agent["id"]: agent["value_type"] for agent in response.json()}
    assert types_by_id["morgan_ecfp4_1024"] == "binary"
    assert types_by_id["maccs_keys_167"] == "binary"
    assert types_by_id["avalon_1024"] == "binary"
    assert types_by_id["rdkit_physchem_descriptors"] == "continuous"
    assert types_by_id["erg_reduced_graph_315"] == "continuous"
    assert types_by_id["rdkit_fragment_descriptors"] == "count"


def test_agents_descriptor_feature_names_are_exposed(client):
    response = client.get("/agents")
    agents_by_id = {agent["id"]: agent for agent in response.json()}

    descriptor_agent = agents_by_id["rdkit_physchem_descriptors"]
    assert descriptor_agent["feature_names"] is not None
    assert len(descriptor_agent["feature_names"]) == 217
    assert "MolWt" in descriptor_agent["feature_names"]

    assert agents_by_id["morgan_ecfp4_1024"]["feature_names"] is None
    assert agents_by_id["maccs_keys_167"]["feature_names"] is None


def test_agents_fragment_feature_names_are_exposed(client):
    response = client.get("/agents")
    agents_by_id = {agent["id"]: agent for agent in response.json()}

    fragment_agent = agents_by_id["rdkit_fragment_descriptors"]
    assert fragment_agent["feature_names"] is not None
    assert len(fragment_agent["feature_names"]) == fragment_agent["output_dim"]
    assert "fr_benzene" in fragment_agent["feature_names"]


def test_agents_metadata_matches_live_registry(client):
    response = client.get("/agents")
    api_agents = {agent["id"]: agent for agent in response.json()}
    for meta in agent_registry.list_agents():
        api_agent = api_agents[meta["id"]]
        assert api_agent["name"] == meta["name"]
        assert api_agent["version"] == meta["version"]
        assert api_agent["output_dim"] == meta["output_dim"]
        assert api_agent["requires_3d"] == meta["requires_3d"]
        assert api_agent["value_type"] == meta["value_type"]


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


ALL_AGENT_IDS = [
    "morgan_ecfp4_1024",
    "maccs_keys_167",
    "rdkit_physchem_descriptors",
    "avalon_1024",
    "erg_reduced_graph_315",
    "rdkit_fragment_descriptors",
]

# Aspirin, not ethanol: ErG's default parameters legitimately produce an
# all-zero vector for a molecule as small/simple as ethanol (see
# tests/test_erg_agent.py), so a richer molecule is used wherever the test
# needs to observe actual nonzero/fractional ErG values.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_compute_ethanol_with_all_agents_returns_six_feature_vectors(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ALL_AGENT_IDS}
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert len(result["features"]) == 6


def test_compute_dimensions_match_actual_outputs(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ALL_AGENT_IDS}
    )
    features = {f["agent_id"]: f for f in response.json()["results"][0]["features"]}
    assert features["morgan_ecfp4_1024"]["dim"] == 1024
    assert features["maccs_keys_167"]["dim"] == 167
    assert features["rdkit_physchem_descriptors"]["dim"] == 217
    assert features["avalon_1024"]["dim"] == 1024
    assert features["erg_reduced_graph_315"]["dim"] == agent_registry.get(
        "erg_reduced_graph_315"
    ).output_dim
    assert features["rdkit_fragment_descriptors"]["dim"] == agent_registry.get(
        "rdkit_fragment_descriptors"
    ).output_dim


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


def test_compute_avalon_alone_returns_expected_feature_vector(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["avalon_1024"]}
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert len(result["features"]) == 1

    feature = result["features"][0]
    assert feature["agent_id"] == "avalon_1024"
    assert feature["agent_version"] == agent_registry.get("avalon_1024").version
    assert feature["dim"] == 1024
    assert len(feature["values"]) == 1024
    assert set(feature["values"]).issubset({0.0, 1.0})


def test_compute_matches_direct_avalon_agent_computation(client):
    """Cross-check the API output for ethanol against direct AvalonFingerprintAgent.compute()."""
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["avalon_1024"]}
    )
    api_values = response.json()["results"][0]["features"][0]["values"]

    mol = Chem.MolFromSmiles(ETHANOL)
    agent = agent_registry.get("avalon_1024")
    direct_values = [float(v) for v in agent.compute(mol).tolist()]

    assert api_values == direct_values


def test_compute_erg_alone_returns_expected_feature_vector(client):
    response = client.post(
        "/features/compute",
        json={"smiles": [ASPIRIN], "agent_ids": ["erg_reduced_graph_315"]},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert len(result["features"]) == 1

    feature = result["features"][0]
    erg_agent = agent_registry.get("erg_reduced_graph_315")
    assert feature["agent_id"] == "erg_reduced_graph_315"
    assert feature["agent_version"] == erg_agent.version
    assert feature["dim"] == erg_agent.output_dim
    assert len(feature["values"]) == erg_agent.output_dim


def test_compute_erg_values_are_continuous_not_binary(client):
    """Guards against any accidental binarization/rounding in the API layer."""
    response = client.post(
        "/features/compute",
        json={"smiles": [ASPIRIN], "agent_ids": ["erg_reduced_graph_315"]},
    )
    values = response.json()["results"][0]["features"][0]["values"]
    assert any(v not in (0.0, 1.0) for v in values)


def test_compute_matches_direct_erg_agent_computation(client):
    """Cross-check the API output for aspirin against direct
    ErgReducedGraphAgent.compute(), verifying fractional values survive the
    HTTP/JSON round trip exactly (no rounding, truncation, or coercion)."""
    response = client.post(
        "/features/compute",
        json={"smiles": [ASPIRIN], "agent_ids": ["erg_reduced_graph_315"]},
    )
    api_values = response.json()["results"][0]["features"][0]["values"]

    mol = Chem.MolFromSmiles(ASPIRIN)
    agent = agent_registry.get("erg_reduced_graph_315")
    direct_values = [float(v) for v in agent.compute(mol).tolist()]

    assert api_values == direct_values


def test_compute_fragments_alone_returns_expected_feature_vector(client):
    response = client.post(
        "/features/compute",
        json={"smiles": [ASPIRIN], "agent_ids": ["rdkit_fragment_descriptors"]},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert len(result["features"]) == 1

    feature = result["features"][0]
    fragment_agent = agent_registry.get("rdkit_fragment_descriptors")
    assert feature["agent_id"] == "rdkit_fragment_descriptors"
    assert feature["agent_version"] == fragment_agent.version
    assert feature["dim"] == fragment_agent.output_dim
    assert len(feature["values"]) == fragment_agent.output_dim


def test_compute_fragment_values_preserve_integer_counts(client):
    """Guards against any accidental binarization/rounding of counts in the API layer."""
    response = client.post(
        "/features/compute",
        json={"smiles": [ASPIRIN], "agent_ids": ["rdkit_fragment_descriptors"]},
    )
    values = response.json()["results"][0]["features"][0]["values"]

    # All values are non-negative integers (represented as JSON floats with
    # no fractional part), and at least one count exceeds 1 for aspirin
    # (verified independently: fr_C_O == 2), ruling out a 0/1 assumption.
    assert all(v == int(v) and v >= 0 for v in values)
    assert any(v > 1 for v in values)


def test_compute_matches_direct_fragment_agent_computation(client):
    """Cross-check the API output for aspirin against direct
    FragmentDescriptorAgent.compute(), verifying integer counts survive the
    HTTP/JSON round trip exactly."""
    response = client.post(
        "/features/compute",
        json={"smiles": [ASPIRIN], "agent_ids": ["rdkit_fragment_descriptors"]},
    )
    api_values = response.json()["results"][0]["features"][0]["values"]

    mol = Chem.MolFromSmiles(ASPIRIN)
    agent = agent_registry.get("rdkit_fragment_descriptors")
    direct_values = [float(v) for v in agent.compute(mol).tolist()]

    assert api_values == direct_values


def test_compute_descriptor_output_contains_ethanol_molecular_weight(client):
    response = client.post(
        "/features/compute",
        json={"smiles": [ETHANOL], "agent_ids": ["rdkit_physchem_descriptors"]},
    )
    values = response.json()["results"][0]["features"][0]["values"]

    agent = agent_registry.get("rdkit_physchem_descriptors")
    mol_wt_index = agent.feature_names.index("MolWt")
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
