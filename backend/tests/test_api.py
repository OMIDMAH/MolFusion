import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

from molfusion_backend.agents import registry as agent_registry
from molfusion_backend.agents.base import FeatureAgent
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


def test_agents_contains_exactly_the_production_agents(client):
    response = client.get("/agents")
    ids = {agent["id"] for agent in response.json()}
    assert ids == {
        "morgan_ecfp4_1024",
        "maccs_keys_167",
        "rdkit_physchem_descriptors",
        "avalon_1024",
        "erg_reduced_graph_315",
        "rdkit_fragment_descriptors",
        "selfies_sequence",
        "smiles_tfidf_4096",
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
    assert dims_by_id["selfies_sequence"] is None


def test_agents_value_type_distinguishes_binary_count_continuous_and_categorical(client):
    response = client.get("/agents")
    types_by_id = {agent["id"]: agent["value_type"] for agent in response.json()}
    assert types_by_id["morgan_ecfp4_1024"] == "binary"
    assert types_by_id["maccs_keys_167"] == "binary"
    assert types_by_id["avalon_1024"] == "binary"
    assert types_by_id["rdkit_physchem_descriptors"] == "continuous"
    assert types_by_id["erg_reduced_graph_315"] == "continuous"
    assert types_by_id["rdkit_fragment_descriptors"] == "count"
    assert types_by_id["selfies_sequence"] == "categorical"


def test_agents_output_structure_is_exposed(client):
    response = client.get("/agents")
    structures_by_id = {agent["id"]: agent["output_structure"] for agent in response.json()}
    for vector_id in (
        "morgan_ecfp4_1024",
        "maccs_keys_167",
        "rdkit_physchem_descriptors",
        "avalon_1024",
        "erg_reduced_graph_315",
        "rdkit_fragment_descriptors",
    ):
        assert structures_by_id[vector_id] == "vector"
    assert structures_by_id["selfies_sequence"] == "sequence"


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
        assert api_agent["output_structure"] == meta["output_structure"]


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
    "selfies_sequence",
]
VECTOR_AGENT_IDS = [aid for aid in ALL_AGENT_IDS if aid != "selfies_sequence"]

# Aspirin, not ethanol: ErG's default parameters legitimately produce an
# all-zero vector for a molecule as small/simple as ethanol (see
# tests/test_erg_agent.py), so a richer molecule is used wherever the test
# needs to observe actual nonzero/fractional ErG values.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_compute_ethanol_with_all_agents_returns_seven_feature_outputs(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ALL_AGENT_IDS}
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert len(result["features"]) == 7


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


def test_compute_values_length_equals_dim_for_vector_agents(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": VECTOR_AGENT_IDS}
    )
    for feature in response.json()["results"][0]["features"]:
        assert feature["output_structure"] == "vector"
        assert len(feature["values"]) == feature["dim"]


def test_compute_tokens_length_equals_length_for_sequence_agents(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["selfies_sequence"]}
    )
    feature = response.json()["results"][0]["features"][0]
    assert feature["output_structure"] == "sequence"
    assert len(feature["tokens"]) == feature["length"]


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


def test_compute_selfies_alone_returns_sequence_output(client):
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["selfies_sequence"]}
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert len(result["features"]) == 1

    feature = result["features"][0]
    selfies_agent = agent_registry.get("selfies_sequence")
    assert feature["output_structure"] == "sequence"
    assert feature["agent_id"] == "selfies_sequence"
    assert feature["agent_version"] == selfies_agent.version
    assert "values" not in feature
    assert "dim" not in feature
    assert len(feature["tokens"]) > 0
    assert feature["length"] == len(feature["tokens"])
    assert all(isinstance(tok, str) for tok in feature["tokens"])


def test_compute_matches_direct_selfies_agent_computation(client):
    """Cross-check the API output for ethanol against direct
    SelfiesSequenceAgent.compute(), verifying the token sequence survives
    the HTTP/JSON round trip exactly."""
    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": ["selfies_sequence"]}
    )
    api_tokens = response.json()["results"][0]["features"][0]["tokens"]

    mol = Chem.MolFromSmiles(ETHANOL)
    agent = agent_registry.get("selfies_sequence")
    direct_tokens = list(agent.compute(mol))

    assert api_tokens == direct_tokens


def test_compute_selfies_together_with_vector_agents(client):
    """SELFIES alongside vector agents in the same request must not crash
    and each feature must carry the correct discriminated shape."""
    response = client.post(
        "/features/compute",
        json={
            "smiles": [ETHANOL],
            "agent_ids": ["morgan_ecfp4_1024", "selfies_sequence", "rdkit_fragment_descriptors"],
        },
    )
    assert response.status_code == 200
    features = {f["agent_id"]: f for f in response.json()["results"][0]["features"]}

    assert features["morgan_ecfp4_1024"]["output_structure"] == "vector"
    assert features["rdkit_fragment_descriptors"]["output_structure"] == "vector"
    assert features["selfies_sequence"]["output_structure"] == "sequence"


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


# ---------------------------------------------------------------------------
# Generic agent-computation failure handling (any FeatureAgent, not a
# selfies_sequence-specific check): an agent.compute() raising ValueError
# for an otherwise RDKit-valid molecule must surface as a normal
# MoleculeResult(valid=True, error=..., features=[]), never an unhandled
# HTTP 500 and never valid=False -- `valid` means molecular/SMILES
# validity, not "every requested representation succeeded". An RDKit parse
# failure is the only case that gets valid=False.
# ---------------------------------------------------------------------------


class _AlwaysFailsAgent(FeatureAgent):
    """Test-only agent whose compute() always raises ValueError, used to
    prove the route's error handling is generic (keyed off exception type,
    not off any particular agent_id)."""

    id = "test_always_fails"
    name = "Test Always Fails"
    version = "0.0.1"
    output_dim = 1
    requires_3d = False
    value_type = "continuous"
    output_structure = "vector"

    def compute(self, mol):
        raise ValueError("synthetic failure for regression test")


def _patch_agent_registry_get(monkeypatch, fake_agent):
    original_get = agent_registry.get

    def patched_get(agent_id):
        if agent_id == fake_agent.id:
            return fake_agent
        return original_get(agent_id)

    monkeypatch.setattr(agent_registry, "get", patched_get)


def test_compute_generic_agent_value_error_keeps_molecule_valid(client, monkeypatch):
    """A chemically valid molecule (RDKit parses it fine) must stay
    valid=True even when the requested agent's computation fails -- the
    failure is reported via `error`, not by flipping `valid`."""
    fake_agent = _AlwaysFailsAgent()
    _patch_agent_registry_get(monkeypatch, fake_agent)

    response = client.post(
        "/features/compute", json={"smiles": [ETHANOL], "agent_ids": [fake_agent.id]}
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert "synthetic failure" in result["error"]
    assert result["features"] == []


def test_compute_generic_agent_value_error_does_not_crash_mixed_batch(client, monkeypatch):
    fake_agent = _AlwaysFailsAgent()
    _patch_agent_registry_get(monkeypatch, fake_agent)

    response = client.post(
        "/features/compute",
        json={"smiles": [ETHANOL, BENZENE], "agent_ids": [fake_agent.id]},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert all(r["valid"] is True and r["features"] == [] for r in results)


HYPERVALENT_IODINE = "Cl[I](Cl)Cl"


def test_hypervalent_iodine_is_rdkit_valid_but_selfies_rejects_it():
    """Sanity check on the fixture molecule itself, verified directly
    against the installed packages (not assumed): RDKit parses it, but
    SELFIES' default constraints reject it (I is capped at valence 1)."""
    assert Chem.MolFromSmiles(HYPERVALENT_IODINE) is not None

    import selfies as sf

    canonical = Chem.MolToSmiles(
        Chem.MolFromSmiles(HYPERVALENT_IODINE), canonical=True, isomericSmiles=True
    )
    with pytest.raises(sf.EncoderError):
        sf.encoder(canonical)


def test_compute_selfies_encoder_failure_is_reported_generically_not_as_http_500(client):
    """Real end-to-end regression: a molecule RDKit parses successfully but
    that SELFIES' pinned default constraints reject (hypervalent iodine)
    must come back as valid=True with a populated error and no features --
    not a server error, and not valid=False (the molecule itself is fine;
    only the SELFIES representation failed)."""
    response = client.post(
        "/features/compute",
        json={"smiles": [HYPERVALENT_IODINE], "agent_ids": ["selfies_sequence"]},
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is True
    assert "failed to encode molecule as SELFIES" in result["error"]
    assert result["features"] == []


def test_compute_selfies_failure_does_not_prevent_other_molecules_in_batch(client):
    """Three molecules in one request: a normal SELFIES success, a
    molecule whose SELFIES representation fails (but stays valid=True),
    and another normal SELFIES success -- the middle failure must not
    block the molecules before or after it."""
    response = client.post(
        "/features/compute",
        json={
            "smiles": [ETHANOL, HYPERVALENT_IODINE, BENZENE],
            "agent_ids": ["selfies_sequence"],
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3

    ethanol_result, iodine_result, benzene_result = results

    assert ethanol_result["valid"] is True
    assert ethanol_result["error"] is None
    assert len(ethanol_result["features"]) == 1
    assert ethanol_result["features"][0]["output_structure"] == "sequence"

    assert iodine_result["valid"] is True
    assert iodine_result["error"] is not None
    assert "SELFIES" in iodine_result["error"]
    assert iodine_result["features"] == []

    assert benzene_result["valid"] is True
    assert benzene_result["error"] is None
    assert len(benzene_result["features"]) == 1
    assert benzene_result["features"][0]["output_structure"] == "sequence"


def test_compute_genuinely_invalid_smiles_still_returns_valid_false(client):
    """Regression guarding the distinction this fix introduces: an actual
    RDKit parse failure must still be valid=False -- only an agent
    computation failure on an otherwise-parseable molecule becomes
    valid=True with error populated."""
    response = client.post(
        "/features/compute",
        json={"smiles": [INVALID], "agent_ids": ["selfies_sequence"]},
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["valid"] is False
    assert result["error"] is not None
    assert result["features"] == []
