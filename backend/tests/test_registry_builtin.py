from molfusion_backend.agents import registry
from molfusion_backend.agents.avalon import AvalonFingerprintAgent
from molfusion_backend.agents.descriptors import PhysicochemicalDescriptorAgent
from molfusion_backend.agents.maccs import MACCSKeysAgent
from molfusion_backend.agents.morgan import MorganFingerprintAgent

EXPECTED_IDS = {
    MorganFingerprintAgent.id,
    MACCSKeysAgent.id,
    PhysicochemicalDescriptorAgent.id,
    AvalonFingerprintAgent.id,
}


def test_production_registry_contains_exactly_the_expected_agents():
    listed_ids = {entry["id"] for entry in registry.list_agents()}
    assert listed_ids == EXPECTED_IDS


def test_ids_are_unique():
    listed_ids = [entry["id"] for entry in registry.list_agents()]
    assert len(listed_ids) == len(set(listed_ids))


def test_metadata_dimensions_are_correct():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}

    assert metadata_by_id[MorganFingerprintAgent.id]["output_dim"] == 1024
    assert metadata_by_id[MACCSKeysAgent.id]["output_dim"] == 167
    assert metadata_by_id[AvalonFingerprintAgent.id]["output_dim"] == 1024
    assert (
        metadata_by_id[PhysicochemicalDescriptorAgent.id]["output_dim"]
        == PhysicochemicalDescriptorAgent.output_dim
    )


def test_agents_are_retrievable_by_id():
    assert isinstance(registry.get(MorganFingerprintAgent.id), MorganFingerprintAgent)
    assert isinstance(registry.get(MACCSKeysAgent.id), MACCSKeysAgent)
    assert isinstance(
        registry.get(PhysicochemicalDescriptorAgent.id), PhysicochemicalDescriptorAgent
    )
    assert isinstance(registry.get(AvalonFingerprintAgent.id), AvalonFingerprintAgent)


def test_descriptor_agent_exposes_feature_names():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    feature_names = metadata_by_id[PhysicochemicalDescriptorAgent.id]["feature_names"]
    assert feature_names is not None
    assert len(feature_names) == PhysicochemicalDescriptorAgent.output_dim
    assert "MolWt" in feature_names


def test_fingerprint_agents_have_no_feature_names():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    assert metadata_by_id[MorganFingerprintAgent.id]["feature_names"] is None
    assert metadata_by_id[MACCSKeysAgent.id]["feature_names"] is None
    assert metadata_by_id[AvalonFingerprintAgent.id]["feature_names"] is None


def test_avalon_agent_metadata():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    avalon = metadata_by_id[AvalonFingerprintAgent.id]
    assert avalon["id"] == "avalon_1024"
    assert avalon["output_dim"] == 1024
    assert avalon["requires_3d"] is False
    assert avalon["version"]
