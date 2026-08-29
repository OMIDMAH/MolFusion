from molfusion_backend.agents import registry
from molfusion_backend.agents.descriptors import PhysicochemicalDescriptorAgent
from molfusion_backend.agents.maccs import MACCSKeysAgent
from molfusion_backend.agents.morgan import MorganFingerprintAgent

EXPECTED_IDS = {
    MorganFingerprintAgent.id,
    MACCSKeysAgent.id,
    PhysicochemicalDescriptorAgent.id,
}


def test_all_three_agents_appear_in_list_agents():
    listed_ids = {entry["id"] for entry in registry.list_agents()}
    assert EXPECTED_IDS.issubset(listed_ids)


def test_ids_are_unique():
    listed_ids = [entry["id"] for entry in registry.list_agents()]
    assert len(listed_ids) == len(set(listed_ids))


def test_metadata_dimensions_are_correct():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}

    assert metadata_by_id[MorganFingerprintAgent.id]["output_dim"] == 1024
    assert metadata_by_id[MACCSKeysAgent.id]["output_dim"] == 167
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
