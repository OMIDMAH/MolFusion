from molfusion_backend.agents import registry
from molfusion_backend.agents.avalon import AvalonFingerprintAgent
from molfusion_backend.agents.descriptors import PhysicochemicalDescriptorAgent
from molfusion_backend.agents.erg import ErgReducedGraphAgent
from molfusion_backend.agents.fragments import FragmentDescriptorAgent
from molfusion_backend.agents.maccs import MACCSKeysAgent
from molfusion_backend.agents.morgan import MorganFingerprintAgent
from molfusion_backend.agents.selfies_agent import SelfiesSequenceAgent
from molfusion_backend.agents.smiles_tfidf import SmilesTfidfAgent

VECTOR_AGENT_CLASSES = (
    MorganFingerprintAgent,
    MACCSKeysAgent,
    PhysicochemicalDescriptorAgent,
    AvalonFingerprintAgent,
    ErgReducedGraphAgent,
    FragmentDescriptorAgent,
    SmilesTfidfAgent,
)

EXPECTED_IDS = {cls.id for cls in VECTOR_AGENT_CLASSES} | {SelfiesSequenceAgent.id}


def test_production_registry_contains_exactly_the_expected_agents():
    listed_ids = {entry["id"] for entry in registry.list_agents()}
    assert listed_ids == EXPECTED_IDS
    assert len(EXPECTED_IDS) == 8


def test_ids_are_unique():
    listed_ids = [entry["id"] for entry in registry.list_agents()]
    assert len(listed_ids) == len(set(listed_ids))


def test_metadata_dimensions_are_correct():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}

    assert metadata_by_id[MorganFingerprintAgent.id]["output_dim"] == 1024
    assert metadata_by_id[MACCSKeysAgent.id]["output_dim"] == 167
    assert metadata_by_id[AvalonFingerprintAgent.id]["output_dim"] == 1024
    assert (
        metadata_by_id[ErgReducedGraphAgent.id]["output_dim"] == ErgReducedGraphAgent.output_dim
    )
    assert (
        metadata_by_id[FragmentDescriptorAgent.id]["output_dim"]
        == FragmentDescriptorAgent.output_dim
    )
    assert (
        metadata_by_id[PhysicochemicalDescriptorAgent.id]["output_dim"]
        == PhysicochemicalDescriptorAgent.output_dim
    )
    assert metadata_by_id[SelfiesSequenceAgent.id]["output_dim"] is None
    assert metadata_by_id[SmilesTfidfAgent.id]["output_dim"] == 4096


def test_agents_are_retrievable_by_id():
    assert isinstance(registry.get(MorganFingerprintAgent.id), MorganFingerprintAgent)
    assert isinstance(registry.get(MACCSKeysAgent.id), MACCSKeysAgent)
    assert isinstance(
        registry.get(PhysicochemicalDescriptorAgent.id), PhysicochemicalDescriptorAgent
    )
    assert isinstance(registry.get(AvalonFingerprintAgent.id), AvalonFingerprintAgent)
    assert isinstance(registry.get(ErgReducedGraphAgent.id), ErgReducedGraphAgent)
    assert isinstance(registry.get(FragmentDescriptorAgent.id), FragmentDescriptorAgent)
    assert isinstance(registry.get(SelfiesSequenceAgent.id), SelfiesSequenceAgent)


def test_descriptor_agent_exposes_feature_names():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    feature_names = metadata_by_id[PhysicochemicalDescriptorAgent.id]["feature_names"]
    assert feature_names is not None
    assert len(feature_names) == PhysicochemicalDescriptorAgent.output_dim
    assert "MolWt" in feature_names


def test_fragment_agent_exposes_feature_names():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    feature_names = metadata_by_id[FragmentDescriptorAgent.id]["feature_names"]
    assert feature_names is not None
    assert len(feature_names) == FragmentDescriptorAgent.output_dim
    assert "fr_benzene" in feature_names


def test_fingerprint_agents_have_no_feature_names():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    assert metadata_by_id[MorganFingerprintAgent.id]["feature_names"] is None
    assert metadata_by_id[MACCSKeysAgent.id]["feature_names"] is None
    assert metadata_by_id[AvalonFingerprintAgent.id]["feature_names"] is None


def test_selfies_agent_has_no_feature_names():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    assert metadata_by_id[SelfiesSequenceAgent.id]["feature_names"] is None


def test_avalon_agent_metadata():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    avalon = metadata_by_id[AvalonFingerprintAgent.id]
    assert avalon["id"] == "avalon_1024"
    assert avalon["output_dim"] == 1024
    assert avalon["requires_3d"] is False
    assert avalon["version"]


def test_erg_agent_metadata():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    erg = metadata_by_id[ErgReducedGraphAgent.id]
    assert erg["id"] == "erg_reduced_graph_315"
    assert erg["output_dim"] == ErgReducedGraphAgent.output_dim
    assert erg["output_dim"] > 0
    assert erg["requires_3d"] is False
    assert erg["value_type"] == "continuous"
    assert erg["version"]
    assert erg["feature_names"] is None


def test_fragment_agent_metadata():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    fragments = metadata_by_id[FragmentDescriptorAgent.id]
    assert fragments["id"] == "rdkit_fragment_descriptors"
    assert fragments["output_dim"] == FragmentDescriptorAgent.output_dim
    assert fragments["output_dim"] > 0
    assert fragments["requires_3d"] is False
    assert fragments["value_type"] == "count"
    assert fragments["version"]
    assert fragments["feature_names"] is not None
    assert len(fragments["feature_names"]) == fragments["output_dim"]


def test_selfies_agent_metadata():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    selfies_meta = metadata_by_id[SelfiesSequenceAgent.id]
    assert selfies_meta["id"] == "selfies_sequence"
    assert selfies_meta["version"] == "1.0.0"
    assert selfies_meta["output_dim"] is None
    assert selfies_meta["requires_3d"] is False
    assert selfies_meta["value_type"] == "categorical"
    assert selfies_meta["output_structure"] == "sequence"
    assert selfies_meta["feature_names"] is None


def test_value_type_distinguishes_binary_count_continuous_and_categorical_agents():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}

    for binary_agent_id in (
        MorganFingerprintAgent.id,
        MACCSKeysAgent.id,
        AvalonFingerprintAgent.id,
    ):
        assert metadata_by_id[binary_agent_id]["value_type"] == "binary"

    for continuous_agent_id in (
        PhysicochemicalDescriptorAgent.id,
        ErgReducedGraphAgent.id,
    ):
        assert metadata_by_id[continuous_agent_id]["value_type"] == "continuous"

    assert metadata_by_id[FragmentDescriptorAgent.id]["value_type"] == "count"
    assert metadata_by_id[SelfiesSequenceAgent.id]["value_type"] == "categorical"


def test_output_structure_is_vector_for_all_pre_selfies_agents():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    for cls in VECTOR_AGENT_CLASSES:
        assert metadata_by_id[cls.id]["output_structure"] == "vector"


def test_output_structure_is_sequence_for_selfies():
    metadata_by_id = {entry["id"]: entry for entry in registry.list_agents()}
    assert metadata_by_id[SelfiesSequenceAgent.id]["output_structure"] == "sequence"
