import numpy as np
import pytest
from rdkit import Chem

from molfusion_backend.agents import AgentRegistry, DuplicateAgentError, UnknownAgentError
from molfusion_backend.agents.base import FeatureAgent


class DummyAgent(FeatureAgent):
    """Minimal test-only agent. Not part of the production registry."""

    id = "dummy"
    name = "Dummy Agent"
    version = "0.0.1"
    output_dim = 4
    requires_3d = False
    value_type = "continuous"

    def compute(self, mol: Chem.Mol) -> np.ndarray:
        return np.zeros(self.output_dim, dtype=float)


@pytest.fixture
def registry() -> AgentRegistry:
    return AgentRegistry()


@pytest.fixture
def dummy_agent() -> DummyAgent:
    return DummyAgent()


def test_agent_can_be_registered(registry, dummy_agent):
    registry.register(dummy_agent)
    assert dummy_agent.id in {a["id"] for a in registry.list_agents()}


def test_agent_can_be_retrieved_by_id(registry, dummy_agent):
    registry.register(dummy_agent)
    retrieved = registry.get("dummy")
    assert retrieved is dummy_agent


def test_duplicate_ids_are_rejected(registry, dummy_agent):
    registry.register(dummy_agent)
    with pytest.raises(DuplicateAgentError):
        registry.register(DummyAgent())


def test_unknown_id_raises_clear_exception(registry):
    with pytest.raises(UnknownAgentError, match="unknown-agent"):
        registry.get("unknown-agent")


def test_list_agents_returns_registered_metadata(registry, dummy_agent):
    registry.register(dummy_agent)
    agents = registry.list_agents()
    assert agents == [
        {
            "id": "dummy",
            "name": "Dummy Agent",
            "version": "0.0.1",
            "output_dim": 4,
            "requires_3d": False,
            "value_type": "continuous",
            "feature_names": None,
        }
    ]


def test_dummy_agent_compute_returns_array_with_declared_dimension(dummy_agent):
    mol = Chem.MolFromSmiles("CCO")
    result = dummy_agent.compute(mol)
    assert isinstance(result, np.ndarray)
    assert result.shape == (dummy_agent.output_dim,)
