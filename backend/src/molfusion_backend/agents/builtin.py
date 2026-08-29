from molfusion_backend.agents.avalon import AvalonFingerprintAgent
from molfusion_backend.agents.descriptors import PhysicochemicalDescriptorAgent
from molfusion_backend.agents.erg import ErgReducedGraphAgent
from molfusion_backend.agents.fragments import FragmentDescriptorAgent
from molfusion_backend.agents.maccs import MACCSKeysAgent
from molfusion_backend.agents.morgan import MorganFingerprintAgent
from molfusion_backend.agents.registry import registry
from molfusion_backend.agents.selfies_agent import SelfiesSequenceAgent


def register_builtin_agents() -> None:
    """Explicitly register the production FeatureAgent implementations.

    Each agent class is imported and registered by name here; no
    reflection or module scanning is used to discover agents.
    """
    registry.register(MorganFingerprintAgent())
    registry.register(MACCSKeysAgent())
    registry.register(PhysicochemicalDescriptorAgent())
    registry.register(AvalonFingerprintAgent())
    registry.register(ErgReducedGraphAgent())
    registry.register(FragmentDescriptorAgent())
    registry.register(SelfiesSequenceAgent())
