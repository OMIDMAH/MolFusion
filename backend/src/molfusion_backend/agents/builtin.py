from molfusion_backend.agents.avalon import AvalonFingerprintAgent
from molfusion_backend.agents.descriptors import PhysicochemicalDescriptorAgent
from molfusion_backend.agents.maccs import MACCSKeysAgent
from molfusion_backend.agents.morgan import MorganFingerprintAgent
from molfusion_backend.agents.registry import registry


def register_builtin_agents() -> None:
    """Explicitly register the production FeatureAgent implementations.

    Each agent class is imported and registered by name here; no
    reflection or module scanning is used to discover agents.
    """
    registry.register(MorganFingerprintAgent())
    registry.register(MACCSKeysAgent())
    registry.register(PhysicochemicalDescriptorAgent())
    registry.register(AvalonFingerprintAgent())
