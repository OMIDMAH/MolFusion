from molfusion_backend.agents.base import FeatureAgent
from molfusion_backend.agents.builtin import register_builtin_agents
from molfusion_backend.agents.registry import (
    AgentRegistry,
    DuplicateAgentError,
    UnknownAgentError,
    registry,
)

# Populate the production registry with the explicit set of built-in
# agents. This runs once, at first import of this package.
register_builtin_agents()

__all__ = [
    "FeatureAgent",
    "AgentRegistry",
    "DuplicateAgentError",
    "UnknownAgentError",
    "registry",
    "register_builtin_agents",
]
