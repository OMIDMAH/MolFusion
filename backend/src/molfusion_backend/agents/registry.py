from molfusion_backend.agents.base import FeatureAgent


class DuplicateAgentError(Exception):
    """Raised when registering an agent whose ID is already registered."""


class UnknownAgentError(Exception):
    """Raised when looking up an agent ID that has not been registered."""


class AgentRegistry:
    """Explicit registry of FeatureAgent instances, keyed by agent ID.

    No reflection or module scanning: agents must be registered explicitly
    via `register()`.
    """

    def __init__(self) -> None:
        self._agents: dict[str, FeatureAgent] = {}

    def register(self, agent: FeatureAgent) -> None:
        if agent.id in self._agents:
            raise DuplicateAgentError(
                f"Agent with id '{agent.id}' is already registered."
            )
        self._agents[agent.id] = agent

    def get(self, agent_id: str) -> FeatureAgent:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise UnknownAgentError(
                f"No agent registered with id '{agent_id}'. "
                f"Known ids: {sorted(self._agents)}"
            ) from None

    def agents(self) -> list["FeatureAgent"]:
        """Every registered agent instance, in registration order.

        Static identity -- which agents exist -- deliberately kept separate
        from whether each can currently run. Availability is computed by
        `molfusion_backend.agents.availability`, not stored here, so a
        transient prerequisite failure never changes what the registry
        contains.
        """
        return list(self._agents.values())

    def list_agents(self) -> list[dict[str, object]]:
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "version": agent.version,
                "output_dim": agent.output_dim,
                "requires_3d": agent.requires_3d,
                "value_type": agent.value_type,
                "output_structure": agent.output_structure,
                # Optional, per-feature names (e.g. descriptor/fragment
                # names) for agents that expose them via a `feature_names`
                # attribute. None for agents with no such metadata (e.g.
                # fingerprints, where individual bit positions are not
                # named).
                "feature_names": getattr(agent, "feature_names", None),
            }
            for agent in self._agents.values()
        ]


# Production registry. No feature agents are registered in Phase 1.
registry = AgentRegistry()
