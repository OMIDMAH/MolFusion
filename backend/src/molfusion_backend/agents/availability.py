"""Whether a FeatureAgent can currently run at all.

Availability answers one question:

    can this agent reasonably accept compute requests in this environment?

and deliberately not this one:

    will this particular molecule succeed?

The two failure kinds are genuinely different. A missing TF-IDF artifact is
a deployment fault that will defeat every molecule identically, and
discovering it 10,000 times during a 10,000-molecule batch tells the caller
nothing the first occurrence did not. A molecule SELFIES cannot encode is a
property of that molecule, and the agent is perfectly healthy. Phase 5H's
`feature_errors` covers the second; this module covers the first.

A health check therefore validates *prerequisites only*. It must never
parse a sample molecule, call `compute()`, fit anything, read the reference
corpus, or touch the network -- a probe that runs the real computation is
not a cheap precondition check, and a molecule it happened to pick could
fail for reasons that say nothing about the agent's health.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from molfusion_backend.agents.base import FeatureAgent

logger = logging.getLogger(__name__)

# Stable, machine-readable failure categories. Generic on purpose: a client
# should be able to branch on "the artifact this agent needs is missing"
# without knowing which representation the agent implements, and adding
# `tfidf_artifact_missing` would push representation knowledge into every
# consumer.
CODE_ARTIFACT_MISSING = "artifact_missing"
CODE_ARTIFACT_CHECKSUM_ERROR = "artifact_checksum_error"
CODE_ARTIFACT_METADATA_ERROR = "artifact_metadata_error"
CODE_ARTIFACT_SEMANTIC_ERROR = "artifact_semantic_error"
CODE_DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
CODE_CONFIGURATION_ERROR = "configuration_error"
# Reserved for a health check that itself raised something unexpected. The
# agent is reported unavailable rather than assumed fine, and the exception
# is logged -- an agent whose own health check is broken is not healthy, but
# it is also a bug worth seeing rather than swallowing.
CODE_HEALTH_CHECK_ERROR = "health_check_error"

AVAILABILITY_CODES = (
    CODE_ARTIFACT_MISSING,
    CODE_ARTIFACT_CHECKSUM_ERROR,
    CODE_ARTIFACT_METADATA_ERROR,
    CODE_ARTIFACT_SEMANTIC_ERROR,
    CODE_DEPENDENCY_UNAVAILABLE,
    CODE_CONFIGURATION_ERROR,
    CODE_HEALTH_CHECK_ERROR,
)


@dataclass(frozen=True)
class AgentAvailability:
    """One agent's current ability to accept compute requests.

    `code` and `message` are populated only when unavailable. The message is
    a short human-readable summary; it never carries a traceback, an
    exception repr, or an absolute filesystem path, because this value is
    returned to API clients.
    """

    available: bool
    code: str | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {"available": self.available, "code": self.code, "message": self.message}


AVAILABLE = AgentAvailability(available=True)


def unavailable(code: str, message: str) -> AgentAvailability:
    return AgentAvailability(available=False, code=code, message=message)


def check_agent(agent: "FeatureAgent") -> AgentAvailability:
    """One agent's availability, with its health check isolated.

    Isolation matters because `GET /agents` reports on every registered
    agent: one agent's broken prerequisite must not remove the other seven
    from the listing. An unexpected exception is logged and reported as
    `health_check_error` rather than silently swallowed or allowed to take
    the endpoint down.
    """
    try:
        return agent.check_availability()
    except Exception:  # noqa: BLE001 - one agent must not break the listing
        logger.exception("availability check failed for agent %r", agent.id)
        return unavailable(
            CODE_HEALTH_CHECK_ERROR,
            "The availability check for this agent failed unexpectedly.",
        )


def check_agents(agents: Iterable["FeatureAgent"]) -> dict[str, AgentAvailability]:
    """Availability for several agents, keyed by agent id, each isolated."""
    return {agent.id: check_agent(agent) for agent in agents}


__all__ = [
    "AVAILABILITY_CODES",
    "AVAILABLE",
    "AgentAvailability",
    "CODE_ARTIFACT_CHECKSUM_ERROR",
    "CODE_ARTIFACT_MISSING",
    "CODE_ARTIFACT_METADATA_ERROR",
    "CODE_ARTIFACT_SEMANTIC_ERROR",
    "CODE_CONFIGURATION_ERROR",
    "CODE_DEPENDENCY_UNAVAILABLE",
    "CODE_HEALTH_CHECK_ERROR",
    "check_agent",
    "check_agents",
    "unavailable",
]
