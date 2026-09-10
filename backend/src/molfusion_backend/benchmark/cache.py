"""The feature-cache key contract.

Molecular representations are deterministic functions of a molecule and an
agent, so caching them across repeated splits is safe -- but only if the key
covers everything the value depends on. A cache keyed on a filename, or on
the SMILES alone, will happily serve a vector computed by a different agent
version or a superseded artifact, and nothing downstream would notice.

Phase 6A freezes the key. The store itself is Phase 6B's, and deliberately
not built here.
"""

import hashlib
from typing import Any

CACHE_KEY_VERSION = 1


def feature_cache_key(
    *,
    canonical_smiles: str,
    agent_id: str,
    agent_version: str,
    normalization_id: str,
    artifact_identity: str | None = None,
) -> str:
    """A content-addressed key covering every input the vector depends on.

    Included, and each for a reason a filename could not cover:

      canonical_smiles    the molecule, under the frozen normalization
      agent_id/version    the code that computed it
      normalization_id    the canonicalization contract that produced the
                          input string
      artifact_identity   the fitted payload for artifact-backed agents
                          (type/id/version), or None for the stateless ones

    An artifact-backed agent whose artifact is replaced by a new version
    produces different vectors from identical source, so the artifact
    identity is part of the key rather than an assumption about it.
    """
    payload = "\x1f".join(
        (
            str(CACHE_KEY_VERSION),
            normalization_id,
            agent_id,
            agent_version,
            artifact_identity or "",
            canonical_smiles,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_contract() -> dict[str, Any]:
    return {
        "cache_key_version": CACHE_KEY_VERSION,
        "digest": "sha256",
        "key_components": [
            "cache_key_version",
            "normalization_id",
            "agent_id",
            "agent_version",
            "artifact_identity (or empty for stateless agents)",
            "canonical_smiles",
        ],
        "separator": "unit separator (0x1f), so no component can impersonate another",
        "not_keyed_on": ["filename", "dataset", "split", "label", "row order"],
        "rationale": (
            "The key covers every input the cached vector depends on. A cache "
            "keyed on a filename would serve a vector computed by a different "
            "agent version or a superseded artifact with nothing downstream "
            "able to detect it."
        ),
    }


__all__ = ["CACHE_KEY_VERSION", "cache_contract", "feature_cache_key"]
