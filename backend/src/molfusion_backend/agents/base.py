from abc import ABC, abstractmethod
from typing import Literal, Union

import numpy as np
from rdkit import Chem

# "binary": every value is 0 or 1 (bit-vector fingerprints: Morgan, MACCS,
#   Avalon). "count": non-negative integer counts (RDKit fragment
#   descriptors). "continuous": real-valued (RDKit physicochemical
#   descriptors, ErG). "categorical": discrete symbols from a vocabulary,
#   e.g. SELFIES tokens -- not a number, so not binary/count/continuous.
# Exposed via GET /agents so API/UI consumers can tell them apart without
# guessing from the agent name.
ValueType = Literal["binary", "count", "continuous", "categorical"]

# "vector": a fixed-size, ordered, numeric array (np.ndarray) -- every
#   agent through Phase 5C. "sequence": a variable-length, ordered sequence
#   of string tokens (e.g. SELFIES) -- has no fixed output_dim and is not a
#   numeric array at all.
OutputStructure = Literal["vector", "sequence"]

# What FeatureAgent.compute() may return, depending on output_structure:
# a fixed-size numeric array for "vector" agents, or an immutable ordered
# sequence of string tokens for "sequence" agents. Using `tuple[str, ...]`
# (rather than `list[str]`) signals immutability: a computed token sequence
# is a result, not a buffer callers should mutate in place.
AgentOutput = Union[np.ndarray, "tuple[str, ...]"]


class FeatureAgent(ABC):
    """Base interface for a molecular feature-extraction agent."""

    id: str
    name: str
    version: str
    # None only for agents whose output_structure is "sequence": sequence
    # length is a property of each molecule's result, not a fixed property
    # of the agent (see output_structure). Every "vector" agent must set
    # this to a concrete, RDKit-derived integer.
    output_dim: int | None
    requires_3d: bool = False
    value_type: ValueType
    output_structure: OutputStructure

    @abstractmethod
    def compute(self, mol: Chem.Mol) -> AgentOutput:
        """Compute this agent's representation for a single molecule.

        Returns an np.ndarray for "vector" agents (shape == (output_dim,)),
        or a tuple[str, ...] token sequence for "sequence" agents (length
        varies per molecule; never padded, truncated, or numerically
        encoded here).
        """
        raise NotImplementedError
