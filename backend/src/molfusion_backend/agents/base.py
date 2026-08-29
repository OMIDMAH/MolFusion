from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from rdkit import Chem

# "binary": every value is 0 or 1 (bit-vector fingerprints: Morgan, MACCS,
#   Avalon). "count": non-negative integer counts (RDKit fragment
#   descriptors). "continuous": real-valued (RDKit physicochemical
#   descriptors, ErG).
# Exposed via GET /agents so API/UI consumers can tell them apart without
# guessing from the agent name.
ValueType = Literal["binary", "count", "continuous"]


class FeatureAgent(ABC):
    """Base interface for a molecular feature-extraction agent."""

    id: str
    name: str
    version: str
    output_dim: int
    requires_3d: bool = False
    value_type: ValueType

    @abstractmethod
    def compute(self, mol: Chem.Mol) -> np.ndarray:
        """Compute the feature vector for a single molecule."""
        raise NotImplementedError
