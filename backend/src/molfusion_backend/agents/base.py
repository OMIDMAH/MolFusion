from abc import ABC, abstractmethod

import numpy as np
from rdkit import Chem


class FeatureAgent(ABC):
    """Base interface for a molecular feature-extraction agent."""

    id: str
    name: str
    version: str
    output_dim: int
    requires_3d: bool = False

    @abstractmethod
    def compute(self, mol: Chem.Mol) -> np.ndarray:
        """Compute the feature vector for a single molecule."""
        raise NotImplementedError
