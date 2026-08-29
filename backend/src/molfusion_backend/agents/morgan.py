import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from molfusion_backend.agents.base import FeatureAgent

MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 1024


class MorganFingerprintAgent(FeatureAgent):
    """Morgan (ECFP-style) circular fingerprint, radius=2, 1024 bits.

    Uses the modern rdFingerprintGenerator API (not the deprecated
    AllChem.GetMorganFingerprintAsBitVect).
    """

    id = "morgan_ecfp4_1024"
    name = "Morgan ECFP (radius=2, 1024 bits)"
    version = "1.0.0"
    output_dim = MORGAN_FP_SIZE
    requires_3d = False

    def __init__(self) -> None:
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS, fpSize=MORGAN_FP_SIZE
        )

    def compute(self, mol: Chem.Mol) -> np.ndarray:
        if mol is None:
            raise ValueError(
                f"{self.id}: compute() received mol=None; a valid RDKit Mol is required."
            )

        bit_vect = self._generator.GetFingerprint(mol)
        arr = np.zeros((self.output_dim,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit_vect, arr)
        return arr
