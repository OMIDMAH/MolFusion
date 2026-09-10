import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

from molfusion_backend.agents.base import FeatureAgent

MACCS_NUM_BITS = 167


class MACCSKeysAgent(FeatureAgent):
    """MACCS structural keys fingerprint (full 167-bit representation)."""

    id = "maccs_keys_167"
    name = "MACCS Keys (167-bit)"
    version = "1.0.0"
    output_dim = MACCS_NUM_BITS
    requires_3d = False
    value_type = "binary"
    output_structure = "vector"

    def compute(self, mol: Chem.Mol) -> np.ndarray:
        if mol is None:
            raise ValueError(
                f"{self.id}: compute() received mol=None; a valid RDKit Mol is required."
            )

        bit_vect = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((self.output_dim,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit_vect, arr)
        return arr
