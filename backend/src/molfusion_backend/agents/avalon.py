import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools

from molfusion_backend.agents.base import FeatureAgent

AVALON_FP_SIZE = 1024

# pyAvalonTools.GetAvalonFP's own default as of RDKit 2026.03.5 (verified via
# its C++ signature). Pinned explicitly here — rather than relying on the
# function's implicit default — so this agent's output stays bit-for-bit
# reproducible even if a future RDKit release changes that default.
AVALON_BIT_FLAGS = 15761407


class AvalonFingerprintAgent(FeatureAgent):
    """Avalon fingerprint, 1024 bits.

    Uses rdkit.Avalon.pyAvalonTools.GetAvalonFP, the only RDKit-provided
    path to Avalon fingerprints: unlike Morgan/AtomPair/TopologicalTorsion,
    Avalon has no rdFingerprintGenerator-based generator in this RDKit
    version, so pyAvalonTools is the current (not deprecated) API.
    """

    id = "avalon_1024"
    name = "Avalon Fingerprint (1024 bits)"
    version = "1.0.0"
    output_dim = AVALON_FP_SIZE
    requires_3d = False
    value_type = "binary"
    output_structure = "vector"

    def compute(self, mol: Chem.Mol) -> np.ndarray:
        if mol is None:
            raise ValueError(
                f"{self.id}: compute() received mol=None; a valid RDKit Mol is required."
            )

        bit_vect = pyAvalonTools.GetAvalonFP(
            mol, nBits=self.output_dim, bitFlags=AVALON_BIT_FLAGS
        )
        arr = np.zeros((self.output_dim,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit_vect, arr)
        return arr
