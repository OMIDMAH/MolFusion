import numpy as np
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs

from molfusion_backend.agents.base import FeatureAgent

# Pinned explicitly rather than relying on GetErGFingerprint's implicit
# defaults, so ErG output stays reproducible even if a future RDKit release
# changes these defaults. Values below match RDKit 2026.03.5's own current
# defaults (verified via GetErGFingerprint's C++ signature: atomTypes=0,
# fuzzIncrement=0.3, minPath=1, maxPath=15).
ERG_ATOM_TYPES = 0  # 0 = RDKit's standard pharmacophore atom-type scheme
ERG_FUZZ_INCREMENT = 0.3
ERG_MIN_PATH = 1
ERG_MAX_PATH = 15


def _compute_erg_output_dim() -> int:
    """Determine the real ErG vector length for the pinned parameters above,
    rather than hardcoding a guessed dimension.

    The dimension is fixed by (number of pharmacophore atom-type pairs) x
    (number of path-length bins from ERG_MIN_PATH to ERG_MAX_PATH); it does
    NOT depend on which molecule is passed in, only on atomTypes/minPath/
    maxPath. A trivial valid molecule is used purely to query the resulting
    array's shape once, at import time.
    """
    probe_mol = Chem.MolFromSmiles("C")
    fp = rdReducedGraphs.GetErGFingerprint(
        probe_mol,
        atomTypes=ERG_ATOM_TYPES,
        fuzzIncrement=ERG_FUZZ_INCREMENT,
        minPath=ERG_MIN_PATH,
        maxPath=ERG_MAX_PATH,
    )
    return int(fp.shape[0])


ERG_OUTPUT_DIM = _compute_erg_output_dim()


class ErgReducedGraphAgent(FeatureAgent):
    """ErG (Extended Reduced Graph) pharmacophore fingerprint.

    Uses rdkit.Chem.rdReducedGraphs.GetErGFingerprint -- RDKit's current,
    non-deprecated API for ErG fingerprints. Unlike the bit-vector
    fingerprint agents (Morgan/MACCS/Avalon), GetErGFingerprint already
    returns a NumPy float64 array directly; no DataStructs conversion is
    needed or applicable here.

    Output is continuous-valued, not binary: each dimension is a
    fuzzy-binned path-length count between a pair of pharmacophore atom
    types, so values are small non-negative floats (observed: 0.0, 0.3,
    0.6, 1.0, 1.6, ...), not bits or plain integer counts. This agent
    returns those values unmodified -- no thresholding, rounding, or
    binarization.
    """

    id = "erg_reduced_graph_315"
    name = "ErG Reduced-Graph Fingerprint"
    version = "1.0.0"
    output_dim = ERG_OUTPUT_DIM
    requires_3d = False
    value_type = "continuous"

    def compute(self, mol: Chem.Mol) -> np.ndarray:
        if mol is None:
            raise ValueError(
                f"{self.id}: compute() received mol=None; a valid RDKit Mol is required."
            )

        fp = rdReducedGraphs.GetErGFingerprint(
            mol,
            atomTypes=ERG_ATOM_TYPES,
            fuzzIncrement=ERG_FUZZ_INCREMENT,
            minPath=ERG_MIN_PATH,
            maxPath=ERG_MAX_PATH,
        )
        return np.asarray(fp, dtype=np.float64)
