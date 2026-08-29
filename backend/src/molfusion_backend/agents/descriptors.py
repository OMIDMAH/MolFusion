import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from molfusion_backend.agents.base import FeatureAgent

# Deterministic descriptor ordering, taken directly from the installed RDKit
# version's descriptor registry (Descriptors._descList). This is NOT
# hardcoded to a fixed count: it reflects whatever descriptors the
# installed rdkit package ships, so output_dim tracks the RDKit version.
DESCRIPTOR_NAMES: tuple[str, ...] = tuple(name for name, _fn in Descriptors._descList)

# Sentinel used when a single descriptor fails to compute for a molecule.
# RDKit's CalcMolDescriptors() lets us supply this via `missingVal`; we use
# NaN rather than a fabricated numeric value (e.g. 0.0) so that failures
# are always distinguishable from genuine results. Callers must check for
# NaN explicitly if they need to detect missing descriptors.
MISSING_DESCRIPTOR_VALUE = float("nan")


class PhysicochemicalDescriptorAgent(FeatureAgent):
    """Full set of RDKit physicochemical descriptors (Descriptors.CalcMolDescriptors).

    Ordering: descriptors are emitted in the order given by
    `Descriptors._descList` at import time (captured in DESCRIPTOR_NAMES),
    which is also the order `CalcMolDescriptors` populates its result dict
    in. This ordering is deterministic for a given RDKit version but is
    NOT guaranteed to be stable across RDKit versions/releases, since RDKit
    may add, remove, or reorder descriptors between versions.

    Missing values: if an individual descriptor fails to compute for a
    given molecule, its slot is set to NaN (see MISSING_DESCRIPTOR_VALUE).
    No fabricated substitute value is ever used.
    """

    id = "rdkit_physchem_descriptors"
    name = "RDKit Physicochemical Descriptors"
    version = "1.0.0"
    output_dim = len(DESCRIPTOR_NAMES)
    requires_3d = False
    value_type = "continuous"

    descriptor_names = DESCRIPTOR_NAMES

    def compute(self, mol: Chem.Mol) -> np.ndarray:
        if mol is None:
            raise ValueError(
                f"{self.id}: compute() received mol=None; a valid RDKit Mol is required."
            )

        values = Descriptors.CalcMolDescriptors(
            mol, missingVal=MISSING_DESCRIPTOR_VALUE, silent=True
        )
        ordered = [values[name] for name in self.descriptor_names]
        return np.array(ordered, dtype=np.float64)
