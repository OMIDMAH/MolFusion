import numpy as np
from rdkit import Chem
from rdkit.Chem import Fragments

from molfusion_backend.agents.base import FeatureAgent

# Explicitly pinned, alphabetically sorted snapshot of RDKit's public
# fr_* substructure-count descriptors (rdkit.Chem.Fragments), as observed
# in RDKit 2026.03.5 (85 functions).
#
# Where these come from: each fr_* function is generated dynamically at
# RDKit import time from a SMARTS pattern library
# (RDConfig.RDDataDir/FragmentDescriptors.csv) -- see rdkit/Chem/Fragments.py
# `_LoadPatterns`. Every one of them is `_CountMatches(mol, pattern,
# unique=True) -> len(mol.GetSubstructMatches(pattern, uniquify=True))`,
# i.e. a non-negative integer count of substructure matches, never a
# binary indicator or a float. Because that CSV-driven set/order is not a
# stable contract from RDKit itself, MolFusion pins a literal tuple here
# rather than deriving it from dir()/Fragments.fns at import time.
#
# Ordering policy: alphabetically sorted names, frozen as a literal tuple.
#
# Reproducibility: if a future RDKit upgrade adds, removes, or renames a
# fragment descriptor, MolFusion's output dimensionality and feature
# ordering must NOT silently change. tests/test_fragment_agent.py includes
# a drift-detection test comparing this pinned tuple against the live
# installed RDKit's fr_* set and fails loudly on any mismatch -- the set
# below must then be reviewed and updated deliberately (with a version
# bump), not auto-expanded.
FRAGMENT_NAMES = (
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
)


class FragmentDescriptorAgent(FeatureAgent):
    """RDKit substructure/fragment count descriptors (rdkit.Chem.Fragments.fr_*).

    Each value is a non-negative integer count of SMARTS substructure
    matches (uniquified) -- not a binary indicator and not a continuous
    value. A molecule with two matching carboxylic-acid groups yields 2,
    not 1 or True; this agent preserves that count exactly.

    Feature names are RDKit's own function names (e.g. "fr_Al_COO"), not
    prettified labels, so results stay traceable back to the exact RDKit
    API that produced them.
    """

    id = "rdkit_fragment_descriptors"
    name = "RDKit Fragment Descriptors"
    version = "1.0.0"
    output_dim = len(FRAGMENT_NAMES)
    requires_3d = False
    value_type = "count"

    feature_names = FRAGMENT_NAMES

    def compute(self, mol: Chem.Mol) -> np.ndarray:
        if mol is None:
            raise ValueError(
                f"{self.id}: compute() received mol=None; a valid RDKit Mol is required."
            )

        values = [getattr(Fragments, name)(mol) for name in FRAGMENT_NAMES]
        return np.array(values, dtype=np.int32)
