import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Fragments

from molfusion_backend.agents.fragments import FRAGMENT_NAMES, FragmentDescriptorAgent

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ACETIC_ACID = "CC(=O)O"
ACETAMIDE = "CC(=O)N"
ANILINE = "Nc1ccccc1"

ALL_MOLECULES = {
    "ethanol": ETHANOL,
    "benzene": BENZENE,
    "aspirin": ASPIRIN,
    "acetic_acid": ACETIC_ACID,
    "acetamide": ACETAMIDE,
    "aniline": ANILINE,
}


@pytest.fixture
def agent() -> FragmentDescriptorAgent:
    return FragmentDescriptorAgent()


# ---------------------------------------------------------------------------
# Basic shape/dtype/value-semantics tests
# ---------------------------------------------------------------------------


def test_output_is_ndarray(agent):
    mol = Chem.MolFromSmiles(ASPIRIN)
    result = agent.compute(mol)
    assert isinstance(result, np.ndarray)


def test_output_shape_equals_declared_output_dim(agent):
    mol = Chem.MolFromSmiles(ASPIRIN)
    result = agent.compute(mol)
    assert result.shape == (agent.output_dim,)
    assert agent.output_dim == len(FRAGMENT_NAMES)


def test_output_dtype_is_integer(agent):
    """RDKit's fr_* functions return native Python `int` (substructure
    match counts), never float -- verified directly against
    rdkit.Chem.Fragments before choosing this dtype."""
    mol = Chem.MolFromSmiles(ASPIRIN)
    result = agent.compute(mol)
    assert np.issubdtype(result.dtype, np.integer)


def test_all_values_are_finite(agent):
    for smiles in ALL_MOLECULES.values():
        result = agent.compute(Chem.MolFromSmiles(smiles))
        assert np.all(np.isfinite(result))


def test_all_values_are_non_negative(agent):
    for smiles in ALL_MOLECULES.values():
        result = agent.compute(Chem.MolFromSmiles(smiles))
        assert np.all(result >= 0)


def test_values_are_integer_valued(agent):
    for smiles in ALL_MOLECULES.values():
        result = agent.compute(Chem.MolFromSmiles(smiles))
        np.testing.assert_array_equal(result, result.astype(int))


def test_deterministic_output(agent):
    mol_a = Chem.MolFromSmiles(ASPIRIN)
    mol_b = Chem.MolFromSmiles(ASPIRIN)
    result_a = agent.compute(mol_a)
    result_b = agent.compute(mol_b)
    np.testing.assert_array_equal(result_a, result_b)


def test_mol_none_raises_clear_exception(agent):
    with pytest.raises(ValueError, match="mol=None"):
        agent.compute(None)


def test_different_molecules_produce_different_vectors(agent):
    vectors = {
        label: agent.compute(Chem.MolFromSmiles(smiles))
        for label, smiles in ALL_MOLECULES.items()
    }
    labels = list(vectors)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            assert not np.array_equal(vectors[a], vectors[b]), f"{a} and {b} did not differ"


# ---------------------------------------------------------------------------
# Cross-checks against direct RDKit computation (no invented values)
# ---------------------------------------------------------------------------


def test_matches_direct_rdkit_reference_api_for_every_molecule(agent):
    """For every reference molecule, every output index must equal calling
    the corresponding rdkit.Chem.Fragments.fr_* function directly -- not
    just a sample of indices."""
    for smiles in ALL_MOLECULES.values():
        mol = Chem.MolFromSmiles(smiles)
        agent_arr = agent.compute(mol)
        reference = np.array(
            [getattr(Fragments, name)(mol) for name in FRAGMENT_NAMES], dtype=np.int32
        )
        np.testing.assert_array_equal(agent_arr, reference)


# ---------------------------------------------------------------------------
# Known chemical motifs -- values below were obtained by directly querying
# rdkit.Chem.Fragments for these molecules first; none are invented.
# ---------------------------------------------------------------------------


def _value_at(agent: FragmentDescriptorAgent, mol: Chem.Mol, feature_name: str) -> int:
    return int(agent.compute(mol)[agent.feature_names.index(feature_name)])


def test_ethanol_has_one_aliphatic_hydroxyl(agent):
    mol = Chem.MolFromSmiles(ETHANOL)
    assert _value_at(agent, mol, "fr_Al_OH") == 1


def test_benzene_has_one_benzene_ring(agent):
    mol = Chem.MolFromSmiles(BENZENE)
    assert _value_at(agent, mol, "fr_benzene") == 1


def test_aspirin_has_aromatic_carboxylic_acid_ester_and_benzene_ring(agent):
    mol = Chem.MolFromSmiles(ASPIRIN)
    assert _value_at(agent, mol, "fr_Ar_COO") == 1
    assert _value_at(agent, mol, "fr_COO") == 1
    assert _value_at(agent, mol, "fr_ester") == 1
    assert _value_at(agent, mol, "fr_benzene") == 1


def test_acetic_acid_has_aliphatic_carboxylic_acid(agent):
    mol = Chem.MolFromSmiles(ACETIC_ACID)
    assert _value_at(agent, mol, "fr_Al_COO") == 1
    assert _value_at(agent, mol, "fr_COO") == 1


def test_acetamide_has_primary_amide(agent):
    mol = Chem.MolFromSmiles(ACETAMIDE)
    assert _value_at(agent, mol, "fr_amide") == 1
    assert _value_at(agent, mol, "fr_priamide") == 1


def test_aniline_has_aniline_and_benzene_ring(agent):
    mol = Chem.MolFromSmiles(ANILINE)
    assert _value_at(agent, mol, "fr_aniline") == 1
    assert _value_at(agent, mol, "fr_benzene") == 1
    assert _value_at(agent, mol, "fr_NH2") == 1


def test_agent_metadata():
    agent = FragmentDescriptorAgent()
    assert agent.id == "rdkit_fragment_descriptors"
    assert agent.output_dim == len(FRAGMENT_NAMES)
    assert agent.requires_3d is False
    assert agent.value_type == "count"
    assert agent.version


# ---------------------------------------------------------------------------
# Reproducibility / drift protection
# ---------------------------------------------------------------------------


def test_fragment_names_has_no_duplicates():
    assert len(FRAGMENT_NAMES) == len(set(FRAGMENT_NAMES))


def test_output_dim_matches_fragment_names_length():
    assert FragmentDescriptorAgent.output_dim == len(FRAGMENT_NAMES)


def test_every_fragment_name_is_a_real_callable_rdkit_fragments_function():
    for name in FRAGMENT_NAMES:
        fn = getattr(Fragments, name, None)
        assert fn is not None, f"{name} does not exist on rdkit.Chem.Fragments"
        assert callable(fn), f"{name} exists but is not callable"


def test_pinned_set_matches_live_installed_rdkit_fr_functions():
    """Drift-detection test: if a future RDKit upgrade adds, removes, or
    renames a public fr_* function, this test must fail loudly rather than
    MolFusion silently changing its output dimensionality/ordering. On
    failure: review the new live set deliberately, update FRAGMENT_NAMES
    with an explicit code change, and bump the agent's version -- do not
    "fix" this test by blindly accepting whatever RDKit now reports.
    """
    live_names = frozenset(n for n in dir(Fragments) if n.startswith("fr_"))
    assert frozenset(FRAGMENT_NAMES) == live_names
