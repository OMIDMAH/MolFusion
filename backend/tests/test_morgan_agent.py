import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from molfusion_backend.agents.morgan import MorganFingerprintAgent

ETHANOL = "CCO"
BENZENE = "c1ccccc1"


@pytest.fixture
def agent() -> MorganFingerprintAgent:
    return MorganFingerprintAgent()


def test_output_shape_is_1024(agent):
    mol = Chem.MolFromSmiles(ETHANOL)
    result = agent.compute(mol)
    assert result.shape == (1024,)


def test_output_is_binary(agent):
    mol = Chem.MolFromSmiles(BENZENE)
    result = agent.compute(mol)
    assert set(np.unique(result)).issubset({0, 1})


def test_same_molecule_produces_identical_fingerprint(agent):
    mol_a = Chem.MolFromSmiles(ETHANOL)
    mol_b = Chem.MolFromSmiles(ETHANOL)
    result_a = agent.compute(mol_a)
    result_b = agent.compute(mol_b)
    np.testing.assert_array_equal(result_a, result_b)


def test_benzene_and_ethanol_differ(agent):
    ethanol_fp = agent.compute(Chem.MolFromSmiles(ETHANOL))
    benzene_fp = agent.compute(Chem.MolFromSmiles(BENZENE))
    assert not np.array_equal(ethanol_fp, benzene_fp)


def test_matches_direct_rdkit_reference_api(agent):
    """Cross-check against a direct rdFingerprintGenerator call (radius=2, fpSize=1024)."""
    mol = Chem.MolFromSmiles(ETHANOL)

    reference_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    reference_bitvect = reference_generator.GetFingerprint(mol)
    reference_arr = np.zeros((1024,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(reference_bitvect, reference_arr)

    agent_arr = agent.compute(mol)
    np.testing.assert_array_equal(agent_arr, reference_arr)


def test_mol_none_raises_clear_exception(agent):
    with pytest.raises(ValueError, match="mol=None"):
        agent.compute(None)


def test_agent_metadata():
    agent = MorganFingerprintAgent()
    assert agent.id == "morgan_ecfp4_1024"
    assert agent.output_dim == 1024
    assert agent.requires_3d is False
    assert agent.version
