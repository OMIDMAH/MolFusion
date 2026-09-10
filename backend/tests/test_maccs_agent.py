import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

from molfusion_backend.agents.maccs import MACCSKeysAgent

ETHANOL = "CCO"
BENZENE = "c1ccccc1"


@pytest.fixture
def agent() -> MACCSKeysAgent:
    return MACCSKeysAgent()


def test_output_shape_is_167(agent):
    mol = Chem.MolFromSmiles(ETHANOL)
    result = agent.compute(mol)
    assert result.shape == (167,)


def test_output_is_binary(agent):
    mol = Chem.MolFromSmiles(BENZENE)
    result = agent.compute(mol)
    assert set(np.unique(result)).issubset({0, 1})


def test_deterministic_output(agent):
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
    mol = Chem.MolFromSmiles(ETHANOL)

    reference_bitvect = MACCSkeys.GenMACCSKeys(mol)
    reference_arr = np.zeros((167,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(reference_bitvect, reference_arr)

    agent_arr = agent.compute(mol)
    np.testing.assert_array_equal(agent_arr, reference_arr)


def test_mol_none_raises_clear_exception(agent):
    with pytest.raises(ValueError, match="mol=None"):
        agent.compute(None)


def test_agent_metadata():
    agent = MACCSKeysAgent()
    assert agent.id == "maccs_keys_167"
    assert agent.output_dim == 167
    assert agent.requires_3d is False
    assert agent.version
