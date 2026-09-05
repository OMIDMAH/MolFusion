import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools

from molfusion_backend.agents.avalon import AVALON_BIT_FLAGS, AvalonFingerprintAgent

ETHANOL = "CCO"
BENZENE = "c1ccccc1"


@pytest.fixture
def agent() -> AvalonFingerprintAgent:
    return AvalonFingerprintAgent()


def test_output_shape_is_1024(agent):
    mol = Chem.MolFromSmiles(ETHANOL)
    result = agent.compute(mol)
    assert result.shape == (1024,)


def test_output_dtype_is_uint8(agent):
    mol = Chem.MolFromSmiles(ETHANOL)
    result = agent.compute(mol)
    assert result.dtype == np.uint8


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
    """Cross-check against a direct pyAvalonTools.GetAvalonFP call (nBits=1024,
    bitFlags=AVALON_BIT_FLAGS — the same pinned value the agent uses).

    No bit positions are hand-picked here: the reference fingerprint is
    computed independently with RDKit's own Avalon API and compared
    element-wise against the agent's output.
    """
    mol = Chem.MolFromSmiles(ETHANOL)

    reference_bitvect = pyAvalonTools.GetAvalonFP(
        mol, nBits=1024, bitFlags=AVALON_BIT_FLAGS
    )
    reference_arr = np.zeros((1024,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(reference_bitvect, reference_arr)

    agent_arr = agent.compute(mol)
    np.testing.assert_array_equal(agent_arr, reference_arr)


def test_matches_direct_rdkit_reference_api_for_benzene(agent):
    mol = Chem.MolFromSmiles(BENZENE)

    reference_bitvect = pyAvalonTools.GetAvalonFP(
        mol, nBits=1024, bitFlags=AVALON_BIT_FLAGS
    )
    reference_arr = np.zeros((1024,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(reference_bitvect, reference_arr)

    agent_arr = agent.compute(mol)
    np.testing.assert_array_equal(agent_arr, reference_arr)


def test_pinned_bit_flags_matches_rdkit_current_default(agent):
    """AVALON_BIT_FLAGS is pinned to RDKit's own current default for
    GetAvalonFP, so that this agent's output stays bit-for-bit reproducible
    even if a future RDKit release changes that default. This test detects
    such a drift: if it ever fails, GetAvalonFP's default has changed and
    AVALON_BIT_FLAGS should be reviewed (not blindly updated) before bumping
    the agent's version.
    """
    assert AVALON_BIT_FLAGS == 15761407

    mol = Chem.MolFromSmiles(ETHANOL)
    default_bitvect = pyAvalonTools.GetAvalonFP(mol, nBits=1024)
    default_arr = np.zeros((1024,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(default_bitvect, default_arr)

    agent_arr = agent.compute(mol)
    np.testing.assert_array_equal(agent_arr, default_arr)


def test_mol_none_raises_clear_exception(agent):
    with pytest.raises(ValueError, match="mol=None"):
        agent.compute(None)


def test_agent_metadata():
    agent = AvalonFingerprintAgent()
    assert agent.id == "avalon_1024"
    assert agent.output_dim == 1024
    assert agent.requires_3d is False
    assert agent.version
