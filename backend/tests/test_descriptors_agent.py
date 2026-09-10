import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from molfusion_backend.agents.descriptors import (
    DESCRIPTOR_NAMES,
    PhysicochemicalDescriptorAgent,
)

ETHANOL = "CCO"
BENZENE = "c1ccccc1"


@pytest.fixture
def agent() -> PhysicochemicalDescriptorAgent:
    return PhysicochemicalDescriptorAgent()


def test_output_dimension_equals_actual_descriptor_count(agent):
    expected_count = len(Descriptors._descList)
    assert agent.output_dim == expected_count
    assert len(DESCRIPTOR_NAMES) == expected_count

    mol = Chem.MolFromSmiles(ETHANOL)
    result = agent.compute(mol)
    assert result.shape == (expected_count,)


def test_ethanol_molecular_weight(agent):
    mol = Chem.MolFromSmiles(ETHANOL)
    result = agent.compute(mol)
    mol_wt = result[agent.feature_names.index("MolWt")]
    assert mol_wt == pytest.approx(46.069, abs=0.01)


def test_benzene_molecular_weight(agent):
    mol = Chem.MolFromSmiles(BENZENE)
    result = agent.compute(mol)
    mol_wt = result[agent.feature_names.index("MolWt")]
    assert mol_wt == pytest.approx(78.114, abs=0.01)


def test_benzene_has_zero_rotatable_bonds(agent):
    mol = Chem.MolFromSmiles(BENZENE)
    result = agent.compute(mol)
    rotatable = result[agent.feature_names.index("NumRotatableBonds")]
    assert rotatable == 0


def test_deterministic_values(agent):
    mol_a = Chem.MolFromSmiles(ETHANOL)
    mol_b = Chem.MolFromSmiles(ETHANOL)
    result_a = agent.compute(mol_a)
    result_b = agent.compute(mol_b)
    np.testing.assert_array_equal(result_a, result_b)


def test_no_nan_or_inf_for_valid_molecules(agent):
    """NaN is the documented sentinel for a descriptor that fails to compute;
    for ordinary valid molecules like ethanol/benzene, no descriptor should
    fail, so no NaN/Inf should appear."""
    for smiles in (ETHANOL, BENZENE):
        mol = Chem.MolFromSmiles(smiles)
        result = agent.compute(mol)
        assert not np.isnan(result).any(), f"unexpected NaN for {smiles}"
        assert not np.isinf(result).any(), f"unexpected Inf for {smiles}"


def test_matches_direct_rdkit_reference_api(agent):
    mol = Chem.MolFromSmiles(ETHANOL)
    reference = Descriptors.CalcMolDescriptors(mol, missingVal=float("nan"), silent=True)
    reference_arr = np.array(
        [reference[name] for name in agent.feature_names], dtype=np.float64
    )
    agent_arr = agent.compute(mol)
    np.testing.assert_array_equal(agent_arr, reference_arr)


def test_mol_none_raises_clear_exception(agent):
    with pytest.raises(ValueError, match="mol=None"):
        agent.compute(None)


def test_agent_metadata():
    agent = PhysicochemicalDescriptorAgent()
    assert agent.id == "rdkit_physchem_descriptors"
    assert agent.output_dim == len(Descriptors._descList)
    assert agent.requires_3d is False
    assert agent.version
