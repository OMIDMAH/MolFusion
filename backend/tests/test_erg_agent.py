import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs

from molfusion_backend.agents.erg import (
    ERG_ATOM_TYPES,
    ERG_FUZZ_INCREMENT,
    ERG_MAX_PATH,
    ERG_MIN_PATH,
    ERG_OUTPUT_DIM,
    ErgReducedGraphAgent,
)

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
# Aspirin: rich enough pharmacophoric structure (H-bond donor/acceptor,
# aromatic ring, ester) to produce a nonzero ErG vector, unlike the two
# small molecules above.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def agent() -> ErgReducedGraphAgent:
    return ErgReducedGraphAgent()


def _reference_fp(mol: Chem.Mol) -> np.ndarray:
    return rdReducedGraphs.GetErGFingerprint(
        mol,
        atomTypes=ERG_ATOM_TYPES,
        fuzzIncrement=ERG_FUZZ_INCREMENT,
        minPath=ERG_MIN_PATH,
        maxPath=ERG_MAX_PATH,
    )


def test_output_is_ndarray(agent):
    mol = Chem.MolFromSmiles(ASPIRIN)
    result = agent.compute(mol)
    assert isinstance(result, np.ndarray)


def test_output_shape_equals_declared_output_dim(agent):
    mol = Chem.MolFromSmiles(ASPIRIN)
    result = agent.compute(mol)
    assert result.shape == (agent.output_dim,)
    assert agent.output_dim == ERG_OUTPUT_DIM


def test_output_dtype_is_floating_point(agent):
    mol = Chem.MolFromSmiles(ETHANOL)
    result = agent.compute(mol)
    assert np.issubdtype(result.dtype, np.floating)


def test_output_values_are_finite(agent):
    for smiles in (ETHANOL, BENZENE, ASPIRIN):
        result = agent.compute(Chem.MolFromSmiles(smiles))
        assert np.all(np.isfinite(result))


def test_deterministic_output(agent):
    mol_a = Chem.MolFromSmiles(ASPIRIN)
    mol_b = Chem.MolFromSmiles(ASPIRIN)
    result_a = agent.compute(mol_a)
    result_b = agent.compute(mol_b)
    np.testing.assert_array_equal(result_a, result_b)


def test_different_molecules_produce_different_vectors(agent):
    """Ethanol and aspirin are structurally very different; aspirin has
    enough pharmacophoric richness to produce nonzero ErG values while
    ethanol does not (see test_small_molecules_can_yield_an_all_zero_vector
    below), so this pair reliably demonstrates differing output."""
    ethanol_fp = agent.compute(Chem.MolFromSmiles(ETHANOL))
    aspirin_fp = agent.compute(Chem.MolFromSmiles(ASPIRIN))
    assert not np.array_equal(ethanol_fp, aspirin_fp)


def test_small_molecules_can_yield_an_all_zero_vector(agent):
    """Documented real RDKit behavior, not a bug: with the pinned default
    parameters, ethanol and benzene are both too small/simple to have any
    qualifying pharmacophore atom-type pair within [minPath, maxPath], so
    GetErGFingerprint legitimately returns an all-zero vector for both.
    This is verified against RDKit directly, not assumed."""
    ethanol_fp = agent.compute(Chem.MolFromSmiles(ETHANOL))
    benzene_fp = agent.compute(Chem.MolFromSmiles(BENZENE))

    assert np.count_nonzero(_reference_fp(Chem.MolFromSmiles(ETHANOL))) == 0
    assert np.count_nonzero(_reference_fp(Chem.MolFromSmiles(BENZENE))) == 0
    assert np.count_nonzero(ethanol_fp) == 0
    assert np.count_nonzero(benzene_fp) == 0


def test_mol_none_raises_clear_exception(agent):
    with pytest.raises(ValueError, match="mol=None"):
        agent.compute(None)


def test_matches_direct_rdkit_reference_api(agent):
    """Cross-check against a direct rdReducedGraphs.GetErGFingerprint call
    using the exact same pinned parameters the agent uses. No vector
    positions or values are hand-picked; the reference is computed
    independently and compared element-wise."""
    mol = Chem.MolFromSmiles(ASPIRIN)

    reference_arr = _reference_fp(mol)
    agent_arr = agent.compute(mol)

    np.testing.assert_array_equal(agent_arr, reference_arr)


def test_matches_direct_rdkit_reference_api_for_ethanol(agent):
    mol = Chem.MolFromSmiles(ETHANOL)
    reference_arr = _reference_fp(mol)
    agent_arr = agent.compute(mol)
    np.testing.assert_array_equal(agent_arr, reference_arr)


def test_values_are_not_binarized(agent):
    """ErG produces fuzzy-binned fractional values (e.g. 0.3, 0.6, 1.6), not
    just 0/1 bits. This asserts the agent's output actually contains such
    fractional values for aspirin, and that they are preserved exactly
    (not rounded/truncated/thresholded)."""
    mol = Chem.MolFromSmiles(ASPIRIN)
    reference_arr = _reference_fp(mol)
    agent_arr = agent.compute(mol)

    non_integer_mask = reference_arr != np.round(reference_arr)
    assert non_integer_mask.any(), "expected at least one fractional ErG value for aspirin"
    np.testing.assert_array_equal(agent_arr[non_integer_mask], reference_arr[non_integer_mask])

    # Sanity: some values also exceed 1, ruling out a 0/1 binary assumption.
    assert (reference_arr > 1).any()


def test_agent_metadata():
    agent = ErgReducedGraphAgent()
    assert agent.id == "erg_reduced_graph_315"
    assert agent.output_dim == ERG_OUTPUT_DIM
    assert agent.output_dim > 0
    assert agent.requires_3d is False
    assert agent.value_type == "continuous"
    assert agent.version
