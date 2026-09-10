import threading
from concurrent.futures import ThreadPoolExecutor

import selfies as sf
from rdkit import Chem
import pytest

from molfusion_backend.agents import selfies_agent
from molfusion_backend.agents.selfies_agent import (
    SELFIES_CONSTRAINT_PRESET,
    SelfiesSequenceAgent,
)

ETHANOL = "CCO"
ETHANOL_REORDERED = "OCC"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CHIRAL = "C[C@H](O)Cl"
CHARGED = "C[N+](C)(C)C"


@pytest.fixture
def agent() -> SelfiesSequenceAgent:
    return SelfiesSequenceAgent()


# ---------------------------------------------------------------------------
# Basic sequence semantics
# ---------------------------------------------------------------------------


def test_compute_returns_a_tuple_of_tokens(agent):
    mol = Chem.MolFromSmiles(ETHANOL)
    result = agent.compute(mol)
    assert isinstance(result, tuple)


def test_sequence_is_non_empty_for_valid_molecules(agent):
    for smiles in (ETHANOL, BENZENE, ASPIRIN, CHIRAL, CHARGED):
        result = agent.compute(Chem.MolFromSmiles(smiles))
        assert len(result) > 0


def test_all_tokens_are_strings(agent):
    mol = Chem.MolFromSmiles(ASPIRIN)
    result = agent.compute(mol)
    assert all(isinstance(token, str) for token in result)


def test_tokens_are_well_formed_selfies_symbols(agent):
    """Every token should be a bracketed SELFIES symbol, e.g. "[C]"."""
    mol = Chem.MolFromSmiles(ASPIRIN)
    result = agent.compute(mol)
    assert all(token.startswith("[") and token.endswith("]") for token in result)


def test_deterministic_output(agent):
    mol_a = Chem.MolFromSmiles(ASPIRIN)
    mol_b = Chem.MolFromSmiles(ASPIRIN)
    result_a = agent.compute(mol_a)
    result_b = agent.compute(mol_b)
    assert result_a == result_b


def test_mol_none_raises_clear_exception(agent):
    with pytest.raises(ValueError, match="mol=None"):
        agent.compute(None)


def test_different_molecules_produce_different_sequences(agent):
    ethanol_seq = agent.compute(Chem.MolFromSmiles(ETHANOL))
    benzene_seq = agent.compute(Chem.MolFromSmiles(BENZENE))
    assert ethanol_seq != benzene_seq


def test_never_pads_or_truncates_across_different_sized_molecules(agent):
    """Sequence length must track molecule size, not be forced to a fixed
    shape -- this is the defining difference from every vector agent."""
    small = agent.compute(Chem.MolFromSmiles(ETHANOL))
    large = agent.compute(Chem.MolFromSmiles(ASPIRIN))
    assert len(small) != len(large)


# ---------------------------------------------------------------------------
# Molecule-centric canonicalization (Section 2 requirement)
# ---------------------------------------------------------------------------


def test_equivalent_smiles_orderings_produce_identical_sequences(agent):
    """CCO and OCC describe the same molecule written in a different atom
    order; MolFusion's SELFIES agent must be molecule-centric, not
    text-centric, so both must produce the identical token sequence."""
    mol_a = Chem.MolFromSmiles(ETHANOL)
    mol_b = Chem.MolFromSmiles(ETHANOL_REORDERED)

    seq_a = agent.compute(mol_a)
    seq_b = agent.compute(mol_b)

    assert seq_a == seq_b


def test_matches_direct_rdkit_canonical_smiles_then_selfies_encoding(agent):
    """Cross-check against directly canonicalizing with RDKit and calling
    selfies.encoder()/split_selfies() -- no invented expected tokens."""
    for smiles in (ETHANOL, ETHANOL_REORDERED, BENZENE, ASPIRIN, CHIRAL, CHARGED):
        mol = Chem.MolFromSmiles(smiles)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        reference = tuple(sf.split_selfies(sf.encoder(canonical)))

        result = agent.compute(mol)

        assert result == reference


# ---------------------------------------------------------------------------
# Stereochemistry and charge preservation
# ---------------------------------------------------------------------------


def test_chiral_center_token_present(agent):
    """C[C@H](O)Cl has one chiral center; verify the encoded sequence
    contains a stereo-marked carbon token (as actually produced by RDKit +
    selfies for this molecule, not an invented token string)."""
    mol = Chem.MolFromSmiles(CHIRAL)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    expected_stereo_token = next(
        tok for tok in sf.split_selfies(sf.encoder(canonical)) if "@" in tok
    )

    result = agent.compute(mol)

    assert expected_stereo_token in result


def test_charged_atom_token_present(agent):
    """C[N+](C)(C)C has a quaternary ammonium cation; verify the encoded
    sequence contains a charge-marked nitrogen token."""
    mol = Chem.MolFromSmiles(CHARGED)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    expected_charge_token = next(
        tok for tok in sf.split_selfies(sf.encoder(canonical)) if "+" in tok
    )

    result = agent.compute(mol)

    assert expected_charge_token in result


# ---------------------------------------------------------------------------
# Round-trip scientific validation (Section 10)
# ---------------------------------------------------------------------------


def _round_trip_canonical_smiles(mol: Chem.Mol) -> str:
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    selfies_string = sf.encoder(canonical)
    decoded_smiles = sf.decoder(selfies_string)
    decoded_mol = Chem.MolFromSmiles(decoded_smiles)
    assert decoded_mol is not None, "decoded SMILES must itself be RDKit-valid"
    return Chem.MolToSmiles(decoded_mol, canonical=True, isomericSmiles=True)


@pytest.mark.parametrize(
    "smiles", [ETHANOL, ETHANOL_REORDERED, BENZENE, ASPIRIN, CHARGED]
)
def test_round_trip_preserves_molecular_identity(smiles):
    """Mol -> canonical SMILES -> SELFIES -> decoder -> Mol -> canonical
    SMILES must describe the same molecule. We compare RDKit-canonicalized
    SMILES (molecular identity), not literal string equality -- aromatic vs.
    Kekule notation may legitimately differ while the molecule is the same.
    """
    mol = Chem.MolFromSmiles(smiles)
    original_canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    round_tripped_canonical = _round_trip_canonical_smiles(mol)

    assert round_tripped_canonical == original_canonical


def test_round_trip_preserves_stereochemistry_for_chiral_molecule():
    """Stereochemistry must survive the round trip where SELFIES supports it
    -- verified by comparing isomeric canonical SMILES (which encodes
    stereo descriptors), not just molecular connectivity."""
    mol = Chem.MolFromSmiles(CHIRAL)
    original_canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    round_tripped_canonical = _round_trip_canonical_smiles(mol)

    assert round_tripped_canonical == original_canonical
    assert "@" in round_tripped_canonical, "chiral marker must survive the round trip"


# ---------------------------------------------------------------------------
# Semantic-constraint reproducibility policy (Section 3 requirement)
#
# Every test in this section saves whatever constraints were active before
# it ran and restores them in a `finally` block, so no test leaks mutated
# global SELFIES state into any other test regardless of pass/fail/raise.
# ---------------------------------------------------------------------------

# A real molecule RDKit parses successfully but that violates SELFIES'
# "default" constraint preset (iodine trichloride: default caps I at
# valence 1), verified directly against the installed selfies package --
# not invented. Used to exercise the EncoderError / restore-on-exception
# paths.
HYPERVALENT_IODINE = "Cl[I](Cl)Cl"


def test_pinned_constraint_preset_is_default():
    assert SELFIES_CONSTRAINT_PRESET == "default"


def test_encoding_output_is_identical_regardless_of_ambient_constraint_state(agent):
    """selfies.encoder()'s own docs state token output for a successfully
    encoded molecule is deterministic and constraint-independent. Verify
    this directly: compute the same molecule under two different *ambient*
    global constraint states (simulating some other code having mutated
    them) and confirm the agent's output is identical both times, because
    it temporarily pins its own policy for the duration of the call rather
    than trusting whatever happens to be active."""
    mol = Chem.MolFromSmiles(ASPIRIN)
    saved = sf.get_semantic_constraints()
    try:
        sf.set_semantic_constraints(sf.get_preset_constraints("default"))
        result_under_default = agent.compute(mol)

        sf.set_semantic_constraints(sf.get_preset_constraints("hypervalent"))
        result_under_hypervalent_ambient_state = agent.compute(mol)

        assert result_under_default == result_under_hypervalent_ambient_state
    finally:
        sf.set_semantic_constraints(saved)


def test_compute_produces_output_under_molfusions_pinned_default_policy(agent):
    """Even when the caller's ambient constraints are a non-default preset
    beforehand, the *result* must reflect MolFusion's pinned "default"
    policy -- verified against a direct reference encode performed under
    an explicitly-forced default preset."""
    mol = Chem.MolFromSmiles(ASPIRIN)
    saved = sf.get_semantic_constraints()
    try:
        sf.set_semantic_constraints(sf.get_preset_constraints("octet_rule"))

        result = agent.compute(mol)

        sf.set_semantic_constraints(sf.get_preset_constraints(SELFIES_CONSTRAINT_PRESET))
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        reference = tuple(sf.split_selfies(sf.encoder(canonical)))
        assert result == reference
    finally:
        sf.set_semantic_constraints(saved)


def test_compute_restores_the_callers_previous_constraints_exactly(agent):
    """After compute() returns, the process-global constraint state must be
    restored to exactly what the caller had active beforehand -- not left
    on MolFusion's pinned "default" policy, and not left on whatever
    compute() used internally."""
    saved = sf.get_semantic_constraints()
    try:
        non_default = sf.get_preset_constraints("octet_rule")
        sf.set_semantic_constraints(non_default)

        agent.compute(Chem.MolFromSmiles(ETHANOL))

        assert sf.get_semantic_constraints() == non_default
    finally:
        sf.set_semantic_constraints(saved)


def test_compute_restores_previous_constraints_even_when_encoding_raises(agent):
    """The restore step must run even when selfies.encoder() raises --
    i.e. it belongs in a finally block, not merely after a successful
    encode."""
    saved = sf.get_semantic_constraints()
    try:
        non_default = sf.get_preset_constraints("hypervalent")
        sf.set_semantic_constraints(non_default)

        mol = Chem.MolFromSmiles(HYPERVALENT_IODINE)
        with pytest.raises(ValueError, match="SELFIES"):
            agent.compute(mol)

        assert sf.get_semantic_constraints() == non_default
    finally:
        sf.set_semantic_constraints(saved)


def test_encoder_error_is_normalized_to_a_clear_value_error_with_chaining(agent):
    """A molecule that violates the pinned default constraints must raise
    a clear ValueError (not a raw selfies.EncoderError), with the original
    exception preserved via `raise ... from exc` for debuggability."""
    mol = Chem.MolFromSmiles(HYPERVALENT_IODINE)

    with pytest.raises(ValueError) as exc_info:
        agent.compute(mol)

    assert "SELFIES" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, sf.EncoderError)


# ---------------------------------------------------------------------------
# Thread safety (Section 2 requirement)
# ---------------------------------------------------------------------------


def test_module_uses_a_reentrant_lock_to_guard_the_constraint_critical_section():
    assert isinstance(selfies_agent._CONSTRAINT_LOCK, type(threading.RLock()))


def test_concurrent_compute_calls_all_produce_correct_deterministic_results():
    """A reasonable concurrency smoke test: many threads calling compute()
    at once on real molecules must all complete without raising and must
    each return the correct (deterministic, reference-matching) sequence --
    i.e. the lock serializes the constraint-mutating critical section
    without corrupting any individual call's output."""
    agent = SelfiesSequenceAgent()
    molecules = [ETHANOL, BENZENE, ASPIRIN, CHIRAL, CHARGED]

    # Compute reference sequences up front, sequentially, so the reference
    # computation itself is never racing against the concurrent agent
    # calls below.
    references = {
        smiles: tuple(
            sf.split_selfies(
                sf.encoder(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True))
            )
        )
        for smiles in molecules
    }

    def compute_for(smiles: str) -> tuple[str, tuple[str, ...]]:
        mol = Chem.MolFromSmiles(smiles)
        return smiles, agent.compute(mol)

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(compute_for, molecules * 8))

    for smiles, result in outcomes:
        assert result == references[smiles]


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------


def test_agent_metadata():
    agent = SelfiesSequenceAgent()
    assert agent.id == "selfies_sequence"
    assert agent.output_dim is None
    assert agent.requires_3d is False
    assert agent.value_type == "categorical"
    assert agent.output_structure == "sequence"
    assert agent.feature_names is None
    assert agent.version
