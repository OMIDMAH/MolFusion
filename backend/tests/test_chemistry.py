import pytest
from rdkit import Chem

from molfusion_backend.chemistry import (
    CANONICAL_SMILES_NORMALIZATION_ID,
    canonical_smiles_from_mol,
    canonicalize_smiles,
    parse_smiles,
)

ETHANOL = "CCO"
ETHANOL_REORDERED = "OCC"
BENZENE = "c1ccccc1"
BENZENE_KEKULE = "C1=CC=CC=C1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CHIRAL = "C[C@H](O)c1ccccc1"
CHIRAL_INVERTED = "C[C@@H](O)c1ccccc1"
ISOTOPIC = "[13CH3]CO"
AMMONIUM = "[NH4+]"
SODIUM_ACETATE = "CC(=O)[O-].[Na+]"
DOUBLE_BOND_STEREO = "C/C=C/C"


# ---------------------------------------------------------------------------
# parse_smiles (pre-existing behaviour, pinned here alongside the new helpers)
# ---------------------------------------------------------------------------


def test_parse_smiles_returns_mol_and_no_error_for_valid_input():
    mol, error = parse_smiles(ETHANOL)
    assert mol is not None
    assert error is None


def test_parse_smiles_returns_error_message_for_invalid_input():
    mol, error = parse_smiles("not_a_molecule")
    assert mol is None
    assert "not_a_molecule" in error


# ---------------------------------------------------------------------------
# canonicalize_smiles: core contract
# ---------------------------------------------------------------------------


def test_ordinary_aliphatic_molecule():
    assert canonicalize_smiles(ETHANOL) == "CCO"


def test_aromatic_molecule_stays_aromatic_not_kekulized():
    """The contract pins kekuleSmiles=False, so an aromatic ring has exactly
    one canonical serialization -- the lowercase aromatic form."""
    canonical = canonicalize_smiles(BENZENE)
    assert canonical == "c1ccccc1"
    assert "c" in canonical


def test_kekule_input_canonicalizes_to_the_same_aromatic_form():
    assert canonicalize_smiles(BENZENE_KEKULE) == canonicalize_smiles(BENZENE)


def test_equivalent_smiles_orderings_canonicalize_identically():
    """The whole point of normalization: two spellings of one molecule must
    collapse to one string, or corpus deduplication is meaningless."""
    assert canonicalize_smiles(ETHANOL) == canonicalize_smiles(ETHANOL_REORDERED)


def test_repeated_calls_are_deterministic():
    first = canonicalize_smiles(ASPIRIN)
    for _ in range(5):
        assert canonicalize_smiles(ASPIRIN) == first


def test_canonicalizing_an_already_canonical_string_is_idempotent():
    once = canonicalize_smiles(ASPIRIN)
    assert canonicalize_smiles(once) == once


def test_empty_string_is_a_successful_empty_result_not_an_error():
    """RDKit parses "" as the empty molecule. That must stay distinguishable
    from unparseable input, which raises."""
    assert canonicalize_smiles("") == ""


# ---------------------------------------------------------------------------
# canonicalize_smiles: what must be preserved
# ---------------------------------------------------------------------------


def test_preserves_stereochemistry():
    canonical = canonicalize_smiles(CHIRAL)
    assert "@" in canonical


def test_distinguishes_opposite_stereoisomers():
    """Stereochemistry removal would silently merge two different molecules
    into one corpus entry; assert it does not happen."""
    assert canonicalize_smiles(CHIRAL) != canonicalize_smiles(CHIRAL_INVERTED)


def test_preserves_double_bond_stereochemistry():
    canonical = canonicalize_smiles(DOUBLE_BOND_STEREO)
    assert "/" in canonical or "\\" in canonical


def test_preserves_isotopic_information():
    assert "13" in canonicalize_smiles(ISOTOPIC)


def test_preserves_formal_charge():
    assert "+" in canonicalize_smiles(AMMONIUM)


def test_preserves_disconnected_components():
    """No salt stripping, no largest-fragment selection: both components of
    sodium acetate must survive."""
    canonical = canonicalize_smiles(SODIUM_ACETATE)
    assert "." in canonical
    assert len(canonical.split(".")) == 2
    assert "[Na+]" in canonical
    assert "[O-]" in canonical


def test_does_not_neutralize_charges():
    canonical = canonicalize_smiles(SODIUM_ACETATE)
    assert "+" in canonical and "-" in canonical


def test_preserves_explicit_bracket_expressions():
    canonical = canonicalize_smiles("[13CH3][15NH2]")
    assert "[13CH3]" in canonical
    assert "[15NH2]" in canonical


def test_does_not_lowercase_aliphatic_atoms():
    """Case is chemically meaningful: "C" and "c" are different atoms."""
    canonical = canonicalize_smiles("CCC")
    assert canonical == "CCC"


def test_does_not_canonicalize_tautomers():
    """Two tautomers are distinct molecules to RDKit's SMILES writer. This
    helper serializes; it does not standardize."""
    keto = canonicalize_smiles("CC(=O)CC(=O)C")
    enol = canonicalize_smiles("CC(O)=CC(=O)C")
    assert keto != enol


# ---------------------------------------------------------------------------
# canonicalize_smiles: failure modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid", ["not_a_molecule", "C(C", "[Xy]", "c1ccccc", "CC(=O"]
)
def test_invalid_smiles_raises_value_error(invalid):
    with pytest.raises(ValueError):
        canonicalize_smiles(invalid)


def test_invalid_smiles_error_message_names_the_offending_input():
    with pytest.raises(ValueError, match="not_a_molecule"):
        canonicalize_smiles("not_a_molecule")


def test_non_string_input_raises_value_error():
    with pytest.raises(ValueError):
        canonicalize_smiles(42)


# ---------------------------------------------------------------------------
# canonical_smiles_from_mol
# ---------------------------------------------------------------------------


def test_from_mol_matches_the_direct_rdkit_call():
    """Cross-check against RDKit itself rather than an invented expected
    string, so the test tracks the installed RDKit's canonical output."""
    for smiles in (ETHANOL, BENZENE, ASPIRIN, CHIRAL, ISOTOPIC, SODIUM_ACETATE):
        mol = Chem.MolFromSmiles(smiles)
        expected = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True, kekuleSmiles=False
        )
        assert canonical_smiles_from_mol(mol) == expected


def test_from_mol_agrees_with_canonicalize_smiles():
    for smiles in (ETHANOL, BENZENE, ASPIRIN, CHIRAL, SODIUM_ACETATE):
        mol = Chem.MolFromSmiles(smiles)
        assert canonical_smiles_from_mol(mol) == canonicalize_smiles(smiles)


def test_from_mol_rejects_none():
    with pytest.raises(ValueError, match="mol=None"):
        canonical_smiles_from_mol(None)


# ---------------------------------------------------------------------------
# Contract identifier
# ---------------------------------------------------------------------------


def test_normalization_contract_id_is_pinned():
    assert CANONICAL_SMILES_NORMALIZATION_ID == "rdkit_canonical_isomeric_smiles_v1"
