import pytest
from rdkit import Chem
from rdkit.Chem import Atom, RWMol

from molfusion_backend.chemistry import canonicalize_smiles
from molfusion_backend.smiles_tokenizer import SMILES_TOKENIZER_ID, tokenize_smiles

# A deliberately broad regression set of RDKit-valid SMILES, chosen to
# exercise every lexical construct the tokenizer claims to support: the
# organic subset, two-character halogens, aromatic atoms, branches, every
# bond symbol RDKit can emit (including dative, quadruple and unspecified
# bonds), bracket atoms carrying isotopes/charges/H-counts/atom maps and
# every chirality class, ring closures, and disconnected components.
REGRESSION_SMILES = [
    # Simple aliphatics and alcohols
    "C",
    "CCO",
    "OCC",
    "CC(C)(C)C",
    # Halogens: the "Cl"/"Br" two-character cases
    "ClC(Cl)(Cl)Cl",
    "BrCCBr",
    "FC(F)(F)Br",
    "ICl",
    # Aromatics, fused rings, heteroaromatics
    "c1ccccc1",
    "c1ccc2ccccc2c1",
    "c1cc[nH]c1",
    "c1ccsc1",
    "c1ccoc1",
    "b1ccccc1",
    "c1cc[se]c1",
    "c1ccc(-c2ccccc2)cc1",
    # Multiple bonds
    "C#N",
    "N#N",
    "O=S(=O)(O)O",
    "[Re]$[Re]",
    "C~C",
    "[Cu]<-[NH3]",
    # Stereochemistry
    "C[C@H](O)c1ccccc1",
    "C[C@@H](O)c1ccccc1",
    "C/C=C/C",
    "C/C=C\\C",
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "[As@TB1](F)(Cl)(Br)(I)O",
    "[Co@OH1](F)(Cl)(Br)(I)(O)N",
    "[Pt@SP1](Cl)(Cl)(N)N",
    # Isotopes, charges, explicit H, atom maps, wildcards
    "[13CH3]CO",
    "[2H]O[2H]",
    "[NH4+]",
    "[O-2]",
    "[Fe+3]",
    "C[N+](C)(C)C",
    "[H][H]",
    "[He]",
    "*CC",
    "[CH3:1][OH:2]",
    # Disconnected components
    "CC(=O)[O-].[Na+]",
    "[Na+].[Cl-]",
    "O.O.O",
    # Ring closures, including a bridged bicyclic with reused numbers
    "C1CCCCC1",
    "C1CC2CCC1CC2",
    "C1CCC2(CC1)CCCCC2",
    # Real drug-like molecules
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Cc1ccc(S(=O)(=O)N)cc1",
    "NC(=O)c1ccccc1O",
]


def _many_open_rings_mol(atom_count: int = 450) -> Chem.Mol:
    """Build a carbon cage with well over 99 *simultaneously open* ring
    closures, which is the only way to make RDKit emit its extended
    "%(nnn)" ring-closure form rather than "%nn"."""
    rw = RWMol()
    for _ in range(atom_count):
        rw.AddAtom(Atom(6))
    for i in range(atom_count - 1):
        rw.AddBond(i, i + 1, Chem.BondType.SINGLE)
    # Nested cross-links (i <-> atom_count-1-i) all stay open at once. Stop
    # short of the chain's midpoint so a cross-link never duplicates an
    # existing chain bond.
    for i in range(0, atom_count // 2 - 2, 2):
        rw.AddBond(i, atom_count - 1 - i, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


# ---------------------------------------------------------------------------
# The fundamental invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smiles", REGRESSION_SMILES)
def test_tokens_rejoin_to_the_input_exactly(smiles):
    """The defining lossless-tokenization property, asserted on the raw
    fixture string."""
    assert "".join(tokenize_smiles(smiles)) == smiles


@pytest.mark.parametrize("smiles", REGRESSION_SMILES)
def test_canonicalize_then_tokenize_round_trips_byte_for_byte(smiles):
    """The property that actually matters downstream: RDKit-valid raw SMILES
    -> canonical SMILES -> tokens -> rejoined must reproduce the canonical
    SMILES byte for byte."""
    canonical = canonicalize_smiles(smiles)
    tokens = tokenize_smiles(canonical)
    assert "".join(tokens) == canonical


def test_no_token_is_empty():
    for smiles in REGRESSION_SMILES:
        assert all(token for token in tokenize_smiles(canonicalize_smiles(smiles)))


def test_returns_an_immutable_tuple_of_strings():
    tokens = tokenize_smiles("CC(=O)O")
    assert isinstance(tokens, tuple)
    assert all(isinstance(token, str) for token in tokens)


def test_tokenization_is_deterministic():
    first = tokenize_smiles("CC(=O)Oc1ccccc1C(=O)O")
    for _ in range(5):
        assert tokenize_smiles("CC(=O)Oc1ccccc1C(=O)O") == first


# ---------------------------------------------------------------------------
# Exact token sequences for the lexical categories
# ---------------------------------------------------------------------------


def test_simple_atoms():
    assert tokenize_smiles("CCO") == ("C", "C", "O")


def test_two_character_halogens_are_single_tokens():
    assert tokenize_smiles("ClCBr") == ("Cl", "C", "Br")


def test_uppercase_c_is_not_greedily_merged_with_a_following_aromatic_atom():
    """"Cl"/"Br" are the only two-character organic-subset symbols; "Sc",
    "Cn" etc. must stay two atoms, exactly as RDKit reads them."""
    assert tokenize_smiles("Sc1ccccc1") == (
        "S", "c", "1", "c", "c", "c", "c", "c", "1",
    )


def test_aromatic_and_aliphatic_case_is_preserved():
    assert tokenize_smiles("Cc1ccccc1") == (
        "C", "c", "1", "c", "c", "c", "c", "c", "1",
    )


def test_branches_and_double_bonds():
    assert tokenize_smiles("CC(=O)O") == ("C", "C", "(", "=", "O", ")", "O")


def test_triple_bond():
    assert tokenize_smiles("C#N") == ("C", "#", "N")


def test_quadruple_bond():
    assert tokenize_smiles("[Re]$[Re]") == ("[Re]", "$", "[Re]")


def test_unspecified_bond():
    assert tokenize_smiles("C~C") == ("C", "~", "C")


def test_explicit_single_and_aromatic_bond_symbols():
    assert tokenize_smiles("C-C") == ("C", "-", "C")
    assert tokenize_smiles("c:c") == ("c", ":", "c")


def test_dative_bonds_are_single_tokens_not_a_bond_plus_an_angle_bracket():
    assert tokenize_smiles("[NH3]->[Cu]") == ("[NH3]", "->", "[Cu]")
    assert tokenize_smiles("[Cu]<-[NH3]") == ("[Cu]", "<-", "[NH3]")


def test_directional_bonds():
    assert tokenize_smiles("C/C=C/C") == ("C", "/", "C", "=", "C", "/", "C")
    assert tokenize_smiles("C/C=C\\C") == ("C", "/", "C", "=", "C", "\\", "C")


def test_wildcard_atom():
    assert tokenize_smiles("*CC") == ("*", "C", "C")


def test_dot_separated_components():
    assert tokenize_smiles("CC(=O)[O-].[Na+]") == (
        "C", "C", "(", "=", "O", ")", "[O-]", ".", "[Na+]",
    )


# ---------------------------------------------------------------------------
# Bracket atoms stay atomic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bracket_atom",
    [
        "[NH4+]",
        "[C@@H]",
        "[C@H]",
        "[13CH3]",
        "[O-]",
        "[O-2]",
        "[nH]",
        "[se]",
        "[Fe+3]",
        "[2H]",
        "[H]",
        "[He]",
        "[CH3:1]",
        "[*:1]",
        "[Co@OH24]",
        "[As@TB8]",
        "[Pt@SP1]",
    ],
)
def test_bracket_atom_is_one_indivisible_token(bracket_atom):
    assert tokenize_smiles(bracket_atom) == (bracket_atom,)


def test_bracket_contents_are_never_split_even_when_they_look_like_other_tokens():
    """"[13CH3]" contains digits and element symbols that would otherwise be
    ring closures and atoms; inside brackets they are not tokens at all."""
    assert tokenize_smiles("[13CH3]CO") == ("[13CH3]", "C", "O")


def test_adjacent_bracket_atoms():
    assert tokenize_smiles("[Na+].[Cl-]") == ("[Na+]", ".", "[Cl-]")


# ---------------------------------------------------------------------------
# Ring closures
# ---------------------------------------------------------------------------


def test_single_digit_ring_closures():
    assert tokenize_smiles("C1CCCCC1") == (
        "C", "1", "C", "C", "C", "C", "C", "1",
    )


def test_ring_closures_are_per_digit_not_grouped():
    """"C12" opens two separate rings (1 and 2), so it must be three tokens
    -- never an atom plus the number twelve."""
    assert tokenize_smiles("C12CCCCC1CC2") == (
        "C", "1", "2", "C", "C", "C", "C", "C", "1", "C", "C", "2",
    )


def test_percent_two_digit_ring_closure():
    tokens = tokenize_smiles("C%10CCCC%10")
    assert tokens == ("C", "%10", "C", "C", "C", "C", "%10")


def test_extended_parenthesized_ring_closure_is_one_token():
    tokens = tokenize_smiles("C%(100)CCCC%(100)")
    assert tokens == ("C", "%(100)", "C", "C", "C", "C", "%(100)")


def test_extended_ring_closure_form_is_not_confused_with_a_branch():
    """"%(100)" must not tokenize as "%" + "(" + digits + ")"."""
    assert "(" not in tokenize_smiles("C%(100)CCCC%(100)")


def test_installed_rdkit_actually_emits_the_extended_ring_closure_form():
    """Guards the assumption behind the "%(nnn)" rule: if a future RDKit
    stopped emitting this form (or started emitting a different one), this
    test fails loudly rather than the tokenizer silently over-supporting a
    syntax that no longer exists."""
    smiles = Chem.MolToSmiles(_many_open_rings_mol())
    assert "%(" in smiles


def test_molecule_with_over_99_open_rings_round_trips():
    smiles = Chem.MolToSmiles(_many_open_rings_mol())
    tokens = tokenize_smiles(smiles)
    assert "".join(tokens) == smiles
    assert any(token.startswith("%(") for token in tokens)
    assert any(token.startswith("%") and not token.startswith("%(") for token in tokens)


# ---------------------------------------------------------------------------
# Empty input vs. failure
# ---------------------------------------------------------------------------


def test_empty_string_tokenizes_to_the_empty_tuple():
    """A successful empty result -- explicitly not an error, and explicitly
    distinguishable from one."""
    assert tokenize_smiles("") == ()


@pytest.mark.parametrize(
    "malformed",
    [
        "CC?CC",          # unknown character
        "C&C",            # unknown character
        "C%1C",           # "%" needs exactly two digits (RDKit rejects "%1")
        "C[CH3",          # unterminated bracket atom
        "CC]",            # stray closing bracket
        "C[]C",           # empty bracket atom
        "C C",            # whitespace is not a SMILES token
        "CC\n",           # trailing newline is not a SMILES token
        "C<C",            # "<" only ever appears as part of "<-"
        "C>C",            # ">" only ever appears as part of "->"
        "C+C",            # charges are only legal inside brackets
    ],
)
def test_unrecognized_input_raises_value_error(malformed):
    with pytest.raises(ValueError):
        tokenize_smiles(malformed)


def test_error_message_names_the_offending_position_and_character():
    with pytest.raises(ValueError) as exc_info:
        tokenize_smiles("CC?CC")
    message = str(exc_info.value)
    assert "position 2" in message
    assert "'?'" in message


def test_unrecognized_input_never_returns_a_lossy_tokenization():
    """The failure mode must be an exception, never a silently shortened
    token sequence -- otherwise a corrupted corpus entry would look like a
    valid one."""
    for malformed in ("CC?CC", "C&C", "C[CH3"):
        with pytest.raises(ValueError):
            tokenize_smiles(malformed)


def test_non_string_input_raises_value_error():
    with pytest.raises(ValueError):
        tokenize_smiles(None)
    with pytest.raises(ValueError):
        tokenize_smiles(42)


# ---------------------------------------------------------------------------
# Normalization + tokenization together
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("CCO", "OCC"),
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("CC(=O)Oc1ccccc1C(=O)O", "O=C(O)c1ccccc1OC(C)=O"),
        ("[Na+].[Cl-]", "[Cl-].[Na+]"),
    ],
)
def test_equivalent_smiles_yield_the_same_canonical_string_and_tokens(left, right):
    canonical_left = canonicalize_smiles(left)
    canonical_right = canonicalize_smiles(right)

    assert canonical_left == canonical_right
    assert tokenize_smiles(canonical_left) == tokenize_smiles(canonical_right)


def test_tokenizer_does_not_canonicalize_its_input():
    """Normalization and tokenization are separately versioned steps, so the
    lexer must report exactly what it was given -- a non-canonical spelling
    tokenizes to that spelling's tokens, not the canonical form's."""
    non_canonical = "OCC"
    assert canonicalize_smiles(non_canonical) == "CCO"
    assert tokenize_smiles(non_canonical) == ("O", "C", "C")


def test_stereochemistry_survives_into_the_token_sequence():
    tokens = tokenize_smiles(canonicalize_smiles("C[C@H](O)c1ccccc1"))
    assert any("@" in token for token in tokens)


def test_disconnected_components_survive_into_the_token_sequence():
    tokens = tokenize_smiles(canonicalize_smiles("CC(=O)[O-].[Na+]"))
    assert tokens.count(".") == 1


# ---------------------------------------------------------------------------
# Contract identifier
# ---------------------------------------------------------------------------


def test_tokenizer_contract_id_is_pinned():
    assert SMILES_TOKENIZER_ID == "rdkit_smiles_lexer_v1"
