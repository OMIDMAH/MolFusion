from rdkit import Chem


def parse_smiles(smiles: str) -> tuple[Chem.Mol | None, str | None]:
    """Parse a SMILES string into an RDKit Mol using Chem.MolFromSmiles.

    This intentionally does NOT use the legacy v1 regex-based validator.

    Returns
    -------
    (mol, None) on success.
    (None, error_message) on failure — either because RDKit could not
    parse the string, or because RDKit raised while parsing (e.g. for a
    non-string input that slipped through).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        return None, f"RDKit failed to parse SMILES '{smiles}': {exc}"

    if mol is None:
        return None, f"Invalid SMILES: '{smiles}' could not be parsed by RDKit."

    return mol, None


# Identifier for the normalization contract implemented by
# canonical_smiles_from_mol() / canonicalize_smiles(). It exists so that a
# future artifact's metadata.json can record *which* normalization produced
# the corpus it was fitted on, without that metadata having to restate the
# RDKit call. Bump the suffix only if the emitted string for a fixed
# molecule and RDKit version would change.
CANONICAL_SMILES_NORMALIZATION_ID = "rdkit_canonical_isomeric_smiles_v1"


def canonical_smiles_from_mol(mol: Chem.Mol) -> str:
    """Serialize an already-parsed RDKit Mol to canonical isomeric SMILES.

    This is *serialization*, not molecular standardization. It deliberately
    performs no salt stripping, largest-fragment selection, neutralization,
    charge normalization, tautomer canonicalization, or stereochemistry
    removal: stereo descriptors, isotopes, formal charges and disconnected
    components all survive verbatim. Aromaticity is written in RDKit's
    aromatic (lowercase) form, not Kekule form, so a molecule has exactly
    one canonical serialization here.

    Raises ValueError if mol is None, mirroring the agents' convention that
    a missing molecule is a caller error, not a recoverable empty result.
    """
    if mol is None:
        raise ValueError(
            "canonical_smiles_from_mol() received mol=None; a valid RDKit Mol is required."
        )

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
        kekuleSmiles=False,
    )


def canonicalize_smiles(smiles: str) -> str:
    """Parse a SMILES string and return its canonical isomeric form.

    Equivalent to parse_smiles() followed by canonical_smiles_from_mol().
    Two SMILES that describe the same molecule (e.g. "CCO" and "OCC")
    always produce the same string; repeated calls are deterministic.

    Note that "" is valid input: RDKit parses it as the empty molecule and
    it canonicalizes to "". That is a successful empty result, distinct
    from the ValueError raised for input RDKit cannot parse.

    Raises ValueError if RDKit cannot parse the input.
    """
    mol, error = parse_smiles(smiles)
    if mol is None:
        raise ValueError(error)

    return canonical_smiles_from_mol(mol)
