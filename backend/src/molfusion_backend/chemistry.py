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
