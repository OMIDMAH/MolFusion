"""Lossless lexer for RDKit canonical SMILES.

Purely lexical: it splits a SMILES *string* into its atomic textual
constructs and knows nothing about valence, aromaticity perception, or
molecular identity. It never parses with RDKit, never canonicalizes its
input, and never builds a vocabulary, integer IDs, padding, or embeddings
-- those belong to whatever consumes the tokens.

The intended pairing (see molfusion_backend.chemistry) keeps normalization
and tokenization as two separate, separately-versioned steps:

    canonical = canonicalize_smiles(raw_smiles)
    tokens = tokenize_smiles(canonical)
"""

import re

# Identifier for the tokenization contract implemented by
# tokenize_smiles(). Recorded alongside
# chemistry.CANONICAL_SMILES_NORMALIZATION_ID by any future artifact whose
# vocabulary/document frequencies depend on this exact tokenization. Bump
# the suffix only if the token sequence for a fixed input string would
# change.
SMILES_TOKENIZER_ID = "rdkit_smiles_lexer_v1"

# Ordered alternation; `re` takes the first alternative that matches at the
# current position, so every prefix-ambiguous pair is ordered longest-first
# ("Cl" before "C", "->" before "-", "%(12)" and "%12" before the bare
# digit rules). Verified against the SMILES actually emitted by the RDKit
# version this repo pins -- in particular RDKit writes dative bonds
# ("[NH3]->[Cu]"), quadruple bonds ("[Re]$[Re]"), unspecified bonds
# ("C~C"), and, past 99 simultaneously-open rings, the extended ring
# closure form "%(100)" rather than only the classic "%nn".
_TOKEN_PATTERN = re.compile(
    "|".join(
        (
            # Bracket atom: kept whole, contents never split. SMILES
            # brackets do not nest, so "everything up to the first ']'" is
            # exact -- this covers isotope, element/aromatic symbol,
            # chirality (@, @@, @TH1, @SP1, @TB8, @OH24, ...), hydrogen
            # count, formal charge and atom map class in one token.
            r"\[[^\[\]]+\]",
            # Ring closures. RDKit emits "%(nnn)" only once ring numbers
            # exceed 99; "%nn" is exactly two digits (RDKit rejects "%1"
            # and "%05"). The bare single digit is matched last, below.
            r"%\(\d+\)",
            r"%\d\d",
            # Dative bonds, before the plain "-" single bond.
            r"->",
            r"<-",
            # Two-character organic-subset elements, before their
            # one-character prefixes "C" and "B".
            r"Cl",
            r"Br",
            # Remaining organic-subset atoms written without brackets,
            # plus the "*" wildcard. Case is chemically meaningful:
            # uppercase is aliphatic, lowercase aromatic -- never folded.
            r"[BCNOPSFI]",
            r"[bcnops]",
            r"\*",
            # Bond symbols, branch delimiters, component separator.
            r"[=#$:/\\~.\-()]",
            # Single-digit ring closure.
            r"\d",
        )
    )
)


def tokenize_smiles(smiles: str) -> tuple[str, ...]:
    """Split a SMILES string into its lexical tokens, losslessly.

    Guarantees ``"".join(tokenize_smiles(s)) == s`` for every input that
    tokenizes successfully: tokens are consecutive, non-overlapping matches
    that together cover the whole string, so no character can be dropped,
    reordered, or rewritten.

    Bracket atoms stay atomic ("[C@@H]", "[13CH3]", "[nH]", "[O-]"), and
    "Cl"/"Br" are single tokens rather than "C"+"l" / "B"+"r".

    "" tokenizes to the empty tuple -- a successful empty result, not a
    failure. Anything the lexer cannot recognize raises ValueError naming
    the offending offset, rather than skipping it and returning a lossy
    token sequence.
    """
    if not isinstance(smiles, str):
        raise ValueError(
            f"tokenize_smiles() expects a str, got {type(smiles).__name__}."
        )

    tokens: list[str] = []
    position = 0
    length = len(smiles)

    while position < length:
        match = _TOKEN_PATTERN.match(smiles, position)
        if match is None:
            raise ValueError(
                f"Unrecognized SMILES syntax at position {position} "
                f"({smiles[position]!r}) in {smiles!r}."
            )
        tokens.append(match.group())
        position = match.end()

    return tuple(tokens)
