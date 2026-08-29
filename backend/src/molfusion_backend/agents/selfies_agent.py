import threading
from contextlib import contextmanager
from typing import Iterator

import selfies as sf
from rdkit import Chem

from molfusion_backend.agents.base import FeatureAgent
from molfusion_backend.chemistry import canonical_smiles_from_mol

# SELFIES exposes semantic constraints as *mutable global module state*
# (selfies.set_semantic_constraints() / selfies.get_semantic_constraints()),
# not as a per-call parameter to encoder()/decoder(). Per selfies.encoder()'s
# own docstring: "This translation is deterministic and does not depend on
# the current semantic constraints" -- the *token output* for a molecule
# that encodes successfully is constraint-independent. However, whether
# encoding *succeeds at all* does depend on the active constraints: with the
# (default) strict=True, encoder() raises EncoderError for molecules that
# violate the currently active constraints. So constraints cannot silently
# change this agent's *output*, but they could silently change whether a
# given molecule *errors*, if some other code in the process later called
# set_semantic_constraints() with different bonding capacities (e.g. the
# "octet_rule" or "hypervalent" presets).
#
# "default" is selfies' own baseline preset. Pinning it changes nothing
# about selfies' out-of-the-box behavior -- it only makes the policy
# explicit and provably immune to later, unrelated global mutation.
SELFIES_CONSTRAINT_PRESET = "default"

# MolFusion is served through FastAPI, whose synchronous `def` route
# handlers run in a threadpool -- so multiple SelfiesSequenceAgent.compute()
# calls really can execute concurrently on different threads. Because
# selfies' semantic constraints are process-global, two concurrent calls'
# save -> set-default -> encode -> restore sequences could otherwise
# interleave (e.g. call B's "set default" landing between call A's "save"
# and call A's "restore", corrupting what A restores). This lock serializes
# that whole critical section so calls never interleave their mutation of
# the shared global state, while still leaving each compute() call with no
# *persistent* side effect once it returns.
_CONSTRAINT_LOCK = threading.RLock()


@contextmanager
def _pinned_selfies_constraints() -> Iterator[None]:
    """Temporarily activate MolFusion's pinned constraint policy for the
    duration of the `with` block, then restore whatever constraints were
    active before -- even if the block raises. The whole save/set/restore
    sequence is serialized by _CONSTRAINT_LOCK (see above), so this agent
    has no persistent process-global side effect regardless of how many
    compute() calls run concurrently.
    """
    with _CONSTRAINT_LOCK:
        previous_constraints = sf.get_semantic_constraints()
        sf.set_semantic_constraints(sf.get_preset_constraints(SELFIES_CONSTRAINT_PRESET))
        try:
            yield
        finally:
            sf.set_semantic_constraints(previous_constraints)


class SelfiesSequenceAgent(FeatureAgent):
    """Native SELFIES token-sequence representation.

    Molecule-centric, not text-centric: this agent encodes the RDKit
    *canonical isomeric SMILES* of the input molecule (via
    Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)), never the
    user's original SMILES text. Equivalent SMILES written in a different
    atom order (e.g. "CCO" and "OCC") therefore always produce identical
    SELFIES token sequences, because they are canonicalized to the same
    RDKit SMILES before encoding.

    Output is the *native*, variable-length SELFIES token sequence, as
    returned by selfies.split_selfies() on the encoder's output. This agent
    never pads, truncates, one-hot encodes, converts tokens to integer IDs,
    or embeds -- the sequence length is a property of each molecule's
    result (see SequenceFeatureOutput.length in the API layer), not a fixed
    agent-level output_dim.
    """

    id = "selfies_sequence"
    name = "SELFIES Sequence"
    version = "1.0.0"
    output_dim = None
    requires_3d = False
    value_type = "categorical"
    output_structure = "sequence"
    feature_names = None

    def compute(self, mol: Chem.Mol) -> tuple[str, ...]:
        if mol is None:
            raise ValueError(
                f"{self.id}: compute() received mol=None; a valid RDKit Mol is required."
            )

        # Shared with every other consumer of canonical isomeric SMILES via
        # molfusion_backend.chemistry, so a single normalization contract
        # governs them all -- identical output to the previous inline
        # Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) call.
        canonical_smiles = canonical_smiles_from_mol(mol)

        with _pinned_selfies_constraints():
            try:
                selfies_string = sf.encoder(canonical_smiles)
            except sf.EncoderError as exc:
                raise ValueError(
                    f"{self.id}: failed to encode molecule as SELFIES: {canonical_smiles!r}"
                ) from exc

        return tuple(sf.split_selfies(selfies_string))
