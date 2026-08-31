"""Production SMILES token n-gram TF-IDF feature agent.

Consumes the frozen artifact built in Phase 5F-D. It fits nothing, chooses
nothing, and reinterprets nothing: the vocabulary, the IDF vector and every
numerical rule arrive from the artifact and from
`molfusion_backend.tfidf`, and this module only wires them to the
FeatureAgent interface.

    Mol -> canonical_smiles_from_mol -> tokenize_smiles
        -> n-grams (orders 1-3) -> counts over the frozen vocabulary
        -> sublinear TF -> x IDF -> L2 -> float32 (4096,)

Three failure modes are deliberately distinct, because collapsing any two
of them would hide a real problem behind a plausible-looking vector:

    invalid molecule       never reaches this agent; the API rejects it
    artifact unusable      ValueError -- the representation cannot be
                           computed at all, and must not be faked
    no retained n-gram     a valid all-zero vector, not an error

The third is a legitimate scientific result for a molecule built entirely
from motifs the ChEMBL vocabulary never saw. The second is a deployment
fault. They must never look alike to a caller.
"""

import threading
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from rdkit import Chem

from molfusion_backend.agents.base import FeatureAgent
from molfusion_backend.artifacts.errors import ArtifactError
from molfusion_backend.chemistry import canonical_smiles_from_mol
from molfusion_backend.smiles_tokenizer import tokenize_smiles
from molfusion_backend.tfidf import contract
from molfusion_backend.tfidf.loader import load_tfidf_artifact
from molfusion_backend.tfidf.transform import TfidfTransformer

AGENT_ID = "smiles_tfidf_4096"

class _RuntimeState:
    """Immutable, validated artifact state shared across compute() calls.

    Plain object rather than a dataclass so the read-only guarantees can be
    established in one place: the IDF buffer is marked non-writeable and
    the vocabulary lookup is wrapped in a read-only mapping, so a
    concurrent caller cannot mutate shared state through a returned
    reference even by accident.
    """

    __slots__ = ("transformer", "feature_names", "dimension", "artifact_directory")

    def __init__(self, artifact: Any) -> None:
        idf = np.array(artifact.idf, dtype=np.float64, copy=True)
        idf.flags.writeable = False

        self.dimension = artifact.dimension
        self.artifact_directory = artifact.descriptor.directory
        # Derived by the vocabulary itself, so the name format has one
        # definition rather than one here and another beside the tokens.
        self.feature_names = tuple(artifact.feature_names())
        self.transformer = TfidfTransformer(
            index_map=MappingProxyType(dict(artifact.vocabulary.index_map())),
            idf=idf,
            dimension=artifact.dimension,
            orders=tuple(
                range(artifact.config.ngram_min, artifact.config.ngram_max + 1)
            ),
        )


class SmilesTfidfAgent(FeatureAgent):
    """Frozen ChEMBL 37 SMILES token n-gram TF-IDF, 4096 dimensions.

    Continuous, not binary or count: values are L2-normalized sublinear-TF
    times smoothed IDF, so a nonzero vector has unit length and its entries
    are small positive reals.

    The artifact is loaded lazily and once. Eager loading in `__init__`
    would tie process start-up -- and every test that registers the builtin
    agents -- to the artifact being present on disk, which is a heavier
    coupling than a representation deserves. Loading is guarded by a lock
    and the result is cached, so concurrent first calls load exactly once
    and every later call reuses read-only state.

    A load *failure* is deliberately not cached: an artifact restored after
    a bad deployment starts working without a process restart. The retry
    cost is a stat call in the common missing-artifact case.
    """

    id = AGENT_ID
    name = "SMILES Token n-gram TF-IDF (ChEMBL 37)"
    version = "1.0.0"
    # From the frozen contract, not from the artifact: declaring the
    # dimension must not require reading the artifact, and the loaded
    # artifact is checked against this value rather than defining it.
    # Overridable per instance only so tests can bind a smaller fixture
    # artifact; production always uses the frozen 4096.
    output_dim = contract.DIMENSION
    requires_3d = False
    value_type = "continuous"
    output_structure = "vector"

    def __init__(
        self,
        *,
        artifact_root: Path | None = None,
        artifact_id: str = contract.ARTIFACT_ID,
        artifact_version: str = contract.ARTIFACT_VERSION,
        artifact_type: str = contract.ARTIFACT_TYPE,
        output_dim: int = contract.DIMENSION,
    ) -> None:
        # `artifact_root=None` means "resolve at load time", which is what
        # keeps MOLFUSION_ARTIFACT_ROOT effective: the generic
        # infrastructure reads the environment when the artifact is
        # actually loaded, so the override works without this agent knowing
        # the variable exists.
        self._artifact_root = artifact_root
        self._artifact_type = artifact_type
        # Every "which artifact am I bound to" parameter travels together,
        # dimension included: a different artifact version could legitimately
        # have a different one, and the agent must declare the dimension it
        # will actually emit rather than discovering it at first compute.
        self.output_dim = output_dim
        self._artifact_id = artifact_id
        self._artifact_version = artifact_version
        self._lock = threading.Lock()
        self._state: _RuntimeState | None = None

    # -- artifact lifecycle -------------------------------------------------

    def load(self) -> _RuntimeState:
        """Load, validate and cache the artifact. Raises on any problem.

        Double-checked locking: the fast path is a plain attribute read
        after the first successful load, and the slow path holds the lock
        so two threads racing on a cold agent cannot both parse the
        vocabulary.
        """
        state = self._state
        if state is not None:
            return state

        with self._lock:
            if self._state is not None:
                return self._state

            artifact = load_tfidf_artifact(
                self._artifact_id,
                self._artifact_version,
                artifact_type=self._artifact_type,
                root=self._artifact_root,
            )
            if artifact.dimension != self.output_dim:
                raise ValueError(
                    f"{self.id}: artifact {self._artifact_id}/{self._artifact_version} "
                    f"has dimension {artifact.dimension}, but this agent declares "
                    f"output_dim={self.output_dim}."
                )
            self._state = _RuntimeState(artifact)
            return self._state

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        """Per-index feature names, or None if the artifact is unavailable.

        `GET /agents` reads this for every registered agent, so raising
        here would take the whole registry listing down with it whenever
        the TF-IDF artifact is missing. `None` is already the value that
        endpoint carries for agents with no per-feature names, so it
        degrades into an existing, understood shape rather than an error.

        This is not a silent fallback for the representation itself:
        `compute()` raises loudly for exactly the same condition, so a
        missing artifact can never produce a vector.
        """
        try:
            return self.load().feature_names
        except (ArtifactError, ValueError, OSError):
            return None

    # -- computation --------------------------------------------------------

    def compute(self, mol: Chem.Mol) -> np.ndarray:
        if mol is None:
            raise ValueError(
                f"{self.id}: compute() received mol=None; a valid RDKit Mol is required."
            )

        try:
            state = self.load()
        except ArtifactError as exc:
            # Surfaced as ValueError so the API reports it the way every
            # other representation failure is reported: the molecule stays
            # valid=True and the error is named. Never a zero vector --
            # that shape is reserved for a molecule with no retained
            # vocabulary term, which is a result, not a fault.
            raise ValueError(
                f"{self.id}: the frozen TF-IDF artifact could not be loaded "
                f"({self._artifact_type}/{self._artifact_id}/{self._artifact_version}): {exc}"
            ) from exc

        # The shared canonicalization contract, not a local MolToSmiles
        # call: the artifact's vocabulary was fitted on strings produced by
        # exactly this function, so any divergence here would silently
        # shift every molecule's tokens away from the fitted vocabulary.
        canonical = canonical_smiles_from_mol(mol)
        tokens = tokenize_smiles(canonical)
        return state.transformer.transform(tokens)


__all__ = ["AGENT_ID", "SmilesTfidfAgent"]
