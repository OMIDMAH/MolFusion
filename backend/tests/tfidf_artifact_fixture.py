"""A small, production-shaped TF-IDF artifact for tests.

Built by the real Phase 5F-D builder from a tiny synthetic corpus, so the
fixture exercises the same selection, serialization, checksum and metadata
code paths the production artifact went through -- it is a smaller
artifact, not a different kind of artifact.

Deliberately not the real production artifact: unit tests must not depend
on a 30-minute ChEMBL build being present, and must be able to corrupt
payloads freely.
"""

from pathlib import Path

from molfusion_backend.corpus.serialization import write_corpus
from molfusion_backend.tfidf import contract
from molfusion_backend.tfidf.builder import build_artifact

# Wide enough that the vocabulary contains all three n-gram orders and that
# ordinary organic molecules retain terms, small enough to build instantly.
FIXTURE_SMILES = sorted(
    {
        "CCO", "CCN", "CCC", "CCCC", "CCCCC", "OCCO", "NCCN", "CCOCC",
        "CC(=O)O", "CC(=O)N", "CC(C)C", "CC(C)O", "CCCl", "CCBr", "CCS",
        "c1ccccc1", "c1ccccc1C", "c1ccccc1O", "c1ccccc1N", "c1ccncc1",
        "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CC(=O)Oc1ccccc1C(=O)O",
        "CC(=O)[O-].[Na+]", "C[N+](C)(C)C.[Cl-]",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    }
    | {f"C{'C' * index}O" for index in range(1, 60)}
    | {f"c1ccccc1{'C' * index}" for index in range(1, 40)}
    | {f"CC(=O)N{'C' * index}" for index in range(1, 30)}
)

# The frozen contract values. The fixture must satisfy the real contract
# so the agent's production-strength validation is genuinely exercised;
# only the corpus is smaller. `max_features` simply never binds here.
FIXTURE_MIN_DF = 5
FIXTURE_MAX_FEATURES = 4096


def build_fixture_artifact(root: Path, *, corpus_dir: Path | None = None) -> Path:
    """Build a fixture artifact under `root` and return its directory.

    Both `min_df` and `max_features` are left at the frozen production
    values, so the artifact passes the same contract validation the real one
    does. The corpus is far too small to reach the cap, so it does not bind.
    """
    corpus_dir = Path(corpus_dir if corpus_dir is not None else root)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / "canonical_smiles.smi"
    sha256, _ = write_corpus(corpus_path, FIXTURE_SMILES)

    build_artifact(
        corpus_path,
        root=root,
        expected_sha256=sha256,
        expected_document_count=len(FIXTURE_SMILES),
        min_df=FIXTURE_MIN_DF,
        max_features=FIXTURE_MAX_FEATURES,
        progress_every=0,
    )
    return root / contract.ARTIFACT_TYPE / contract.ARTIFACT_ID / contract.ARTIFACT_VERSION


def fixture_dimension(root: Path) -> int:
    """The dimension the fixture artifact actually produced."""
    import json

    directory = root / contract.ARTIFACT_TYPE / contract.ARTIFACT_ID / contract.ARTIFACT_VERSION
    payload = json.loads((directory / contract.CONFIG_FILENAME).read_text(encoding="utf-8"))
    return payload["dimension"]


__all__ = [
    "FIXTURE_MAX_FEATURES",
    "FIXTURE_MIN_DF",
    "FIXTURE_SMILES",
    "build_fixture_artifact",
    "fixture_dimension",
]
