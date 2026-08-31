import json
import threading

import numpy as np
import pytest
from rdkit import Chem

from molfusion_backend.agents.smiles_tfidf import AGENT_ID, SmilesTfidfAgent
from molfusion_backend.artifacts import sha256_file
from molfusion_backend.chemistry import canonical_smiles_from_mol
from molfusion_backend.tfidf import contract
from molfusion_backend.tfidf.idf import idf_bytes, load_idf
from tests.tfidf_artifact_fixture import build_fixture_artifact, fixture_dimension


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    """One fixture artifact for the whole module -- building it is the slow
    part and the artifact is immutable, so sharing it is safe."""
    root = tmp_path_factory.mktemp("artifacts")
    build_fixture_artifact(root, corpus_dir=tmp_path_factory.mktemp("corpus"))
    return root


@pytest.fixture()
def agent(fixture_root):
    return SmilesTfidfAgent(
        artifact_root=fixture_root, output_dim=fixture_dimension(fixture_root)
    )


def mol(smiles):
    parsed = Chem.MolFromSmiles(smiles)
    assert parsed is not None, smiles
    return parsed


# ---------------------------------------------------------------------------
# agent metadata
# ---------------------------------------------------------------------------


def test_agent_metadata_matches_the_frozen_representation():
    fresh = SmilesTfidfAgent()
    assert fresh.id == AGENT_ID == "smiles_tfidf_4096"
    assert fresh.value_type == "continuous"
    assert fresh.output_structure == "vector"
    assert fresh.output_dim == 4096 == contract.DIMENSION
    assert fresh.requires_3d is False
    assert fresh.version


def test_the_representation_is_continuous_not_binary_or_count():
    """L2-normalized real values, so neither a bit vector nor integer counts."""
    assert SmilesTfidfAgent.value_type == "continuous"
    assert SmilesTfidfAgent.value_type not in ("binary", "count", "categorical")


def test_the_class_level_dimension_stays_the_frozen_production_value():
    """Instances may bind a smaller fixture artifact; the class must not."""
    assert SmilesTfidfAgent.output_dim == 4096
    assert SmilesTfidfAgent(output_dim=7).output_dim == 7
    assert SmilesTfidfAgent.output_dim == 4096


# ---------------------------------------------------------------------------
# artifact loading
# ---------------------------------------------------------------------------


def test_loads_a_production_shaped_fixture_artifact(agent, fixture_root):
    state = agent.load()
    assert state.dimension == fixture_dimension(fixture_root)
    assert state.transformer.dimension == state.dimension
    assert len(state.feature_names) == state.dimension


def test_the_artifact_is_loaded_once_and_cached(agent):
    assert agent.load() is agent.load()


def test_the_artifact_is_not_loaded_during_construction(fixture_root):
    """Registration must not depend on the artifact being present."""
    fresh = SmilesTfidfAgent(artifact_root=fixture_root / "absent")
    assert fresh._state is None  # not loaded merely by constructing


def test_respects_the_artifact_root_environment_override(fixture_root, monkeypatch):
    """The agent never names the environment variable; the generic
    infrastructure resolves it at load time."""
    monkeypatch.setenv("MOLFUSION_ARTIFACT_ROOT", str(fixture_root))
    unrooted = SmilesTfidfAgent(output_dim=fixture_dimension(fixture_root))
    vector = unrooted.compute(mol("CCO"))
    assert vector.shape == (fixture_dimension(fixture_root),)


def test_a_missing_artifact_fails_clearly(tmp_path):
    lost = SmilesTfidfAgent(artifact_root=tmp_path / "nothing-here")
    with pytest.raises(ValueError, match="artifact could not be loaded"):
        lost.compute(mol("CCO"))


def test_a_missing_artifact_never_yields_a_zero_vector(tmp_path):
    """The failure must not be dressed up as a computable representation."""
    lost = SmilesTfidfAgent(artifact_root=tmp_path / "nothing-here")
    with pytest.raises(ValueError):
        lost.compute(mol("CC(=O)Oc1ccccc1C(=O)O"))


def test_checksum_corruption_fails(tmp_path, fixture_root):
    import shutil

    root = tmp_path / "corrupt-checksum"
    shutil.copytree(fixture_root, root)
    directory = root / contract.ARTIFACT_TYPE / contract.ARTIFACT_ID / contract.ARTIFACT_VERSION
    path = directory / contract.VOCABULARY_FILENAME
    path.write_bytes(path.read_bytes().replace(b'"index": 0', b'"index": 1', 1))

    broken = SmilesTfidfAgent(artifact_root=root, output_dim=fixture_dimension(root))
    with pytest.raises(ValueError, match="artifact could not be loaded"):
        broken.compute(mol("CCO"))


def test_semantic_corruption_fails(tmp_path, fixture_root):
    """Checksum-valid but semantically wrong: only the TF-IDF loader can
    catch this, and the agent must surface it rather than compute."""
    import shutil

    root = tmp_path / "corrupt-semantics"
    shutil.copytree(fixture_root, root)
    directory = root / contract.ARTIFACT_TYPE / contract.ARTIFACT_ID / contract.ARTIFACT_VERSION

    values = load_idf(directory / contract.IDF_FILENAME)
    values[0] += 0.5
    (directory / contract.IDF_FILENAME).write_bytes(idf_bytes(values))
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for record in metadata["payload_files"]:
        if record["filename"] == contract.IDF_FILENAME:
            record["sha256"] = sha256_file(directory / contract.IDF_FILENAME)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    broken = SmilesTfidfAgent(artifact_root=root, output_dim=fixture_dimension(root))
    with pytest.raises(ValueError, match="artifact could not be loaded"):
        broken.compute(mol("CCO"))


def test_a_dimension_mismatch_is_refused(fixture_root):
    wrong = SmilesTfidfAgent(artifact_root=fixture_root, output_dim=99999)
    with pytest.raises(ValueError, match="output_dim"):
        wrong.load()


def test_feature_names_degrade_to_none_when_the_artifact_is_unavailable(tmp_path):
    """GET /agents reads this for every agent, so it must not take the whole
    registry listing down. compute() still raises for the same condition."""
    lost = SmilesTfidfAgent(artifact_root=tmp_path / "nothing-here")
    assert lost.feature_names is None
    with pytest.raises(ValueError):
        lost.compute(mol("CCO"))


# ---------------------------------------------------------------------------
# canonicalization
# ---------------------------------------------------------------------------


def test_uses_the_shared_canonicalization_contract(agent):
    """Two spellings of the same molecule must give the same vector, because
    the agent canonicalizes before tokenizing."""
    assert np.array_equal(agent.compute(mol("CCO")), agent.compute(mol("OCC")))
    assert np.array_equal(agent.compute(mol("c1ccccc1")), agent.compute(mol("C1=CC=CC=C1")))


def test_the_canonical_form_is_what_gets_tokenized(agent):
    """Pins the pipeline: the vector equals what the frozen transformer
    produces from canonical_smiles_from_mol()'s output, not from the input
    string."""
    from molfusion_backend.smiles_tokenizer import tokenize_smiles

    molecule = mol("OCC")
    canonical = canonical_smiles_from_mol(molecule)
    assert canonical == "CCO"
    expected = agent.load().transformer.transform(tokenize_smiles(canonical))
    assert np.array_equal(agent.compute(molecule), expected)


def test_a_none_molecule_is_a_caller_error(agent):
    with pytest.raises(ValueError, match="mol=None"):
        agent.compute(None)


# ---------------------------------------------------------------------------
# transformation
# ---------------------------------------------------------------------------


TRANSFORM_CASES = [
    ("ordinary", "CC(=O)Oc1ccccc1C(=O)O"),
    ("aromatic", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
    ("stereochemical", "C[C@H](N)C(=O)O"),
    ("disconnected", "CC(=O)[O-].[Na+]"),
    ("repeated motif", "CCCCCCCCCCCCCCCCCCCC"),
    ("contains OOV n-grams", "CCO"),
]


@pytest.mark.parametrize(("label", "smiles"), TRANSFORM_CASES)
def test_transformation_shape_dtype_and_finiteness(agent, label, smiles, fixture_root):
    vector = agent.compute(mol(smiles))
    assert vector.shape == (fixture_dimension(fixture_root),)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))
    assert not np.any(np.isnan(vector))


@pytest.mark.parametrize(("label", "smiles"), TRANSFORM_CASES)
def test_nonzero_vectors_are_l2_normalized(agent, label, smiles):
    vector = agent.compute(mol(smiles))
    if vector.any():
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, rel=1e-6)


def test_an_all_oov_molecule_gives_an_exact_zero_vector(agent, fixture_root):
    """A valid molecule made only of motifs the vocabulary never saw. This
    is a result, not a failure: no raise, no None, no error."""
    vector = agent.compute(mol("[Xe]"))
    assert vector.shape == (fixture_dimension(fixture_root),)
    assert vector.dtype == np.float32
    assert not vector.any()
    assert np.all(vector == 0.0)
    assert np.all(np.isfinite(vector))


def test_out_of_vocabulary_ngrams_do_not_change_the_dimension(agent, fixture_root):
    before = agent.load().dimension
    agent.compute(mol("[Xe]"))
    agent.compute(mol("CC(=O)Oc1ccccc1C(=O)O"))
    assert agent.load().dimension == before
    assert len(agent.load().feature_names) == fixture_dimension(fixture_root)


def test_a_repeated_motif_is_damped_not_counted(agent):
    """Sublinear TF: twenty carbons must not weigh twenty times one."""
    short = agent.compute(mol("CCCC"))
    long_chain = agent.compute(mol("CCCCCCCCCCCCCCCCCCCC"))
    assert float(np.linalg.norm(short)) == pytest.approx(1.0, rel=1e-6)
    assert float(np.linalg.norm(long_chain)) == pytest.approx(1.0, rel=1e-6)
    assert not np.array_equal(short, long_chain)


# ---------------------------------------------------------------------------
# determinism and thread safety
# ---------------------------------------------------------------------------


def test_repeated_calls_are_identical(agent):
    molecule = mol("CC(=O)Oc1ccccc1C(=O)O")
    first = agent.compute(molecule)
    for _ in range(5):
        assert np.array_equal(agent.compute(molecule), first)


def test_separate_agent_instances_agree(fixture_root):
    dimension = fixture_dimension(fixture_root)
    left = SmilesTfidfAgent(artifact_root=fixture_root, output_dim=dimension)
    right = SmilesTfidfAgent(artifact_root=fixture_root, output_dim=dimension)
    for smiles in ("CCO", "c1ccccc1", "CC(=O)[O-].[Na+]", "[Xe]"):
        assert np.array_equal(left.compute(mol(smiles)), right.compute(mol(smiles)))


def test_concurrent_first_use_loads_once_and_agrees(fixture_root):
    """Two threads racing on a cold agent must not both build state, and
    must not produce different vectors."""
    agent = SmilesTfidfAgent(
        artifact_root=fixture_root, output_dim=fixture_dimension(fixture_root)
    )
    results = {}
    barrier = threading.Barrier(4)

    def work(index):
        barrier.wait()
        results[index] = agent.compute(mol("CC(=O)Oc1ccccc1C(=O)O"))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 4
    first = results[0]
    assert all(np.array_equal(vector, first) for vector in results.values())


def test_runtime_state_is_read_only(agent):
    state = agent.load()
    assert state.transformer.idf.flags.writeable is False
    with pytest.raises(ValueError):
        state.transformer.idf[0] = 1.0
    with pytest.raises(TypeError):
        state.transformer.index_map[("C",)] = 0


# ---------------------------------------------------------------------------
# feature names
# ---------------------------------------------------------------------------


def test_feature_names_count_and_uniqueness(agent, fixture_root):
    names = agent.feature_names
    assert len(names) == fixture_dimension(fixture_root)
    assert len(set(names)) == len(names)


def test_feature_names_align_with_vector_indices(agent):
    """Name i must describe column i -- checked against the vocabulary the
    artifact actually stores, not against a re-derivation."""
    state = agent.load()
    for entry in state.transformer.index_map.items():
        tokens, index = entry
        name = state.feature_names[index]
        prefix, payload = name.split(":", 1)
        assert prefix == f"ngram{len(tokens)}"
        assert tuple(json.loads(payload)) == tokens


def test_feature_names_are_unambiguous_and_round_trip(agent):
    for name in agent.feature_names:
        prefix, payload = name.split(":", 1)
        tokens = json.loads(payload)
        assert isinstance(tokens, list)
        assert all(isinstance(token, str) for token in tokens)
        assert prefix == f"ngram{len(tokens)}"


def test_feature_names_avoid_the_csv_name_separator(agent):
    """The frontend CSV export joins names with ';'."""
    assert all(";" not in name for name in agent.feature_names)


def test_feature_names_are_stable_across_instances_and_reloads(fixture_root):
    dimension = fixture_dimension(fixture_root)
    left = SmilesTfidfAgent(artifact_root=fixture_root, output_dim=dimension)
    right = SmilesTfidfAgent(artifact_root=fixture_root, output_dim=dimension)
    assert left.feature_names == right.feature_names
    assert left.feature_names == left.feature_names


def test_feature_names_do_not_leak_df_or_idf(agent):
    names = agent.feature_names
    state = agent.load()
    for index in (0, len(names) // 2, len(names) - 1):
        assert str(float(state.transformer.idf[index])) not in names[index]
