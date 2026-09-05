"""sklearn as an independent reference implementation, never as a dependency.

Phase 5F-C.1 chose a MolFusion-owned NumPy transform over sklearn, and the
risk that choice carries is reimplementing standard mathematics wrongly.
These tests close that gap by checking the frozen arithmetic against a
mature third-party implementation of the same equations.

What sklearn is *not* allowed to do here, because doing any of it would
make the comparison circular or hand it a decision that is MolFusion's:

  * select the vocabulary  -- the frozen vocabulary is injected
  * tokenize SMILES        -- MolFusion's frozen tokenizer produces the
                              n-grams, and counts are handed over already
                              aligned to MolFusion's column order
  * choose feature order   -- columns are MolFusion's index ordering

sklearn only ever receives a count matrix and returns weighted rows. It is
a dev-only dependency; production never imports it, and these tests skip
cleanly where it is absent.
"""

import numpy as np
import pytest

from molfusion_backend.smiles_tokenizer import tokenize_smiles
from molfusion_backend.tfidf import weighting
from molfusion_backend.tfidf.idf import compute_idf
from molfusion_backend.tfidf.ngrams import document_ngram_counts_over_orders
from molfusion_backend.tfidf.transform import TfidfTransformer
from molfusion_backend.tfidf.vocabulary import select_vocabulary

sklearn = pytest.importorskip(
    "sklearn", reason="scikit-learn is a dev-only reference dependency"
)
from sklearn.feature_extraction.text import TfidfTransformer as SklearnTfidfTransformer  # noqa: E402

# The frozen MolFusion contract corresponds exactly to these settings
# applied to a count matrix.
REFERENCE_KWARGS = dict(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)

DOCUMENTS = [
    "CCO",
    "CCN",
    "CCC",
    "CC(=O)O",
    "CC(=O)N",
    "c1ccccc1",
    "c1ccccc1C",
    "CCOCC",
    "CC(C)C",
    "C[C@H](N)C(=O)O",
    "CCCl",
    "CCBr",
    "OCCO",
    "NCCN",
    "CCCCCCCCCC",
]


def _corpus_counts(orders=(1, 2, 3)):
    """Document frequencies over the fixture corpus, MolFusion's way."""
    document_frequency: dict[tuple[str, ...], int] = {}
    tokenized = [tokenize_smiles(smiles) for smiles in DOCUMENTS]
    for tokens in tokenized:
        for ngram in document_ngram_counts_over_orders(tokens, orders):
            document_frequency[ngram] = document_frequency.get(ngram, 0) + 1
    return tokenized, document_frequency


@pytest.fixture()
def fixture_artifact():
    tokenized, document_frequency = _corpus_counts()
    vocabulary, _ = select_vocabulary(document_frequency, min_df=2, max_features=64)
    idf = compute_idf(vocabulary.document_frequencies(), len(DOCUMENTS))
    transformer = TfidfTransformer(
        index_map=vocabulary.index_map(),
        idf=idf,
        dimension=vocabulary.dimension,
        orders=(1, 2, 3),
    )
    counts = np.zeros((len(tokenized), vocabulary.dimension), dtype=np.float64)
    for row, tokens in enumerate(tokenized):
        counts[row] = transformer.counts(tokens)
    return vocabulary, idf, transformer, counts


def test_sklearn_is_available_as_a_reference():
    """If this fails the parity tests below are silently skipped, so the
    absence is asserted loudly rather than discovered later."""
    assert sklearn.__version__


def test_idf_matches_sklearn_idf(fixture_artifact):
    """sklearn's `idf_` is exactly ln((1+N)/(1+df)) + 1 under smooth_idf."""
    _, idf, _, counts = fixture_artifact
    reference = SklearnTfidfTransformer(**REFERENCE_KWARGS).fit(counts)
    assert reference.idf_.shape == idf.shape
    assert reference.idf_ == pytest.approx(idf, rel=1e-12, abs=1e-12)


def test_transformed_vectors_match_sklearn(fixture_artifact):
    _, _, transformer, counts = fixture_artifact
    reference = SklearnTfidfTransformer(**REFERENCE_KWARGS).fit(counts)
    expected = np.asarray(reference.transform(counts).todense(), dtype=np.float64)

    produced = np.vstack(
        [transformer.transform(tokenize_smiles(smiles)) for smiles in DOCUMENTS]
    ).astype(np.float64)

    assert produced.shape == expected.shape
    # float32 output is the only source of divergence; the tolerance is the
    # width of that cast, not slack in the formulas.
    assert np.max(np.abs(produced - expected)) < 1e-6
    for row in range(len(DOCUMENTS)):
        cosine = float(
            np.dot(produced[row], expected[row])
            / (np.linalg.norm(produced[row]) * np.linalg.norm(expected[row]))
        )
        assert cosine == pytest.approx(1.0, abs=1e-9)


def test_float64_output_matches_sklearn_even_more_tightly(fixture_artifact):
    """With the runtime cast removed, the two implementations agree to
    within float64 rounding -- confirming the formulas, not just the shape."""
    _, idf, _, counts = fixture_artifact
    reference = SklearnTfidfTransformer(**REFERENCE_KWARGS).fit(counts)
    expected = np.asarray(reference.transform(counts).todense(), dtype=np.float64)
    produced = weighting.tfidf(counts, idf, dtype=np.float64)
    assert np.max(np.abs(produced - expected)) < 1e-12


def test_sublinear_tf_setting_is_what_molfusion_implements(fixture_artifact):
    """Turning sublinear_tf off in the reference makes it disagree, which
    shows the parity above is actually testing the TF rule."""
    _, idf, _, counts = fixture_artifact
    raw_reference = SklearnTfidfTransformer(
        norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False
    ).fit(counts)
    expected = np.asarray(raw_reference.transform(counts).todense(), dtype=np.float64)
    produced = weighting.tfidf(counts, idf, dtype=np.float64)
    assert np.max(np.abs(produced - expected)) > 1e-6


def test_smooth_idf_setting_is_what_molfusion_implements(fixture_artifact):
    """Same check for the IDF formula: the unsmoothed reference disagrees."""
    _, _, _, counts = fixture_artifact
    unsmoothed = SklearnTfidfTransformer(
        norm="l2", use_idf=True, smooth_idf=False, sublinear_tf=True
    ).fit(counts)
    smoothed_idf = compute_idf(
        [int(column.astype(bool).sum()) for column in counts.T], len(DOCUMENTS)
    )
    assert np.max(np.abs(unsmoothed.idf_ - smoothed_idf)) > 1e-9


def test_a_zero_row_is_handled_the_same_way(fixture_artifact):
    """sklearn also leaves an all-zero row at zero rather than producing
    NaN, so the zero-vector contract is not a MolFusion peculiarity."""
    _, idf, _, counts = fixture_artifact
    with_zero = np.vstack([counts, np.zeros((1, counts.shape[1]))])
    reference = SklearnTfidfTransformer(**REFERENCE_KWARGS).fit(counts)
    expected = np.asarray(reference.transform(with_zero).todense(), dtype=np.float64)
    produced = weighting.tfidf(with_zero, idf, dtype=np.float64)

    assert np.all(expected[-1] == 0.0)
    assert np.all(produced[-1] == 0.0)
    assert np.all(np.isfinite(produced))
    assert np.max(np.abs(produced - expected)) < 1e-12


def test_sklearn_never_chose_the_vocabulary(fixture_artifact):
    """Guards the premise of these tests: the columns are MolFusion's, in
    MolFusion's lexicographic index order."""
    vocabulary, _, _, _ = fixture_artifact
    tokens = [entry.tokens for entry in vocabulary.entries]
    assert tokens == sorted(tokens)
    assert all(entry.index == position for position, entry in enumerate(vocabulary.entries))


def test_parity_holds_through_the_agent_canonicalization_step():
    """Parity for the pipeline the agent actually runs.

    The agent canonicalizes before tokenizing, so its counts come from
    `canonical_smiles_from_mol` output rather than the input string. This
    builds the whole fixture -- document frequencies, IDF and counts -- from
    those canonical strings, so sklearn and MolFusion are again describing
    the same corpus, and any divergence would be arithmetic rather than
    preprocessing.
    """
    from rdkit import Chem

    from molfusion_backend.chemistry import canonical_smiles_from_mol

    canonical = [
        canonical_smiles_from_mol(Chem.MolFromSmiles(smiles)) for smiles in DOCUMENTS
    ]
    tokenized = [tokenize_smiles(text) for text in canonical]

    document_frequency: dict[tuple[str, ...], int] = {}
    for tokens in tokenized:
        for ngram in document_ngram_counts_over_orders(tokens, (1, 2, 3)):
            document_frequency[ngram] = document_frequency.get(ngram, 0) + 1

    vocabulary, _ = select_vocabulary(document_frequency, min_df=2, max_features=64)
    idf = compute_idf(vocabulary.document_frequencies(), len(canonical))
    transformer = TfidfTransformer(
        index_map=vocabulary.index_map(),
        idf=idf,
        dimension=vocabulary.dimension,
        orders=(1, 2, 3),
    )

    counts = np.vstack([transformer.counts(tokens) for tokens in tokenized])
    reference = SklearnTfidfTransformer(**REFERENCE_KWARGS).fit(counts)
    expected = np.asarray(reference.transform(counts).todense(), dtype=np.float64)

    # sklearn's idf_ is fitted on these same counts, so both sides describe
    # the same corpus; the float32 runtime cast is the only divergence.
    assert reference.idf_ == pytest.approx(idf, rel=1e-12, abs=1e-12)
    produced = np.vstack([transformer.transform(tokens) for tokens in tokenized]).astype(
        np.float64
    )
    assert np.max(np.abs(produced - expected)) < 1e-6
    assert np.max(np.abs(weighting.tfidf(counts, idf, dtype=np.float64) - expected)) < 1e-12
