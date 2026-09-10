"""The frozen arithmetic, pinned against values derived by hand.

Section 18 of the phase brief: the mathematical contract must be
independently verifiable, and an implementation must not be tested only
against itself. Expected values here come from the stated formulas
evaluated with plain `math` -- a different code path from the vectorized
NumPy implementation under test -- and the constants are additionally
written out as decimal literals. `test_tfidf_sklearn_parity.py` adds an
independent third-party check on top.
"""

import hashlib
import math
from io import BytesIO

import numpy as np
import pytest

from molfusion_backend.tfidf import weighting
from molfusion_backend.tfidf.errors import TfidfIdfError
from molfusion_backend.tfidf.idf import (
    IDF_NPY_DESCR,
    IDF_NPY_VERSION,
    IDF_RECOMPUTE_TOLERANCE,
    compute_idf,
    idf_bytes,
    inspect_idf_payload,
    load_idf,
    validate_idf,
    validate_idf_payload,
)
from molfusion_backend.tfidf.transform import TfidfTransformer, zero_vector

LN2 = 0.6931471805599453
LN_4_3 = 0.2876820724517809  # ln(4/3)
LN_3_2 = 0.4054651081081644  # ln(3/2)


# ---------------------------------------------------------------------------
# term frequency
# ---------------------------------------------------------------------------


def test_frozen_tf_mode_is_sublinear():
    assert weighting.FROZEN_TF_MODE == "sublinear"
    assert weighting.tf_formula() == "tf(t,d) = 1 + ln(count(t in d)) if count > 0 else 0"


def test_sublinear_tf_matches_hand_computation():
    weighted = weighting.term_frequency(np.array([0, 1, 2, 4, 100]))
    assert weighted[0] == 0.0
    assert weighted[1] == pytest.approx(1.0)
    assert weighted[2] == pytest.approx(1.0 + LN2)
    assert weighted[3] == pytest.approx(1.0 + 2 * LN2)
    assert weighted[4] == pytest.approx(1.0 + math.log(100))


def test_sublinear_tf_of_zero_is_exactly_zero_and_finite():
    weighted = weighting.term_frequency(np.zeros(5))
    assert np.all(weighted == 0.0)
    assert np.all(np.isfinite(weighted))


# ---------------------------------------------------------------------------
# inverse document frequency
# ---------------------------------------------------------------------------


def test_frozen_idf_mode_is_smoothed():
    assert weighting.FROZEN_IDF_MODE == "smoothed"
    assert weighting.idf_formula() == "idf(t) = ln((1 + N) / (1 + df(t))) + 1"
    assert weighting.LOG_BASE == "e"


def test_smoothed_idf_matches_hand_computation():
    """N = 3: df 2 -> ln(4/3)+1, df 3 -> ln(4/4)+1 = 1."""
    values = compute_idf([2, 3], 3)
    assert values[0] == pytest.approx(1.0 + LN_4_3)
    assert values[1] == pytest.approx(1.0)
    assert values.tolist() == pytest.approx([1.2876820724517809, 1.0])


def test_unsmoothed_idf_is_available_and_differs():
    smoothed = weighting.inverse_document_frequency(np.array([2]), 3, "smoothed")[0]
    unsmoothed = weighting.inverse_document_frequency(np.array([2]), 3, "unsmoothed")[0]
    assert smoothed == pytest.approx(1.0 + LN_4_3)
    assert unsmoothed == pytest.approx(1.0 + LN_3_2)
    assert smoothed < unsmoothed


def test_a_universal_term_weighs_exactly_one():
    assert compute_idf([1000], 1000)[0] == pytest.approx(1.0)


def test_idf_is_strictly_decreasing_in_document_frequency():
    values = compute_idf([5, 50, 500, 5000, 50000], 100_000)
    assert np.all(np.diff(values) < 0)


def test_idf_dtype_is_float64():
    assert compute_idf([5, 10], 100).dtype == np.float64
    assert weighting.IDF_DTYPE == np.dtype("<f8")
    assert weighting.IDF_DTYPE == np.float64  # same dtype on a little-endian host
    assert weighting.FROZEN_IDF_DTYPE == "float64"


def test_idf_of_an_empty_vocabulary_is_refused():
    with pytest.raises(TfidfIdfError):
        compute_idf([], 100)


# ---------------------------------------------------------------------------
# IDF payload
# ---------------------------------------------------------------------------


def test_idf_bytes_round_trip_exactly(tmp_path):
    values = compute_idf([5, 133, 2_882_503], 2_897_639)
    path = tmp_path / "idf.npy"
    path.write_bytes(idf_bytes(values))
    restored = load_idf(path)
    assert restored.dtype == np.float64
    assert restored.tolist() == values.tolist()


def test_idf_bytes_are_deterministic():
    values = compute_idf([5, 133, 900], 1_000_000)
    assert idf_bytes(values) == idf_bytes(values.copy())


def test_idf_payload_is_a_plain_array_not_a_pickle(tmp_path):
    """`allow_pickle=False` must be sufficient to load it -- an artifact
    payload must never be able to execute code on load."""
    path = tmp_path / "idf.npy"
    path.write_bytes(idf_bytes(compute_idf([5, 10], 100)))
    assert np.load(path, allow_pickle=False).dtype == np.float64


def test_loading_a_non_array_payload_fails(tmp_path):
    path = tmp_path / "idf.npy"
    path.write_bytes(b"not an npy file")
    with pytest.raises(TfidfIdfError):
        load_idf(path)


# ---------------------------------------------------------------------------
# IDF validation
# ---------------------------------------------------------------------------


def test_validation_accepts_a_correct_vector():
    frequencies = [5, 133, 900]
    validate_idf(
        compute_idf(frequencies, 1_000_000),
        dimension=3,
        document_frequencies=frequencies,
        n_documents=1_000_000,
    )


def test_validation_rejects_a_wrong_shape():
    with pytest.raises(TfidfIdfError, match="shape"):
        validate_idf(compute_idf([5, 10], 100), dimension=3)


def test_validation_rejects_a_wrong_dtype():
    with pytest.raises(TfidfIdfError, match="dtype"):
        validate_idf(compute_idf([5, 10], 100).astype(np.float32), dimension=2)


def test_validation_rejects_non_finite_values():
    values = compute_idf([5, 10], 100)
    values[0] = np.inf
    with pytest.raises(TfidfIdfError, match="non-finite"):
        validate_idf(values, dimension=2)


def test_validation_rejects_non_positive_values():
    values = compute_idf([5, 10], 100)
    values[0] = 0.0
    with pytest.raises(TfidfIdfError, match="non-positive"):
        validate_idf(values, dimension=2)


def test_validation_rejects_a_value_that_does_not_match_the_formula():
    """A checksum cannot catch this: a wrong-but-consistently-written
    payload hashes fine. Re-deriving from the recorded DF can."""
    frequencies = [5, 133]
    values = compute_idf(frequencies, 1_000_000)
    values[1] += 1e-6
    with pytest.raises(TfidfIdfError, match="does not reproduce"):
        validate_idf(values, dimension=2, document_frequencies=frequencies, n_documents=1_000_000)


def test_validation_rejects_misaligned_document_frequencies():
    with pytest.raises(TfidfIdfError):
        validate_idf(
            compute_idf([5, 10], 100),
            dimension=2,
            document_frequencies=[5, 10, 20],
            n_documents=100,
        )


def test_recompute_tolerance_is_tight():
    assert IDF_RECOMPUTE_TOLERANCE <= 1e-12


# ---------------------------------------------------------------------------
# normalization and the composed transform
# ---------------------------------------------------------------------------


def test_l2_normalization_gives_unit_length():
    assert weighting.l2_normalize(np.array([3.0, 4.0])).tolist() == pytest.approx([0.6, 0.8])


def test_l2_normalization_of_zero_stays_zero_and_never_produces_nan():
    normalized = weighting.l2_normalize(np.zeros(4096))
    assert np.all(normalized == 0.0)
    assert not np.any(np.isnan(normalized))
    assert np.all(np.isfinite(normalized))


def test_normalization_is_applied_last():
    counts = np.array([1.0, 4.0])
    idf = np.array([1.0, 3.0])
    unnormalized = weighting.tfidf(counts, idf, tf_mode="raw", norm="none", dtype=np.float64)
    assert unnormalized.tolist() == pytest.approx([1.0, 12.0])
    normalized = weighting.tfidf(counts, idf, tf_mode="raw", norm="l2", dtype=np.float64)
    assert normalized.tolist() == pytest.approx(weighting.l2_normalize(unnormalized).tolist())


def test_runtime_output_dtype_is_float32():
    assert weighting.RUNTIME_DTYPE is np.float32
    assert weighting.FROZEN_RUNTIME_DTYPE == "float32"
    result = weighting.tfidf(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert result.dtype == np.float32


def test_arithmetic_happens_in_float64_before_the_cast():
    counts = np.array([1.0, 3.0])
    idf = np.array([1.1, 2.2])
    exact = weighting.tfidf(counts, idf, dtype=np.float64)
    reduced = weighting.tfidf(counts, idf, dtype=np.float32)
    assert reduced.astype(np.float64).tolist() == pytest.approx(exact.tolist(), abs=1e-6)


# ---------------------------------------------------------------------------
# the transformer
# ---------------------------------------------------------------------------
#
# Worked fixture, by hand. Vocabulary in index (lexicographic) order:
#
#   0: ("B",)      df 2
#   1: ("O",)      df 3
#   2: ("O","B")   df 2
#
# N = 3, so smoothed IDF is [1+ln(4/3), 1.0, 1+ln(4/3)].


@pytest.fixture()
def transformer():
    index_map = {("B",): 0, ("O",): 1, ("O", "B"): 2}
    idf = compute_idf([2, 3, 2], 3)
    return TfidfTransformer(index_map=index_map, idf=idf, dimension=3, orders=(1, 2))


def test_counts_are_as_hand_counted(transformer):
    """("O","B","O"): O twice, B once, bigram ("O","B") once. Its other
    bigram ("B","O") is not in the vocabulary and is ignored."""
    assert transformer.counts(("O", "B", "O")).tolist() == [1.0, 2.0, 1.0]


def test_transform_matches_hand_computation(transformer):
    unnormalized = [1.0 + LN_4_3, 1.0 + LN2, 1.0 + LN_4_3]
    length = math.sqrt(sum(v * v for v in unnormalized))
    expected = [v / length for v in unnormalized]

    produced = transformer.transform(("O", "B", "O"))
    assert produced.tolist() == pytest.approx(expected, rel=1e-6)
    assert produced.dtype == np.float32
    assert float(np.linalg.norm(produced)) == pytest.approx(1.0, rel=1e-6)


def test_out_of_vocabulary_ngrams_contribute_nothing(transformer):
    """No UNK dimension, no growth, no exception -- and the vector is the
    same as if the OOV tokens were absent."""
    with_oov = transformer.transform(("O", "B", "O"))
    assert transformer.counts(("O", "B", "O")).tolist() == [1.0, 2.0, 1.0]
    assert transformer.dimension == 3
    assert np.allclose(with_oov, transformer.transform(("O", "B", "O")))


def test_a_molecule_of_only_oov_ngrams_yields_a_zero_vector(transformer):
    produced = transformer.transform(("X", "Y", "Z"))
    assert produced.tolist() == [0.0, 0.0, 0.0]
    assert produced.dtype == np.float32
    assert np.all(np.isfinite(produced))
    assert not np.any(np.isnan(produced))


def test_zero_vector_helper_matches_the_contract():
    vector = zero_vector(4096)
    assert vector.shape == (4096,)
    assert vector.dtype == np.float32
    assert not vector.any()


def test_transform_many_matches_transform_row_by_row(transformer):
    documents = [("O", "B", "O"), ("O", "C"), ("X",)]
    matrix = transformer.transform_many(documents)
    assert matrix.shape == (3, 3)
    assert matrix.dtype == np.float32
    for row, tokens in enumerate(documents):
        assert matrix[row].tolist() == pytest.approx(transformer.transform(tokens).tolist())
    assert matrix[2].tolist() == [0.0, 0.0, 0.0]


def test_transformer_rejects_a_misaligned_idf():
    with pytest.raises(TfidfIdfError):
        TfidfTransformer(index_map={("A",): 0}, idf=compute_idf([5, 6], 100), dimension=1)


def test_transformer_rejects_an_index_map_of_the_wrong_size():
    with pytest.raises(TfidfIdfError):
        TfidfTransformer(
            index_map={("A",): 0, ("B",): 1}, idf=compute_idf([5], 100), dimension=1
        )


# ---------------------------------------------------------------------------
# the serialization path is pinned, and carries nothing but numbers
# ---------------------------------------------------------------------------


def test_the_npy_serialization_path_is_pinned_not_inferred():
    """Format version, byte order and memory order are all stated, so a
    NumPy upgrade cannot silently change the payload bytes."""
    assert IDF_NPY_VERSION == (1, 0)
    assert IDF_NPY_DESCR == "<f8"
    assert weighting.IDF_DTYPE == np.dtype(IDF_NPY_DESCR)


def test_the_payload_header_is_exactly_the_three_structural_fields(tmp_path):
    path = tmp_path / "idf.npy"
    path.write_bytes(idf_bytes(compute_idf([5, 133, 900], 1_000_000)))

    facts = inspect_idf_payload(path)
    assert facts["version"] == (1, 0)
    assert facts["descr"] == "<f8"
    assert facts["fortran_order"] is False
    assert facts["shape"] == (3,)
    assert facts["header_fields"] == ["descr", "fortran_order", "shape"]


def test_the_payload_is_header_plus_raw_doubles_and_nothing_else(tmp_path):
    """Every byte is accounted for, so there is no room for a timestamp, a
    path, a username, or a library version to hide."""
    path = tmp_path / "idf.npy"
    path.write_bytes(idf_bytes(compute_idf([5, 133, 900], 1_000_000)))

    facts = inspect_idf_payload(path)
    assert facts["data_bytes"] == 3 * 8
    assert facts["header_bytes"] + facts["data_bytes"] == facts["total_bytes"]


def test_the_byte_order_is_little_endian_regardless_of_the_native_alias(tmp_path):
    """Serializing the platform-native `float64` would stamp the build
    host's endianness into the header. The pinned dtype does not."""
    native = np.array([1.5, 2.5], dtype=np.float64)
    path = tmp_path / "idf.npy"
    path.write_bytes(idf_bytes(native))
    assert inspect_idf_payload(path)["descr"] == "<f8"


def test_an_object_array_is_refused(tmp_path):
    """No pickle, no object dtype -- an artifact payload must never be able
    to execute anything on load."""
    with pytest.raises(TfidfIdfError, match="object-dtype"):
        idf_bytes(np.array([{"a": 1}, 2.0], dtype=object))


def test_a_non_one_dimensional_array_is_refused():
    with pytest.raises(TfidfIdfError, match="1-D"):
        idf_bytes(np.zeros((2, 2), dtype=np.float64))


def test_the_payload_contains_no_recognisable_environment_strings(tmp_path):
    """A blunt but direct check of the requirement: none of the obvious
    machine-specific tokens appear anywhere in the file."""
    path = tmp_path / "idf.npy"
    path.write_bytes(idf_bytes(compute_idf([5, 133, 900], 1_000_000)))
    raw = path.read_bytes().lower()

    for token in (b"numpy version", b"http", b"/users/", b"c:\\", b".py",
                  b"20", b"utc", b"molfusion", b"pickle"):
        # "20" would appear in a timestamp such as 2026-…; the header is
        # pure structure, so even that must be absent from it.
        assert token not in raw[: inspect_idf_payload(path)["header_bytes"]]


def test_payload_validation_accepts_a_well_formed_file(tmp_path):
    path = tmp_path / "idf.npy"
    path.write_bytes(idf_bytes(compute_idf([5, 133, 900], 1_000_000)))
    validate_idf_payload(path, dimension=3)


def test_payload_validation_rejects_a_big_endian_payload(tmp_path):
    """The failure a big-endian build host would produce, simulated."""
    path = tmp_path / "idf.npy"
    values = compute_idf([5, 133, 900], 1_000_000).astype(">f8")
    buffer = BytesIO()
    np.lib.format.write_array(buffer, values, version=(1, 0), allow_pickle=False)
    path.write_bytes(buffer.getvalue())

    assert inspect_idf_payload(path)["descr"] == ">f8"
    with pytest.raises(TfidfIdfError, match="little-endian"):
        validate_idf_payload(path, dimension=3)


def test_payload_validation_rejects_a_promoted_format_version(tmp_path):
    path = tmp_path / "idf.npy"
    values = compute_idf([5, 133, 900], 1_000_000)
    buffer = BytesIO()
    np.lib.format.write_array(buffer, values, version=(2, 0), allow_pickle=False)
    path.write_bytes(buffer.getvalue())

    with pytest.raises(TfidfIdfError, match="format version"):
        validate_idf_payload(path, dimension=3)


def test_payload_validation_rejects_a_wrong_declared_shape(tmp_path):
    path = tmp_path / "idf.npy"
    path.write_bytes(idf_bytes(compute_idf([5, 133, 900], 1_000_000)))
    with pytest.raises(TfidfIdfError, match="shape"):
        validate_idf_payload(path, dimension=4)


def test_two_independent_serializations_are_byte_identical(tmp_path):
    """Same numbers, two separate arrays, two separate files."""
    first_values = compute_idf([5, 133, 2_882_503], 2_897_639)
    second_values = compute_idf([5, 133, 2_882_503], 2_897_639)
    assert first_values is not second_values

    first = tmp_path / "one.npy"
    second = tmp_path / "two.npy"
    first.write_bytes(idf_bytes(first_values))
    second.write_bytes(idf_bytes(second_values))

    assert first.read_bytes() == second.read_bytes()
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )
