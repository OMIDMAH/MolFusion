import hashlib

import pytest

from molfusion_backend.corpus.serialization import (
    CORPUS_ENCODING,
    CORPUS_HAS_FINAL_NEWLINE,
    CORPUS_NEWLINE,
    CORPUS_SERIALIZATION_ID,
    corpus_bytes,
    corpus_sha256,
    iter_corpus_lines,
    write_corpus,
)


def test_serialization_contract_constants_are_pinned():
    assert CORPUS_ENCODING == "utf-8"
    assert CORPUS_NEWLINE == "\n"
    assert CORPUS_HAS_FINAL_NEWLINE is True
    assert CORPUS_SERIALIZATION_ID == "utf8_lf_sorted_unique_final_newline_v1"


# ---------------------------------------------------------------------------
# The byte contract
# ---------------------------------------------------------------------------


def test_one_entry_per_line_with_a_final_newline():
    assert corpus_bytes(["CCO", "c1ccccc1"]) == b"CCO\nc1ccccc1\n"


def test_single_entry_corpus_is_terminated():
    assert corpus_bytes(["CCO"]) == b"CCO\n"


def test_empty_corpus_serializes_to_no_bytes_at_all():
    """Zero lines means zero terminators -- not a lone newline, and not a
    BOM."""
    assert corpus_bytes([]) == b""


def test_no_byte_order_mark():
    assert not corpus_bytes(["CCO"]).startswith(b"\xef\xbb\xbf")


def test_no_carriage_returns_anywhere():
    """The whole point of writing bytes rather than text: a Windows build
    must not emit CRLF."""
    assert b"\r" not in corpus_bytes(["CCO", "CCC", "CCN"])


def test_bytes_are_utf8():
    assert corpus_bytes(["CCO"]).decode("utf-8") == "CCO\n"


def test_matches_the_documented_join_expression():
    entries = ["CC(=O)O", "CCO", "c1ccccc1"]
    assert corpus_bytes(entries) == ("\n".join(entries) + "\n").encode("utf-8")


def test_entry_containing_a_newline_is_rejected():
    """A record with an embedded break would silently become two corpus
    documents."""
    with pytest.raises(ValueError, match="line break"):
        corpus_bytes(["CCO\nCCC"])


def test_entry_containing_a_carriage_return_is_rejected():
    with pytest.raises(ValueError, match="line break"):
        corpus_bytes(["CCO\r"])


def test_iter_corpus_lines_yields_one_terminated_line_per_entry():
    assert list(iter_corpus_lines(["CCO", "CCC"])) == [b"CCO\n", b"CCC\n"]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_sha256_matches_an_independently_computed_digest():
    """Expected value comes from hashlib over literal bytes, not from the
    production helper -- otherwise the assertion would be circular."""
    expected = hashlib.sha256(b"CCO\nc1ccccc1\n").hexdigest()
    assert corpus_sha256(["CCO", "c1ccccc1"]) == expected


def test_sha256_of_a_known_tiny_corpus_is_a_hard_coded_constant():
    """Digests computed outside Python entirely -- `printf 'CCO\\n' |
    sha256sum` -- and pinned here, so nothing in this assertion can drift
    along with the implementation. If the byte contract ever changes, this
    fails loudly."""
    assert corpus_sha256(["CCO"]) == (
        "5c9aa2a3024d56c903d547798cbd04ff743433ba0950dbfd8e19238e40651172"
    )
    assert corpus_sha256(["CCO", "c1ccccc1"]) == (
        "f55bceec112a842f5b0dab580060718ff85b388e6738f067fd623cae167935e1"
    )


def test_empty_corpus_hashes_to_the_digest_of_no_bytes():
    assert corpus_sha256([]) == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# Writing to disk
# ---------------------------------------------------------------------------


def test_write_corpus_writes_exactly_the_logical_bytes(tmp_path):
    entries = ["CC(=O)O", "CCO", "c1ccccc1"]
    path = tmp_path / "corpus.smi"

    write_corpus(path, entries)

    assert path.read_bytes() == corpus_bytes(entries)


def test_write_corpus_returns_the_digest_and_size_of_what_it_wrote(tmp_path):
    entries = ["CCO", "c1ccccc1"]
    path = tmp_path / "corpus.smi"

    digest, size = write_corpus(path, entries)

    written = path.read_bytes()
    assert digest == hashlib.sha256(written).hexdigest()
    assert size == len(written)


def test_written_file_has_no_crlf_on_any_platform(tmp_path):
    path = tmp_path / "corpus.smi"
    write_corpus(path, ["CCO", "CCC"])
    assert path.read_bytes() == b"CCO\nCCC\n"


def test_write_corpus_agrees_with_corpus_sha256(tmp_path):
    entries = [f"C{'C' * n}O" for n in range(200)]
    path = tmp_path / "corpus.smi"

    digest, _ = write_corpus(path, entries)

    assert digest == corpus_sha256(entries)


def test_write_corpus_handles_a_corpus_larger_than_one_write_chunk(tmp_path):
    """Exercises the buffered path: the streaming writer must produce the
    same bytes as the single-shot helper."""
    entries = sorted({f"C{'C' * (n % 300)}O{n}" for n in range(20_000)})
    path = tmp_path / "corpus.smi"

    digest, size = write_corpus(path, entries)

    assert size > 1024 * 1024
    assert digest == corpus_sha256(entries)
    assert path.read_bytes() == corpus_bytes(entries)


def test_unicode_entries_round_trip_as_utf8(tmp_path):
    """SMILES are ASCII in practice, but the contract names UTF-8 and must
    behave like it."""
    path = tmp_path / "corpus.smi"
    write_corpus(path, ["CéO"])
    assert path.read_bytes() == "CéO\n".encode("utf-8")
