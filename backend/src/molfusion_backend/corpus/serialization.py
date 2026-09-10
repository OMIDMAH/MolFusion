"""The frozen logical-corpus serialization contract.

The "logical corpus" is the exact byte sequence that will later be fed to a
fitting step (TF-IDF first). Its SHA-256 is the corpus's scientific
identity, so the bytes must be reproducible from the sorted unique
canonical SMILES alone -- independent of SQLite physical layout, archive
metadata, source row order, filesystem, and above all the host platform's
text-mode newline translation.

Every write therefore goes through raw binary I/O with explicit b"\\n"
terminators. Nothing here ever opens a file in text mode.
"""

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path

# The serialization contract, frozen. These are exported (and recorded in
# the build report) so a future consumer can assert the corpus it loads was
# produced under the same rules rather than assuming it.
CORPUS_ENCODING = "utf-8"
CORPUS_NEWLINE = "\n"
CORPUS_HAS_FINAL_NEWLINE = True
CORPUS_SERIALIZATION_ID = "utf8_lf_sorted_unique_final_newline_v1"

_LINE_TERMINATOR = b"\n"
_WRITE_CHUNK_BYTES = 1024 * 1024


def iter_corpus_lines(smiles: Iterable[str]) -> Iterator[bytes]:
    """Yield one UTF-8 encoded, LF-terminated line per SMILES.

    The single source of truth for the byte contract: both corpus_bytes()
    and write_corpus() build on this, so they cannot drift apart. Every
    line -- including the last -- is terminated, which is what makes the
    "final newline: yes" rule true by construction rather than by a
    special case appended afterwards.

    A record containing a newline would silently become two corpus
    documents, so it is rejected rather than written.
    """
    for entry in smiles:
        if "\n" in entry or "\r" in entry:
            raise ValueError(
                f"Corpus entry contains a line break and cannot be serialized: {entry!r}"
            )
        yield entry.encode(CORPUS_ENCODING) + _LINE_TERMINATOR


def corpus_bytes(smiles: Iterable[str]) -> bytes:
    """The full logical corpus as one bytes object.

    Equivalent to ("\\n".join(smiles) + "\\n").encode("utf-8") for a
    non-empty corpus, and to b"" for an empty one -- the empty corpus is
    zero lines, so it gets no final newline and no BOM either. Convenient
    for tests and small corpora; use write_corpus() for real ones.
    """
    return b"".join(iter_corpus_lines(smiles))


def corpus_sha256(smiles: Iterable[str]) -> str:
    """SHA-256 of the logical corpus bytes, computed without materializing
    them."""
    digest = hashlib.sha256()
    for line in iter_corpus_lines(smiles):
        digest.update(line)
    return digest.hexdigest()


def write_corpus(path: Path, smiles: Iterable[str]) -> tuple[str, int]:
    """Write the logical corpus to `path` and return (sha256, size_bytes).

    Writes and hashes in one streaming pass, so the returned digest is
    necessarily the digest of the bytes that actually landed on disk --
    they are never computed from two independent traversals that could
    disagree. Opened "wb": no text mode, so no CRLF translation on Windows.
    """
    digest = hashlib.sha256()
    size = 0
    buffer = bytearray()

    with path.open("wb") as handle:
        for line in iter_corpus_lines(smiles):
            digest.update(line)
            size += len(line)
            buffer += line
            if len(buffer) >= _WRITE_CHUNK_BYTES:
                handle.write(buffer)
                buffer.clear()
        if buffer:
            handle.write(buffer)

    return digest.hexdigest(), size
