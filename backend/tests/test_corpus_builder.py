import hashlib
import json
import sqlite3

import pytest
from rdkit import Chem

from molfusion_backend.chemistry import CANONICAL_SMILES_NORMALIZATION_ID
from molfusion_backend.corpus import chembl
from molfusion_backend.corpus.builder import (
    CORPUS_FILENAME,
    CORPUS_ID,
    REPORT_FILENAME,
    REPORT_SCHEMA_VERSION,
    build_corpus,
    describe_source_asset,
    deterministic_report_view,
)
from molfusion_backend.corpus.errors import (
    CorpusBuildError,
    CorpusOutputExistsError,
    CorpusSourceError,
    TokenizerContractViolation,
)
from molfusion_backend.corpus.serialization import corpus_bytes
from molfusion_backend.smiles_tokenizer import SMILES_TOKENIZER_ID, tokenize_smiles
from tests.chembl_fixture import (
    EXPECTED_DOCUMENTS,
    EXPECTED_DUPLICATES,
    EXPECTED_EMPTY_SMILES,
    EXPECTED_NULL_SMILES,
    EXPECTED_PARSE_FAILURES,
    EXPECTED_ROWS_EXAMINED,
    EXPECTED_UNIQUE,
    EXPECTED_VALID_PRE_DEDUP,
    EXPECTED_ZERO_ATOM,
    FIXTURE_ROWS,
    create_chembl_fixture,
)


@pytest.fixture
def source_db(tmp_path):
    return create_chembl_fixture(tmp_path / "chembl_fixture.db")


@pytest.fixture
def built(tmp_path, source_db):
    """One build of the standard fixture, reused by most assertions."""
    output_dir = tmp_path / "out"
    report = build_corpus(source_db=source_db, output_dir=output_dir)
    return report, output_dir


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_reads_every_structure_row(source_db):
    connection = chembl.open_source_database(source_db)
    try:
        records = list(chembl.iter_source_records(connection))
    finally:
        connection.close()

    assert len(records) == len(FIXTURE_ROWS)
    assert [record.molregno for record in records] == [row[0] for row in FIXTURE_ROWS]
    assert records[0].chembl_id == "CHEMBL1"
    assert records[0].smiles == "CCO"


def test_null_smiles_survives_extraction_as_none(source_db):
    connection = chembl.open_source_database(source_db)
    try:
        by_molregno = {r.molregno: r for r in chembl.iter_source_records(connection)}
    finally:
        connection.close()

    assert by_molregno[12].smiles is None


def test_source_database_is_opened_read_only(source_db):
    connection = chembl.open_source_database(source_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM compound_structures")
    finally:
        connection.close()


def test_missing_source_database_raises_a_clear_error(tmp_path):
    with pytest.raises(CorpusSourceError, match="not found"):
        chembl.open_source_database(tmp_path / "absent.db")


def test_database_without_the_structure_table_is_rejected(tmp_path):
    path = tmp_path / "wrong.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE something_else (id INTEGER)")
    connection.commit()
    connection.close()

    with pytest.raises(CorpusSourceError, match="compound_structures"):
        chembl.open_source_database(path)


def test_structure_table_missing_the_smiles_column_is_rejected(tmp_path):
    path = tmp_path / "partial.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE compound_structures (molregno INTEGER)")
    connection.commit()
    connection.close()

    with pytest.raises(CorpusSourceError, match="canonical_smiles"):
        chembl.open_source_database(path)


def test_builds_without_the_optional_compound_dictionary(tmp_path):
    """chembl_id is used only to make failure reports readable, so a source
    lacking molecule_dictionary must still produce the same corpus."""
    path = create_chembl_fixture(
        tmp_path / "no_dict.db", include_compound_dictionary=False
    )
    report = build_corpus(source_db=path, output_dir=tmp_path / "out")

    assert report["source"]["compound_ids_available"] is False
    assert report["fit_corpus"]["document_count"] == len(EXPECTED_DOCUMENTS)


def test_extraction_query_touches_no_downstream_table(source_db):
    """The leakage policy in one assertion: the SQL must name only the
    structural tables."""
    connection = chembl.open_source_database(source_db)
    try:
        query = chembl.structure_query(chembl.has_compound_dictionary(connection))
    finally:
        connection.close()

    for forbidden in ("activities", "assays", "target", "docs", "max_phase"):
        assert forbidden not in query


# ---------------------------------------------------------------------------
# Canonicalization and the corpus contents
# ---------------------------------------------------------------------------


def test_corpus_contents_match_the_hand_written_expectation(built):
    _report, output_dir = built
    written = (output_dir / CORPUS_FILENAME).read_bytes().decode("utf-8")
    assert written.splitlines() == EXPECTED_DOCUMENTS


def test_uses_the_phase_5fa_normalization_contract(built):
    report, _ = built
    assert report["contract"]["normalization_id"] == CANONICAL_SMILES_NORMALIZATION_ID
    assert report["contract"]["normalization_id"] == "rdkit_canonical_isomeric_smiles_v1"


def test_equivalent_smiles_collapse_to_one_document(built):
    """"CCO" and "OCC" are one molecule; the corpus must contain it once."""
    _report, output_dir = built
    documents = (output_dir / CORPUS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert documents.count("CCO") == 1


def test_kekule_and_aromatic_input_collapse_to_one_document(built):
    _report, output_dir = built
    documents = (output_dir / CORPUS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert documents.count("c1ccccc1") == 1
    assert "C1=CC=CC=C1" not in documents


def test_distinct_stereoisomers_are_not_merged(built):
    _report, output_dir = built
    documents = (output_dir / CORPUS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert "C[C@H](O)c1ccccc1" in documents
    assert "C[C@@H](O)c1ccccc1" in documents


def test_salts_are_not_stripped(built):
    _report, output_dir = built
    documents = (output_dir / CORPUS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert "CC(=O)[O-].[Na+]" in documents


def test_isotopes_and_charges_are_preserved(built):
    _report, output_dir = built
    documents = (output_dir / CORPUS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert "[13CH3]CO" in documents
    assert "C[N+](C)(C)C" in documents


# ---------------------------------------------------------------------------
# Tokenizer integration
# ---------------------------------------------------------------------------


def test_every_corpus_document_satisfies_the_lossless_invariant(built):
    _report, output_dir = built
    for document in (output_dir / CORPUS_FILENAME).read_text(encoding="utf-8").splitlines():
        assert "".join(tokenize_smiles(document)) == document


def test_report_records_the_tokenizer_contract(built):
    report, _ = built
    assert report["contract"]["tokenizer_id"] == SMILES_TOKENIZER_ID


def test_a_lossy_tokenization_aborts_the_build(tmp_path, source_db, monkeypatch):
    """A tokenizer failure on our own normalizer's output is a contract
    breach, not input noise -- it must stop the build, not quietly drop the
    record."""
    monkeypatch.setattr(
        "molfusion_backend.corpus.builder.tokenize_smiles",
        lambda smiles: ("X",),
    )

    with pytest.raises(TokenizerContractViolation, match="lossy"):
        build_corpus(source_db=source_db, output_dir=tmp_path / "out")


def test_a_tokenizer_exception_aborts_the_build(tmp_path, source_db, monkeypatch):
    def explode(smiles):
        raise ValueError("unrecognized syntax")

    monkeypatch.setattr("molfusion_backend.corpus.builder.tokenize_smiles", explode)

    with pytest.raises(TokenizerContractViolation, match="could not be tokenized"):
        build_corpus(source_db=source_db, output_dir=tmp_path / "out")


def test_a_tokenizer_abort_names_the_offending_chembl_record(tmp_path, source_db, monkeypatch):
    """A contract violation must be traceable back to a specific ChEMBL
    record, not just to an anonymous SMILES string."""
    monkeypatch.setattr(
        "molfusion_backend.corpus.builder.tokenize_smiles", lambda smiles: ("X",)
    )

    with pytest.raises(TokenizerContractViolation) as exc_info:
        build_corpus(source_db=source_db, output_dir=tmp_path / "out")

    message = str(exc_info.value)
    assert "molregno=1" in message
    assert "CHEMBL1" in message
    assert "CCO" in message


def test_an_aborted_build_leaves_no_output_behind(tmp_path, source_db, monkeypatch):
    monkeypatch.setattr(
        "molfusion_backend.corpus.builder.tokenize_smiles", lambda smiles: ("X",)
    )
    output_dir = tmp_path / "out"

    with pytest.raises(TokenizerContractViolation):
        build_corpus(source_db=source_db, output_dir=output_dir)

    assert not (output_dir / CORPUS_FILENAME).exists()
    assert not (output_dir / REPORT_FILENAME).exists()


def test_tokenizer_failures_can_be_allowed_but_are_recorded(tmp_path, source_db, monkeypatch):
    """The documented-exception escape hatch must never let a lenient build
    masquerade as a clean one."""
    real = tokenize_smiles

    def fail_one(smiles):
        if smiles == "CCO":
            raise ValueError("simulated lexer failure")
        return real(smiles)

    monkeypatch.setattr("molfusion_backend.corpus.builder.tokenize_smiles", fail_one)

    report = build_corpus(
        source_db=source_db,
        output_dir=tmp_path / "out",
        allow_tokenizer_failures=True,
    )

    assert report["build"]["tokenizer_failures_allowed"] is True
    # Two source rows canonicalize to "CCO" ("CCO" and "OCC"), and each is
    # excluded on its own -- the exclusion is per record, before dedup.
    assert report["counts"]["tokenization_failures"] == 2
    assert report["counts"]["document_count"] == len(EXPECTED_DOCUMENTS) - 1


def test_a_clean_build_records_that_failures_were_not_allowed(built):
    report, _ = built
    assert report["build"]["tokenizer_failures_allowed"] is False
    assert report["counts"]["tokenization_failures"] == 0


# ---------------------------------------------------------------------------
# Counts: nothing disappears silently
# ---------------------------------------------------------------------------


def test_every_excluded_record_is_accounted_for(built):
    report, _ = built
    counts = report["counts"]

    assert counts["rows_examined"] == EXPECTED_ROWS_EXAMINED
    assert counts["null_smiles"] == EXPECTED_NULL_SMILES
    assert counts["empty_smiles"] == EXPECTED_EMPTY_SMILES
    assert counts["rdkit_parse_failures"] == EXPECTED_PARSE_FAILURES
    assert counts["zero_atom_molecules"] == EXPECTED_ZERO_ATOM
    assert counts["valid_pre_dedup"] == EXPECTED_VALID_PRE_DEDUP
    assert counts["duplicate_canonical_smiles"] == EXPECTED_DUPLICATES
    assert counts["unique_canonical_smiles"] == EXPECTED_UNIQUE
    assert counts["document_count"] == len(EXPECTED_DOCUMENTS)


def test_record_accounting_balances(built):
    report, _ = built
    counts = report["counts"]

    excluded = (
        counts["null_smiles"]
        + counts["empty_smiles"]
        + counts["rdkit_parse_failures"]
        + counts["zero_atom_molecules"]
        + counts["tokenization_failures"]
    )
    assert excluded + counts["valid_pre_dedup"] == counts["rows_examined"]
    assert (
        counts["duplicate_canonical_smiles"] + counts["unique_canonical_smiles"]
        == counts["valid_pre_dedup"]
    )
    assert counts["unique_canonical_smiles"] == counts["document_count"]


def test_whitespace_only_smiles_counts_as_empty_not_a_parse_failure(built):
    """"   " is empty in substance; classifying it as a parse failure would
    misattribute a data-quality category."""
    report, _ = built
    assert report["counts"]["empty_smiles"] == 2


def test_empty_source_smiles_never_becomes_a_document(tmp_path):
    """Phase 5F-A deliberately allows "" -> "" at the helper level. The
    corpus must not inherit that: an empty structure is excluded and
    counted, never emitted as an empty line."""
    rows = [(1, "CHEMBL1", "CCO"), (2, "CHEMBL2", ""), (3, "CHEMBL3", "   ")]
    path = create_chembl_fixture(tmp_path / "z.db", rows=rows)

    report = build_corpus(source_db=path, output_dir=tmp_path / "out")

    assert report["counts"]["empty_smiles"] == 2
    assert report["counts"]["document_count"] == 1
    assert (tmp_path / "out" / CORPUS_FILENAME).read_bytes() == b"CCO\n"


def test_a_zero_atom_molecule_is_excluded_and_counted(tmp_path, monkeypatch):
    """The zero-atom guard is defensive -- with the empty/whitespace check
    in front of it, no current RDKit input reaches it -- so it is exercised
    here by forcing the parser to hand back a zero-atom molecule."""
    def parse_zero_atom(smiles):
        return Chem.MolFromSmiles("" if smiles == "TRIGGER" else smiles), None

    monkeypatch.setattr("molfusion_backend.corpus.builder.parse_smiles", parse_zero_atom)

    rows = [(1, "CHEMBL1", "CCO"), (2, "CHEMBL2", "TRIGGER")]
    path = create_chembl_fixture(tmp_path / "zero.db", rows=rows)

    report = build_corpus(source_db=path, output_dir=tmp_path / "out")

    assert report["counts"]["zero_atom_molecules"] == 1
    assert report["counts"]["document_count"] == 1
    assert (tmp_path / "out" / CORPUS_FILENAME).read_bytes() == b"CCO\n"


def test_a_corpus_with_no_usable_structures_is_refused(tmp_path):
    rows = [(1, "CHEMBL1", None), (2, "CHEMBL2", "not_a_molecule")]
    path = create_chembl_fixture(tmp_path / "empty.db", rows=rows)

    with pytest.raises(CorpusBuildError, match="empty corpus"):
        build_corpus(source_db=path, output_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# Sorting and serialization
# ---------------------------------------------------------------------------


def test_corpus_is_byte_for_byte_lexically_sorted(built):
    _report, output_dir = built
    written = (output_dir / CORPUS_FILENAME).read_bytes()
    assert written == corpus_bytes(EXPECTED_DOCUMENTS)


def test_corpus_lines_are_in_nondecreasing_order(built):
    _report, output_dir = built
    documents = (output_dir / CORPUS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert documents == sorted(documents)


def test_written_corpus_uses_lf_only_and_ends_with_one(built):
    _report, output_dir = built
    written = (output_dir / CORPUS_FILENAME).read_bytes()

    assert b"\r" not in written
    assert written.endswith(b"\n")
    assert not written.endswith(b"\n\n")
    assert not written.startswith(b"\xef\xbb\xbf")
    assert written.count(b"\n") == len(EXPECTED_DOCUMENTS)


def test_report_records_the_serialization_contract(built):
    report, _ = built
    contract = report["contract"]

    assert contract["encoding"] == "utf-8"
    assert contract["newline"] == "\n"
    assert contract["final_newline"] is True
    assert contract["deduplication_key"] == "canonical_isomeric_smiles"
    assert contract["sort"] == "lexicographic_unicode_codepoint"


# ---------------------------------------------------------------------------
# fit_corpus_sha256
# ---------------------------------------------------------------------------


def test_fit_corpus_sha256_is_the_digest_of_the_written_bytes(built):
    """Computed here with hashlib over the file's raw bytes, independently
    of the production hashing path."""
    report, output_dir = built
    written = (output_dir / CORPUS_FILENAME).read_bytes()

    assert report["fit_corpus"]["sha256"] == hashlib.sha256(written).hexdigest()
    assert report["fit_corpus"]["size_bytes"] == len(written)


def test_fit_corpus_sha256_of_the_fixture_is_a_hard_coded_constant(built):
    """End-to-end pin. The byte literal is spelled out here and its digest
    was produced outside Python (`printf ... | sha256sum`), so neither side
    of this assertion can drift along with the builder."""
    report, output_dir = built
    expected_bytes = (
        b"CC(=O)Oc1ccccc1C(=O)O\n"
        b"CC(=O)[O-].[Na+]\n"
        b"CCO\n"
        b"C[C@@H](O)c1ccccc1\n"
        b"C[C@H](O)c1ccccc1\n"
        b"C[N+](C)(C)C\n"
        b"[13CH3]CO\n"
        b"c1ccccc1\n"
    )
    expected_sha256 = "9c8ff518ca4a63062c4988bbc9d0c4c27b7b8e43e9b25d32aba31fba7dbd842e"

    assert (output_dir / CORPUS_FILENAME).read_bytes() == expected_bytes
    assert report["fit_corpus"]["sha256"] == expected_sha256
    assert corpus_bytes(EXPECTED_DOCUMENTS) == expected_bytes


def test_fit_corpus_sha256_differs_from_the_source_database_sha256(built):
    """The two hashes identify different things and must never be
    conflated."""
    report, _ = built
    assert report["fit_corpus"]["sha256"] != report["source"]["assets"]["database"]["sha256"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_builds_in_separate_directories_are_identical(tmp_path, source_db):
    first = build_corpus(source_db=source_db, output_dir=tmp_path / "a")
    second = build_corpus(source_db=source_db, output_dir=tmp_path / "b")

    assert (tmp_path / "a" / CORPUS_FILENAME).read_bytes() == (
        tmp_path / "b" / CORPUS_FILENAME
    ).read_bytes()
    assert first["fit_corpus"]["sha256"] == second["fit_corpus"]["sha256"]
    assert first["counts"] == second["counts"]
    assert first["statistics"] == second["statistics"]
    assert deterministic_report_view(first) == deterministic_report_view(second)


def test_only_the_timestamp_is_allowed_to_differ_between_builds(tmp_path, source_db):
    first = build_corpus(source_db=source_db, output_dir=tmp_path / "a")
    second = build_corpus(source_db=source_db, output_dir=tmp_path / "b")

    assert "built_at" in first["build"]
    assert "built_at" not in deterministic_report_view(first)["build"]
    assert deterministic_report_view(first) == deterministic_report_view(second)


def test_source_row_order_does_not_change_the_corpus(tmp_path):
    """The same molecules inserted in a different order must produce
    byte-identical output -- SQLite row order must not leak into the
    corpus."""
    forward = create_chembl_fixture(tmp_path / "forward.db", rows=FIXTURE_ROWS)
    reverse_rows = [
        (index, chembl_id, smiles)
        for index, (_molregno, chembl_id, smiles) in enumerate(reversed(FIXTURE_ROWS), start=1)
    ]
    reversed_db = create_chembl_fixture(tmp_path / "reverse.db", rows=reverse_rows)

    first = build_corpus(source_db=forward, output_dir=tmp_path / "a")
    second = build_corpus(source_db=reversed_db, output_dir=tmp_path / "b")

    assert (tmp_path / "a" / CORPUS_FILENAME).read_bytes() == (
        tmp_path / "b" / CORPUS_FILENAME
    ).read_bytes()
    assert first["fit_corpus"]["sha256"] == second["fit_corpus"]["sha256"]
    assert first["counts"] == second["counts"]


def test_output_directory_path_does_not_affect_the_corpus(tmp_path, source_db):
    shallow = build_corpus(source_db=source_db, output_dir=tmp_path / "x")
    deep = build_corpus(
        source_db=source_db, output_dir=tmp_path / "a" / "much" / "deeper" / "path"
    )
    assert shallow["fit_corpus"]["sha256"] == deep["fit_corpus"]["sha256"]


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------


def test_rebuilding_over_an_existing_corpus_fails_by_default(tmp_path, source_db):
    output_dir = tmp_path / "out"
    build_corpus(source_db=source_db, output_dir=output_dir)

    with pytest.raises(CorpusOutputExistsError, match="Refusing to overwrite"):
        build_corpus(source_db=source_db, output_dir=output_dir)


def test_a_refused_rebuild_leaves_the_existing_corpus_untouched(tmp_path, source_db):
    output_dir = tmp_path / "out"
    build_corpus(source_db=source_db, output_dir=output_dir)
    before = (output_dir / CORPUS_FILENAME).read_bytes()

    with pytest.raises(CorpusOutputExistsError):
        build_corpus(source_db=source_db, output_dir=output_dir)

    assert (output_dir / CORPUS_FILENAME).read_bytes() == before


def test_force_allows_an_intentional_rebuild(tmp_path, source_db):
    output_dir = tmp_path / "out"
    first = build_corpus(source_db=source_db, output_dir=output_dir)
    second = build_corpus(source_db=source_db, output_dir=output_dir, force=True)

    assert first["fit_corpus"]["sha256"] == second["fit_corpus"]["sha256"]


def test_a_failed_forced_rebuild_does_not_corrupt_the_existing_corpus(
    tmp_path, source_db, monkeypatch
):
    """--force must not be able to leave mixed old/new payloads: staging
    then replacing means a mid-build failure never touches the finalized
    files."""
    output_dir = tmp_path / "out"
    build_corpus(source_db=source_db, output_dir=output_dir)
    before_corpus = (output_dir / CORPUS_FILENAME).read_bytes()
    before_report = (output_dir / REPORT_FILENAME).read_bytes()

    monkeypatch.setattr(
        "molfusion_backend.corpus.builder.tokenize_smiles", lambda smiles: ("X",)
    )
    with pytest.raises(TokenizerContractViolation):
        build_corpus(source_db=source_db, output_dir=output_dir, force=True)

    assert (output_dir / CORPUS_FILENAME).read_bytes() == before_corpus
    assert (output_dir / REPORT_FILENAME).read_bytes() == before_report


def test_no_staging_directory_is_left_behind(built):
    _report, output_dir = built
    leftovers = [path.name for path in output_dir.iterdir() if path.name.startswith(".build-")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_source_database_is_hashed_and_sized(built, source_db):
    report, _ = built
    database = report["source"]["assets"]["database"]

    assert database["filename"] == source_db.name
    assert database["sha256"] == hashlib.sha256(source_db.read_bytes()).hexdigest()
    assert database["size_bytes"] == source_db.stat().st_size


def test_source_checksums_can_be_skipped_and_the_report_says_so(tmp_path, source_db):
    report = build_corpus(
        source_db=source_db,
        output_dir=tmp_path / "out",
        compute_source_checksums=False,
    )
    assert report["source"]["assets"]["database"]["sha256"] is None
    assert report["source"]["assets"]["database"]["size_bytes"] > 0


def test_an_archive_is_recorded_as_a_separate_asset(tmp_path, source_db):
    archive = tmp_path / "chembl_37_sqlite.tar.gz"
    archive.write_bytes(b"pretend archive bytes")

    report = build_corpus(
        source_db=source_db, output_dir=tmp_path / "out", source_archive=archive
    )
    assets = report["source"]["assets"]

    assert assets["archive"]["filename"] == "chembl_37_sqlite.tar.gz"
    assert assets["archive"]["sha256"] == hashlib.sha256(b"pretend archive bytes").hexdigest()
    assert assets["archive"]["sha256"] != assets["database"]["sha256"]


def test_matching_expected_checksum_is_recorded_as_verified(tmp_path, source_db):
    digest = hashlib.sha256(source_db.read_bytes()).hexdigest()
    report = build_corpus(
        source_db=source_db, output_dir=tmp_path / "out", expected_db_sha256=digest
    )
    assert report["source"]["assets"]["database"]["checksum_verified"] is True


def test_mismatched_expected_checksum_refuses_to_build(tmp_path, source_db):
    with pytest.raises(CorpusBuildError, match="checksum verification failed"):
        build_corpus(
            source_db=source_db,
            output_dir=tmp_path / "out",
            expected_db_sha256="0" * 64,
        )


def test_verification_status_is_null_when_nothing_was_supplied(built):
    report, _ = built
    assert report["source"]["assets"]["database"]["checksum_verified"] is None


def test_describe_source_asset_rejects_a_missing_file(tmp_path):
    with pytest.raises(CorpusBuildError, match="not found"):
        describe_source_asset("database", tmp_path / "nope.db")


def test_report_records_the_release_and_a_concrete_versioned_url(built):
    report, _ = built
    assert report["source"]["name"] == "ChEMBL"
    assert report["source"]["release"] == 37
    assert "chembl_37" in report["source"]["url"]
    assert "latest" not in report["source"]["url"]


def test_report_declares_that_no_downstream_labels_were_used(built):
    report, _ = built
    assert report["source"]["uses_downstream_labels"] is False


# ---------------------------------------------------------------------------
# Report shape and statistics
# ---------------------------------------------------------------------------


def test_report_is_written_as_utf8_json_with_lf_only(built):
    _report, output_dir = built
    raw = (output_dir / REPORT_FILENAME).read_bytes()

    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    json.loads(raw.decode("utf-8"))


def test_written_report_matches_the_returned_report(built):
    report, output_dir = built
    on_disk = json.loads((output_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == report


def test_report_identifies_itself(built):
    report, _ = built
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["corpus_id"] == CORPUS_ID


def test_report_records_the_software_versions(built):
    report, _ = built
    software = report["build"]["software"]

    assert software["python"].startswith("3.11")
    assert software["rdkit"]
    assert software["sqlite"]


def test_statistics_describe_the_corpus(built):
    report, _ = built
    statistics = report["statistics"]

    assert statistics["document_count"] == len(EXPECTED_DOCUMENTS)
    assert statistics["smiles_length"]["min"] == min(len(s) for s in EXPECTED_DOCUMENTS)
    assert statistics["smiles_length"]["max"] == max(len(s) for s in EXPECTED_DOCUMENTS)
    assert statistics["token_count"]["min"] >= 1
    assert statistics["token_count"]["max"] >= statistics["token_count"]["min"]


def test_statistics_count_salts_and_stereochemistry(built):
    report, _ = built
    statistics = report["statistics"]

    assert statistics["with_disconnected_components"] == 1  # CC(=O)[O-].[Na+]
    assert statistics["with_stereochemistry"] == 2  # the two stereoisomers


def test_report_contains_no_ngram_or_tfidf_analysis(built):
    """Phase 5F-B is corpus construction only; vocabulary work is 5F-C."""
    serialized = json.dumps(built[0])
    for term in ("ngram", "n_gram", "tfidf", "vocabulary", "idf", "min_df", "max_features"):
        assert term not in serialized.lower()
