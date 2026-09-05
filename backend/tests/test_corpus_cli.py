import hashlib
import json

import pytest

from molfusion_backend.corpus.builder import CORPUS_FILENAME, REPORT_FILENAME
from molfusion_backend.corpus.cli import build_parser, main
from tests.chembl_fixture import EXPECTED_DOCUMENTS, create_chembl_fixture


@pytest.fixture
def source_db(tmp_path):
    return create_chembl_fixture(tmp_path / "chembl_fixture.db")


def test_source_db_and_output_dir_are_required():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_defaults_target_chembl_37(source_db, tmp_path):
    args = build_parser().parse_args(
        ["--source-db", str(source_db), "--output-dir", str(tmp_path)]
    )
    assert args.source_release == 37
    assert "chembl_37" in args.source_url
    assert args.force is False
    assert args.allow_tokenizer_failures is False


def test_a_successful_run_writes_both_outputs(tmp_path, source_db):
    output_dir = tmp_path / "out"

    exit_code = main(["--source-db", str(source_db), "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / CORPUS_FILENAME).read_bytes().count(b"\n") == len(EXPECTED_DOCUMENTS)
    report = json.loads((output_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["fit_corpus"]["document_count"] == len(EXPECTED_DOCUMENTS)


def test_rerunning_without_force_exits_nonzero_instead_of_overwriting(tmp_path, source_db):
    output_dir = tmp_path / "out"
    argv = ["--source-db", str(source_db), "--output-dir", str(output_dir)]
    assert main(argv) == 0
    before = (output_dir / CORPUS_FILENAME).read_bytes()

    assert main(argv) == 1

    assert (output_dir / CORPUS_FILENAME).read_bytes() == before


def test_force_permits_a_rebuild(tmp_path, source_db):
    output_dir = tmp_path / "out"
    argv = ["--source-db", str(source_db), "--output-dir", str(output_dir)]
    assert main(argv) == 0
    assert main([*argv, "--force"]) == 0


def test_a_missing_source_database_exits_nonzero(tmp_path):
    exit_code = main(
        ["--source-db", str(tmp_path / "absent.db"), "--output-dir", str(tmp_path / "out")]
    )
    assert exit_code == 1


def test_skipping_checksums_while_asking_for_verification_is_rejected(tmp_path, source_db):
    """The two flags contradict each other; failing fast beats silently
    ignoring one of them."""
    exit_code = main(
        [
            "--source-db", str(source_db),
            "--output-dir", str(tmp_path / "out"),
            "--skip-source-checksums",
            "--expected-db-sha256", "0" * 64,
        ]
    )
    assert exit_code == 2
    assert not (tmp_path / "out" / CORPUS_FILENAME).exists()


def test_a_mismatched_expected_checksum_exits_nonzero(tmp_path, source_db):
    exit_code = main(
        [
            "--source-db", str(source_db),
            "--output-dir", str(tmp_path / "out"),
            "--expected-db-sha256", "0" * 64,
        ]
    )
    assert exit_code == 1
    assert not (tmp_path / "out" / CORPUS_FILENAME).exists()


def test_a_matching_expected_checksum_is_accepted(tmp_path, source_db):
    digest = hashlib.sha256(source_db.read_bytes()).hexdigest()
    exit_code = main(
        [
            "--source-db", str(source_db),
            "--output-dir", str(tmp_path / "out"),
            "--expected-db-sha256", digest,
        ]
    )
    assert exit_code == 0


def test_provenance_flags_reach_the_report(tmp_path, source_db):
    output_dir = tmp_path / "out"
    archive = tmp_path / "chembl_37_sqlite.tar.gz"
    archive.write_bytes(b"archive")

    main(
        [
            "--source-db", str(source_db),
            "--output-dir", str(output_dir),
            "--source-archive", str(archive),
            "--source-release", "37",
            "--source-url", "https://example.invalid/chembl_37_sqlite.tar.gz",
        ]
    )
    report = json.loads((output_dir / REPORT_FILENAME).read_text(encoding="utf-8"))

    assert report["source"]["release"] == 37
    assert report["source"]["url"] == "https://example.invalid/chembl_37_sqlite.tar.gz"
    assert report["source"]["assets"]["archive"]["filename"] == "chembl_37_sqlite.tar.gz"
