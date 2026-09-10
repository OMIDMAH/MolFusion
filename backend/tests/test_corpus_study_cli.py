import json

import pytest

from molfusion_backend.corpus.serialization import write_corpus
from molfusion_backend.corpus.study.cli import build_parser, main
from molfusion_backend.corpus.study.runner import (
    FROZEN_DOCUMENT_COUNT,
    FROZEN_FIT_CORPUS_SHA256,
    STUDY_REPORT_FILENAME,
)

FIXTURE_SMILES = sorted(
    {"CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "N", "O"}
    | {f"C{'C' * index}O" for index in range(1, 80)}
)


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "canonical_smiles.smi"
    sha256, _ = write_corpus(path, FIXTURE_SMILES)
    return path, sha256


def test_corpus_and_output_dir_are_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_defaults_target_the_frozen_corpus(tmp_path):
    args = build_parser().parse_args(
        ["--corpus", str(tmp_path / "c.smi"), "--output-dir", str(tmp_path)]
    )
    assert args.expected_sha256 == FROZEN_FIT_CORPUS_SHA256
    assert args.expected_documents == FROZEN_DOCUMENT_COUNT
    assert args.force is False


def test_a_successful_run_writes_the_study_report(corpus, tmp_path, capsys):
    path, sha256 = corpus
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--corpus",
            str(path),
            "--output-dir",
            str(output_dir),
            "--expected-sha256",
            sha256,
            "--expected-documents",
            str(len(FIXTURE_SMILES)),
            "--progress-every",
            "0",
        ]
    )

    assert exit_code == 0
    report = json.loads((output_dir / STUDY_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["corpus"]["verified_sha256"] == sha256
    assert report["corpus"]["document_count"] == len(FIXTURE_SMILES)
    assert "corpus sha256 verified" in capsys.readouterr().err


def test_a_digest_mismatch_exits_non_zero_without_writing_a_report(corpus, tmp_path, capsys):
    path, _ = corpus
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--corpus",
            str(path),
            "--output-dir",
            str(output_dir),
            "--expected-sha256",
            "0" * 64,
            "--progress-every",
            "0",
        ]
    )

    assert exit_code == 1
    assert "identity mismatch" in capsys.readouterr().err
    assert not (output_dir / STUDY_REPORT_FILENAME).exists()


def test_running_against_the_real_default_digest_fails_on_a_fixture(corpus, tmp_path):
    """The default guard is live, not decorative: a fixture corpus is not
    the frozen corpus and must be refused."""
    path, _ = corpus
    assert (
        main(
            [
                "--corpus",
                str(path),
                "--output-dir",
                str(tmp_path / "out"),
                "--progress-every",
                "0",
            ]
        )
        == 1
    )


def test_zero_expected_documents_disables_the_count_check(corpus, tmp_path):
    path, sha256 = corpus
    assert (
        main(
            [
                "--corpus",
                str(path),
                "--output-dir",
                str(tmp_path / "out"),
                "--expected-sha256",
                sha256,
                "--expected-documents",
                "0",
                "--progress-every",
                "0",
            ]
        )
        == 0
    )
