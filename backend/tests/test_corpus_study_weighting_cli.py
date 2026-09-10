import json

import pytest

from molfusion_backend.corpus.serialization import write_corpus
from molfusion_backend.corpus.study.runner import (
    FROZEN_DOCUMENT_COUNT,
    FROZEN_FIT_CORPUS_SHA256,
)
from molfusion_backend.corpus.study.weighting.cli import build_parser, main
from molfusion_backend.corpus.study.weighting.payload import (
    FROZEN_DIMENSION,
    FROZEN_MIN_DF,
    INDEX_ORDER_LEXICOGRAPHIC,
)
from molfusion_backend.corpus.study.weighting.report import REPORT_FILENAME

FIXTURE_SMILES = sorted(
    {"CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "N", "O"}
    | {f"C{'C' * index}O" for index in range(1, 200)}
    | {f"c1ccccc1{'C' * index}" for index in range(1, 150)}
)


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "canonical_smiles.smi"
    sha256, _ = write_corpus(path, FIXTURE_SMILES)
    return path, sha256


def test_corpus_and_output_dir_are_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_defaults_are_the_frozen_phase_5fc_policy(tmp_path):
    args = build_parser().parse_args(
        ["--corpus", str(tmp_path / "c.smi"), "--output-dir", str(tmp_path)]
    )
    assert args.expected_sha256 == FROZEN_FIT_CORPUS_SHA256
    assert args.expected_documents == FROZEN_DOCUMENT_COUNT
    assert args.min_df == FROZEN_MIN_DF
    assert args.dimension == FROZEN_DIMENSION
    assert args.index_order == INDEX_ORDER_LEXICOGRAPHIC
    assert args.force is False
    assert args.no_cache is False


def test_a_successful_run_writes_the_weighting_report(corpus, tmp_path, capsys):
    path, sha256 = corpus
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--corpus", str(path),
            "--output-dir", str(output_dir),
            "--expected-sha256", sha256,
            "--expected-documents", str(len(FIXTURE_SMILES)),
            "--min-df", "5",
            "--dimension", "48",
            "--progress-every", "0",
        ]
    )

    assert exit_code == 0
    report = json.loads((output_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["corpus"]["verified_sha256"] == sha256
    assert report["produces_production_artifact"] is False
    assert "corpus sha256 verified" in capsys.readouterr().err


def test_a_digest_mismatch_exits_non_zero_without_a_report(corpus, tmp_path, capsys):
    path, _ = corpus
    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "--corpus", str(path),
            "--output-dir", str(output_dir),
            "--expected-sha256", "0" * 64,
            "--progress-every", "0",
        ]
    )
    assert exit_code == 1
    assert "identity mismatch" in capsys.readouterr().err
    assert not (output_dir / REPORT_FILENAME).exists()


def test_the_live_default_digest_refuses_a_fixture_corpus(corpus, tmp_path):
    path, _ = corpus
    assert (
        main(
            [
                "--corpus", str(path),
                "--output-dir", str(tmp_path / "out"),
                "--progress-every", "0",
            ]
        )
        == 1
    )
