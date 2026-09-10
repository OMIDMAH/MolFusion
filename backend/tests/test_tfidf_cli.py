import json

import pytest

from molfusion_backend.corpus.serialization import write_corpus
from molfusion_backend.tfidf import contract
from molfusion_backend.tfidf.builder import FROZEN_DOCUMENT_COUNT, FROZEN_FIT_CORPUS_SHA256
from molfusion_backend.tfidf.cli import build_parser, main

FIXTURE_SMILES = sorted(
    {"CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "N", "O"}
    | {f"C{'C' * index}O" for index in range(1, 150)}
)


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "canonical_smiles.smi"
    sha256, _ = write_corpus(path, FIXTURE_SMILES)
    return path, sha256


def build_argv(corpus, root, **overrides):
    path, sha256 = corpus
    argv = [
        "--artifact-root", str(root),
        "build",
        "--corpus", str(path),
        "--expected-sha256", overrides.get("sha256", sha256),
        "--expected-documents", str(len(FIXTURE_SMILES)),
        "--progress-every", "0",
    ]
    return argv


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_defaults_are_the_frozen_identity_and_corpus(tmp_path):
    args = build_parser().parse_args(
        ["build", "--corpus", str(tmp_path / "c.smi")]
    )
    assert args.artifact_type == contract.ARTIFACT_TYPE
    assert args.artifact_id == contract.ARTIFACT_ID
    assert args.artifact_version == contract.ARTIFACT_VERSION
    assert args.expected_sha256 == FROZEN_FIT_CORPUS_SHA256
    assert args.expected_documents == FROZEN_DOCUMENT_COUNT


def test_there_is_no_force_flag():
    """Immutability is enforced by the absence of an escape hatch, not
    by discipline; verify-rebuild is the supported alternative."""
    help_text = build_parser().format_help()
    assert "--force" not in help_text
    assert "verify-rebuild" in help_text


def test_build_then_audit(corpus, tmp_path, capsys):
    root = tmp_path / "artifacts"
    assert main(build_argv(corpus, root)) == 0
    capsys.readouterr()

    exit_code = main([
        "--artifact-root", str(root),
        "audit",
        "--expect-fit-corpus-sha256", corpus[1],
    ])
    assert exit_code == 0
    assert "checksums verified" in capsys.readouterr().err


def test_a_digest_mismatch_exits_non_zero_without_writing(corpus, tmp_path, capsys):
    root = tmp_path / "artifacts"
    argv = build_argv(corpus, root)
    argv[argv.index("--expected-sha256") + 1] = "0" * 64
    assert main(argv) == 1
    assert "identity mismatch" in capsys.readouterr().err
    assert not root.exists()


def test_building_twice_exits_non_zero(corpus, tmp_path, capsys):
    root = tmp_path / "artifacts"
    assert main(build_argv(corpus, root)) == 0
    capsys.readouterr()
    assert main(build_argv(corpus, root)) == 1
    assert "immutable" in capsys.readouterr().err


def test_verify_rebuild_reports_identical_payloads(corpus, tmp_path, capsys):
    root = tmp_path / "artifacts"
    assert main(build_argv(corpus, root)) == 0
    capsys.readouterr()

    path, sha256 = corpus
    exit_code = main([
        "--artifact-root", str(root),
        "verify-rebuild",
        "--corpus", str(path),
        "--expected-sha256", sha256,
        "--expected-documents", str(len(FIXTURE_SMILES)),
        "--scratch-root", str(tmp_path / "scratch"),
        "--progress-every", "0",
    ])
    captured = capsys.readouterr().err
    assert exit_code == 0
    assert "all scientific payloads identical: True" in captured
    assert "IDENTICAL" in captured
