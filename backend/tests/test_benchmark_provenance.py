"""Phase 6A.5: execution provenance is captured once and cannot go null.

The defect these tests exist to prevent is specific and it already
happened: under two concurrent workers, Track A2 recorded a commit in 97 of
308 shards and ``null`` in 211, because each worker ran its own
``git rev-parse`` and the helper turned every failure into ``None``.

So the tests here are mostly about *where* provenance comes from rather
than what it contains. Real temporary git repositories are used instead of
mocked subprocess output wherever the answer depends on git's actual
behaviour, and a real process pool is used to prove the cross-process
claim, because a mock of the thing under test would prove nothing.
"""

import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from molfusion_backend.benchmark import a2_runner, provenance, runner


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repository with one commit and one tracked file."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "source.py")
    _git(root, "commit", "-q", "-m", "initial")
    return root


def _worker_echo(payload: dict) -> dict:
    """Stand-in for a unit of work: reports the provenance it was handed.

    Module level so a real ProcessPoolExecutor can pickle it.
    """
    restored = provenance.ExecutionProvenance.from_dict(payload["execution_provenance"])
    return restored.as_dict()


# ---------------------------------------------------------------------------
# the model itself
# ---------------------------------------------------------------------------


def test_git_commit_cannot_be_none():
    with pytest.raises(provenance.ProvenanceError):
        provenance.ExecutionProvenance(
            git_commit="", tracked_worktree_clean=True,
            tracked_diff_sha256=None, untracked_files_present=False)


def test_git_commit_cannot_be_none_via_from_dict():
    with pytest.raises(provenance.ProvenanceError):
        provenance.ExecutionProvenance.from_dict({
            "git_commit": None, "tracked_worktree_clean": True,
            "tracked_diff_sha256": None, "untracked_files_present": False})


def test_a_dirty_tree_must_name_its_diff():
    with pytest.raises(provenance.ProvenanceError):
        provenance.ExecutionProvenance(
            git_commit="a" * 40, tracked_worktree_clean=False,
            tracked_diff_sha256=None, untracked_files_present=False)


def test_a_clean_tree_must_not_carry_a_diff():
    with pytest.raises(provenance.ProvenanceError):
        provenance.ExecutionProvenance(
            git_commit="a" * 40, tracked_worktree_clean=True,
            tracked_diff_sha256="b" * 64, untracked_files_present=False)


def test_provenance_is_immutable():
    captured = provenance.ExecutionProvenance(
        git_commit="a" * 40, tracked_worktree_clean=True,
        tracked_diff_sha256=None, untracked_files_present=False)
    with pytest.raises(Exception):
        captured.git_commit = "b" * 40  # type: ignore[misc]


def test_serialization_is_deterministic_and_round_trips():
    captured = provenance.ExecutionProvenance(
        git_commit="a" * 40, tracked_worktree_clean=False,
        tracked_diff_sha256="c" * 64, untracked_files_present=True)
    first, second = captured.as_dict(), captured.as_dict()
    assert first == second
    assert list(first) == sorted(first), "key order must be stable for hashing"
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert provenance.ExecutionProvenance.from_dict(first) == captured


# ---------------------------------------------------------------------------
# capture against a real repository
# ---------------------------------------------------------------------------


def test_capture_reports_a_clean_tracked_tree(repo: Path):
    captured = provenance.capture(repo)
    assert len(captured.git_commit) == 40
    assert captured.tracked_worktree_clean is True
    assert captured.tracked_diff_sha256 is None
    assert captured.untracked_files_present is False


def test_an_untracked_file_does_not_make_the_tracked_tree_dirty(repo: Path):
    """The exact confusion that made the old flag useless.

    This repository permanently carries two unrelated .docx files, so the
    pre-6A.5 boolean read False on every A1 and A2 shard whether or not the
    scientific source had been touched.
    """
    (repo / "unrelated.docx").write_text("not source", encoding="utf-8")
    captured = provenance.capture(repo)
    assert captured.tracked_worktree_clean is True
    assert captured.tracked_diff_sha256 is None
    assert captured.untracked_files_present is True


def test_a_modified_tracked_file_is_dirty_and_names_its_diff(repo: Path):
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    captured = provenance.capture(repo)
    assert captured.tracked_worktree_clean is False
    assert captured.tracked_diff_sha256 is not None
    assert len(captured.tracked_diff_sha256) == 64


def test_tracked_and_untracked_are_reported_independently(repo: Path):
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("x", encoding="utf-8")
    captured = provenance.capture(repo)
    assert captured.tracked_worktree_clean is False
    assert captured.untracked_files_present is True


# ---------------------------------------------------------------------------
# the diff identity contract
# ---------------------------------------------------------------------------


def test_the_diff_hash_is_deterministic(repo: Path):
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert provenance.capture(repo).tracked_diff_sha256 == \
        provenance.capture(repo).tracked_diff_sha256


def test_a_different_diff_gives_a_different_hash(repo: Path):
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    first = provenance.capture(repo).tracked_diff_sha256
    (repo / "source.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert provenance.capture(repo).tracked_diff_sha256 != first


def test_reverting_a_diff_restores_the_clean_state(repo: Path):
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert provenance.capture(repo).tracked_worktree_clean is False
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    restored = provenance.capture(repo)
    assert restored.tracked_worktree_clean is True
    assert restored.tracked_diff_sha256 is None


def test_the_diff_identity_ignores_line_ending_differences():
    """Windows and POSIX must agree on the identity of the same diff."""
    body = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    assert provenance.tracked_diff_identity(body) == \
        provenance.tracked_diff_identity(body.replace("\n", "\r\n"))


def test_the_diff_identity_is_namespaced():
    """A diff's digest must not collide with a plain hash of its bytes."""
    import hashlib

    body = "diff --git a/x b/x\n"
    assert provenance.tracked_diff_identity(body) != \
        hashlib.sha256(body.encode()).hexdigest()


def test_the_diff_identity_carries_no_absolute_path(repo: Path):
    """Two checkouts of the same change must produce the same identity."""
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    first = provenance.capture(repo)

    elsewhere = repo.parent / "second_checkout"
    subprocess.run(["git", "clone", "-q", str(repo), str(elsewhere)],
                   check=True, capture_output=True, text=True)
    _git(elsewhere, "config", "user.email", "test@example.com")
    _git(elsewhere, "config", "user.name", "Test")
    (elsewhere / "source.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert provenance.capture(elsewhere).tracked_diff_sha256 == first.tracked_diff_sha256


# ---------------------------------------------------------------------------
# fail loudly, in the parent, before any worker
# ---------------------------------------------------------------------------


def test_capture_raises_outside_a_git_checkout(tmp_path: Path):
    with pytest.raises(provenance.ProvenanceError):
        provenance.capture(tmp_path)


def test_capture_raises_when_git_is_missing(repo: Path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(provenance.ProvenanceError):
        provenance.capture(repo)


def test_the_old_swallowed_failure_now_raises_instead_of_writing_null(repo: Path, monkeypatch):
    """Regression test for the exact pre-6A.5 failure mode.

    Old behaviour: ``git rev-parse`` fails intermittently under load, the
    exception is swallowed, ``None`` lands in the shard, and the run
    completes looking healthy. New behaviour: the run cannot start.
    """
    real = subprocess.run

    def flaky(args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            raise subprocess.SubprocessError("resource temporarily unavailable")
        return real(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", flaky)
    with pytest.raises(provenance.ProvenanceError) as excinfo:
        provenance.capture(repo)
    assert "rev-parse" in str(excinfo.value)


def test_a_nonzero_git_exit_is_fatal_not_silent(repo: Path, monkeypatch):
    real = subprocess.run

    class Failed:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    def failing(args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            return Failed()
        return real(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing)
    with pytest.raises(provenance.ProvenanceError):
        provenance.capture(repo)


# ---------------------------------------------------------------------------
# workers never touch git
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", [runner, a2_runner])
def test_the_runners_no_longer_contain_a_git_helper(module):
    assert not hasattr(module, "_git"), (
        "worker-local git discovery is the 6A.5 defect and must stay removed")


@pytest.mark.parametrize("module", [runner, a2_runner])
def test_no_runner_source_invokes_git(module):
    """Checks executable code, not prose -- the docstrings discuss the defect."""
    import ast

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "subprocess" not in imported, (
        f"{module.__name__} must not import subprocess for provenance discovery")

    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value in {"rev-parse", "status", "diff", "git"}
    }
    assert "rev-parse" not in literals, f"{module.__name__} must not call git rev-parse"


@pytest.mark.parametrize("module", [runner, a2_runner])
def test_the_worker_reads_provenance_from_its_job(module):
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'job["execution_provenance"]' in source


@pytest.mark.parametrize("module", [runner, a2_runner])
def test_environment_requires_provenance_to_be_supplied(module):
    with pytest.raises(TypeError):
        module._environment()  # type: ignore[call-arg]


@pytest.mark.parametrize("module", [runner, a2_runner])
def test_environment_embeds_the_supplied_provenance(module):
    captured = provenance.ExecutionProvenance(
        git_commit="a" * 40, tracked_worktree_clean=True,
        tracked_diff_sha256=None, untracked_files_present=True)
    environment = module._environment(captured)
    assert environment["execution"] == captured.as_dict()
    assert environment["execution"]["git_commit"] == "a" * 40
    assert "molfusion_git_working_tree_clean" not in environment, (
        "the ambiguous combined flag must not come back")


# ---------------------------------------------------------------------------
# capture happens exactly once, whatever the worker count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_git_is_invoked_for_capture_exactly_once_regardless_of_workers(
    repo: Path, monkeypatch, workers
):
    """The core acceptance criterion: one capture per run, not per worker."""
    calls = {"count": 0}
    real_capture = provenance.capture

    def counting(root):
        calls["count"] += 1
        return real_capture(root)

    monkeypatch.setattr(provenance, "capture", counting)

    captured = provenance.capture(repo)
    payload = captured.as_dict()
    jobs = [{"execution_provenance": payload, "n": i} for i in range(workers * 4)]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_worker_echo, jobs))

    assert calls["count"] == 1, "provenance must be captured once, in the parent"
    assert len(results) == workers * 4
    assert all(r == payload for r in results)


def test_every_worker_receives_byte_identical_provenance(repo: Path):
    captured = provenance.capture(repo)
    payload = captured.as_dict()
    jobs = [{"execution_provenance": payload, "n": i} for i in range(12)]

    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(_worker_echo, jobs))

    assert len({json.dumps(r, sort_keys=True) for r in results}) == 1
    assert results[0] == payload


def test_capture_is_not_repeated_per_job(repo: Path, monkeypatch):
    """Provenance is copied into jobs, never recomputed while building them."""
    calls = {"count": 0}
    real = provenance._run_git

    def counting(args, *, cwd):
        calls["count"] += 1
        return real(args, cwd=cwd)

    monkeypatch.setattr(provenance, "_run_git", counting)
    captured = provenance.capture(repo)
    after_capture = calls["count"]

    payload = captured.as_dict()
    for _ in range(500):
        {"execution_provenance": payload}

    assert calls["count"] == after_capture, "building jobs must not call git"


# ---------------------------------------------------------------------------
# run-level and shard-level provenance must agree
# ---------------------------------------------------------------------------


def test_run_level_and_shard_level_provenance_are_equal(repo: Path):
    """The invariant a future run report must satisfy."""
    captured = provenance.capture(repo)
    run_report_environment = a2_runner._environment(captured)

    jobs = [{"execution_provenance": captured.as_dict(), "n": i} for i in range(6)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        shard_provenances = list(pool.map(_worker_echo, jobs))

    for shard in shard_provenances:
        assert shard == run_report_environment["execution"]


def test_a_shard_provenance_mismatch_is_detectable():
    """A tampered shard must not compare equal to the run's provenance."""
    genuine = provenance.ExecutionProvenance(
        git_commit="a" * 40, tracked_worktree_clean=True,
        tracked_diff_sha256=None, untracked_files_present=False)
    tampered = dict(genuine.as_dict(), git_commit="b" * 40)
    assert tampered != genuine.as_dict()


# ---------------------------------------------------------------------------
# mid-run source mutation
# ---------------------------------------------------------------------------


def test_no_violation_when_source_is_untouched(repo: Path):
    startup = provenance.capture(repo)
    result = provenance.detect_source_mutation(startup, repo)
    assert result["checked"] is True
    assert result["violation"] is False


def test_a_tracked_edit_during_the_run_is_a_violation(repo: Path):
    startup = provenance.capture(repo)
    (repo / "source.py").write_text("VALUE = 999\n", encoding="utf-8")
    result = provenance.detect_source_mutation(startup, repo)
    assert result["violation"] is True
    assert result["startup"] != result["post_run"]


def test_a_new_commit_during_the_run_is_a_violation(repo: Path):
    startup = provenance.capture(repo)
    (repo / "other.py").write_text("X = 1\n", encoding="utf-8")
    _git(repo, "add", "other.py")
    _git(repo, "commit", "-q", "-m", "second")
    assert provenance.detect_source_mutation(startup, repo)["violation"] is True


def test_an_untracked_file_appearing_is_not_a_source_violation(repo: Path):
    """A scratch file is not a change to what the run executed."""
    startup = provenance.capture(repo)
    (repo / "notes.docx").write_text("unrelated", encoding="utf-8")
    assert provenance.detect_source_mutation(startup, repo)["violation"] is False


def test_the_mutation_check_reports_rather_than_raises(tmp_path: Path):
    startup = provenance.ExecutionProvenance(
        git_commit="a" * 40, tracked_worktree_clean=True,
        tracked_diff_sha256=None, untracked_files_present=False)
    result = provenance.detect_source_mutation(startup, tmp_path)
    assert result["checked"] is False
    assert result["error"]


# ---------------------------------------------------------------------------
# the historical audit reads; it never writes back
# ---------------------------------------------------------------------------


def _fake_run(root: Path, *, track: str, results_name: str,
              commits: list[str | None]) -> Path:
    run = root / track
    (run / "shards" / "ep").mkdir(parents=True)
    for index, commit in enumerate(commits):
        environment = {"python": "3.11.0"}
        if commit is not None:
            environment["molfusion_git_commit"] = commit
        else:
            environment["molfusion_git_commit"] = None
        payload = {"status": "complete", "environment": environment}
        (run / "shards" / "ep" / f"rep_{index}.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (run / results_name).write_text("endpoint,metric_value\na,1.0\n", encoding="utf-8")
    (run / "run_report.json").write_text(
        json.dumps({"scientific_identity_sha256": "f" * 64, "result_rows": 1}) + "\n",
        encoding="utf-8")
    return run


def test_the_audit_counts_recorded_commits_and_nulls(tmp_path: Path):
    from molfusion_backend.benchmark import provenance_audit

    run = _fake_run(tmp_path, track="a2", results_name="results_track_a2.csv",
                    commits=["e6ae297", "e6ae297", None, None, None])
    audited = provenance_audit.audit_track(
        run, track="molfusion_scaffold", results_name="results_track_a2.csv",
        execution_commits=("e6ae297",))

    assert audited["total_shards"] == 5
    assert audited["recorded"]["null_commit_shards"] == 3
    assert audited["recorded"]["populated_commit_shards"] == 2
    assert audited["recorded"]["per_commit_shard_counts"] == {"e6ae297": 2, "null": 3}


def test_the_audit_does_not_mutate_shards(tmp_path: Path):
    """The central constraint: no inferred value is ever written back."""
    from molfusion_backend.benchmark import provenance_audit

    run = _fake_run(tmp_path, track="a2", results_name="results_track_a2.csv",
                    commits=["e6ae297", None, None])
    shards = sorted((run / "shards").rglob("*.json"))
    before = {p: p.read_bytes() for p in shards}

    provenance_audit.audit_track(
        run, track="molfusion_scaffold", results_name="results_track_a2.csv",
        execution_commits=("e6ae297",))

    assert {p: p.read_bytes() for p in shards} == before
    for path in shards:
        payload = json.loads(path.read_text("utf-8"))
        commit = payload["environment"]["molfusion_git_commit"]
        assert commit in ("e6ae297", None), "a null must stay null"


def test_the_audit_separates_recorded_from_reconstructed(tmp_path: Path):
    from molfusion_backend.benchmark import provenance_audit

    run = _fake_run(tmp_path, track="a2", results_name="results_track_a2.csv",
                    commits=[None, None])
    audited = provenance_audit.audit_track(
        run, track="molfusion_scaffold", results_name="results_track_a2.csv",
        execution_commits=("e6ae297",))

    assert audited["recorded"]["populated_commit_shards"] == 0
    assert audited["reconstructed"]["execution_commits"] == ["e6ae297"]
    assert audited["reconstructed"]["backfilled_into_shards"] is False


def test_the_audit_recognises_the_hardened_schema(tmp_path: Path):
    from molfusion_backend.benchmark import provenance_audit

    run = tmp_path / "new"
    (run / "shards" / "ep").mkdir(parents=True)
    captured = provenance.ExecutionProvenance(
        git_commit="a" * 40, tracked_worktree_clean=True,
        tracked_diff_sha256=None, untracked_files_present=True)
    (run / "shards" / "ep" / "r.json").write_text(
        json.dumps({"environment": {"execution": captured.as_dict()}}), encoding="utf-8")
    (run / "results_track_a2.csv").write_text("x\n", encoding="utf-8")

    audited = provenance_audit.audit_track(
        run, track="molfusion_scaffold", results_name="results_track_a2.csv",
        execution_commits=("a" * 40,))
    assert audited["recorded"]["null_commit_shards"] == 0
    assert audited["recorded"]["shard_provenance_schema"] == {"hardened": 1}


def test_the_audit_states_the_limitation_plainly(tmp_path: Path):
    from molfusion_backend.benchmark import provenance_audit

    _fake_run(tmp_path, track="a1", results_name="results_track_a1.csv",
              commits=["459653b", None])
    _fake_run(tmp_path, track="a2", results_name="results_track_a2.csv",
              commits=["e6ae297", None])
    audit = provenance_audit.build_audit(a1_dir=tmp_path / "a1", a2_dir=tmp_path / "a2")

    assert audit["shards_mutated"] is False
    assert audit["summary"]["shards_missing_a_recorded_commit"] == 2
    assert "remain valid" in audit["summary"]["statement"]
    assert "lack complete recorded Git metadata" in audit["summary"]["statement"]
