"""Execution provenance for benchmark runs, captured once in the parent.

Phase 6A.5. The pre-6A.5 design let every worker discover the commit for
itself:

    worker -> subprocess git rev-parse -> intermittent failure under load
           -> exception swallowed -> null written into the shard

Track A2 recorded a commit in 97 of 308 shards and ``null`` in the other
211 for exactly that reason. The scientific values were unaffected -- the
same code produced every shard -- but the shards could no longer say so on
their own, and a provenance field that fails silently under load is worse
than none: it is indistinguishable from a run that genuinely had no commit.

So provenance is no longer *discovered*. It is captured once, in the
orchestrator, before any worker exists, validated there, frozen, and then
handed to every worker as data. A worker that never asks a question cannot
get an intermittent answer to it.

Two further things this module fixes.

**Tracked and untracked status are different facts.** The old single
boolean came from ``git status --porcelain``, which counts untracked files,
and this repository permanently carries two unrelated ``.docx`` files. Every
A1 and A2 shard therefore recorded ``working_tree_clean: false`` whether the
scientific source was modified or not, which is precisely when a
cleanliness flag stops being informative. They are now reported separately.

**A dirty tracked tree names its diff.** ``tracked_diff_sha256`` makes the
modification itself identifiable, so an audit can compare two runs' diffs
instead of reconstructing by hand what a bare ``false`` meant -- which is
what Phase 6A.4 had to do for the A2 prose edit.
"""

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Bumped when the recorded field set or the diff contract changes.
PROVENANCE_SCHEMA_VERSION = 1

#: Namespace for the tracked-diff digest, so the hash of a diff can never be
#: confused with the hash of a file that happens to hold the same bytes.
DIFF_SERIALIZATION_ID = "molfusion_tracked_diff_v1"

_TIMEOUT = 30


class ProvenanceError(RuntimeError):
    """Execution provenance could not be established.

    Raised in the parent, before workers exist. Deliberately fatal: an
    official benchmark run that cannot name its own commit must not start,
    because nothing downstream can tell such a run apart from one whose
    provenance merely failed to be written.
    """


@dataclass(frozen=True)
class ExecutionProvenance:
    """What produced a run. Immutable, captured once, passed to workers.

    ``git_commit`` is non-optional by construction -- there is no value of
    this type that does not name a commit.

    ``tracked_worktree_clean`` covers tracked files only, so it answers the
    question a reader actually has: did the scientific source differ from
    the named commit?

    ``tracked_diff_sha256`` is present exactly when the tracked tree is
    dirty, and identifies which modification was in effect.

    ``untracked_files_present`` is reported alongside rather than folded in.
    An untracked file is not a source modification; conflating the two is
    what made the old flag useless here.
    """

    git_commit: str
    tracked_worktree_clean: bool
    tracked_diff_sha256: str | None
    untracked_files_present: bool
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.git_commit:
            raise ProvenanceError("git_commit is required and must be non-empty")
        if self.tracked_worktree_clean and self.tracked_diff_sha256 is not None:
            raise ProvenanceError(
                "tracked_diff_sha256 must be None when the tracked tree is clean")
        if not self.tracked_worktree_clean and not self.tracked_diff_sha256:
            raise ProvenanceError(
                "tracked_diff_sha256 is required when the tracked tree is dirty")

    def as_dict(self) -> dict[str, Any]:
        """Deterministic mapping for shard and run-report serialization."""
        return dict(sorted(asdict(self).items()))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionProvenance":
        fields = {f: payload[f] for f in
                  ("git_commit", "tracked_worktree_clean",
                   "tracked_diff_sha256", "untracked_files_present")}
        return cls(**fields, schema_version=payload.get(
            "schema_version", PROVENANCE_SCHEMA_VERSION))


def _run_git(args: list[str], *, cwd: Path) -> str:
    """One git invocation, or ProvenanceError. Never returns a sentinel.

    The old helper's ``except: return None`` is the whole defect, so this
    one has no failure value to return -- every path either yields output or
    raises, and the raise happens in the parent where it stops the run.
    """
    try:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            timeout=_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProvenanceError(f"git {' '.join(args)} could not run: {exc}") from exc
    if completed.returncode != 0:
        raise ProvenanceError(
            f"git {' '.join(args)} failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()[:400]}")
    return completed.stdout


def tracked_diff_identity(diff_text: str) -> str:
    """Deterministic digest of a tracked diff.

    Normalizes line endings and trailing whitespace-only tails so the same
    modification hashes identically on Windows and POSIX, then namespaces
    the digest. Nothing volatile participates: git's own diff output carries
    no timestamps, and paths inside it are repository-relative (``a/`` and
    ``b/`` prefixed), so the identity does not depend on where the
    repository is checked out.
    """
    normalized = diff_text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    payload = f"{DIFF_SERIALIZATION_ID}\x1f{normalized}\n".encode()
    return hashlib.sha256(payload).hexdigest()


def default_repo_root() -> Path:
    """The repository root, derived from this file rather than the cwd.

    src/molfusion_backend/benchmark/provenance.py -> repository root.
    Deriving it from ``__file__`` keeps provenance correct when a runner is
    invoked from somewhere other than the repository root.
    """
    return Path(__file__).resolve().parents[4]


def capture(repo_root: Path) -> ExecutionProvenance:
    """Establish execution provenance. Parent process only.

    Call once, after configuration is final and before the worker pool
    exists. Raises ProvenanceError rather than degrading, so a run that
    cannot describe itself fails at startup instead of producing 6160 rows
    that no one can attribute.
    """
    root = Path(repo_root)
    commit = _run_git(["rev-parse", "HEAD"], cwd=root).strip()
    if not commit:
        raise ProvenanceError("git rev-parse HEAD returned no commit")

    # Tracked status: --untracked-files=no is what separates "the source
    # changed" from "there is a stray file in the directory".
    tracked_status = _run_git(
        ["status", "--porcelain", "--untracked-files=no"], cwd=root).strip()
    tracked_clean = tracked_status == ""

    untracked = _run_git(
        ["ls-files", "--others", "--exclude-standard"], cwd=root).strip()
    untracked_present = untracked != ""

    diff_sha = None
    if not tracked_clean:
        diff = _run_git(
            ["diff", "HEAD", "--no-color", "--no-ext-diff", "--ignore-submodules"],
            cwd=root)
        diff_sha = tracked_diff_identity(diff)

    return ExecutionProvenance(
        git_commit=commit,
        tracked_worktree_clean=tracked_clean,
        tracked_diff_sha256=diff_sha,
        untracked_files_present=untracked_present,
    )


def detect_source_mutation(
    startup: ExecutionProvenance, repo_root: Path
) -> dict[str, Any]:
    """Did tracked source change after the run started?

    Called once at the end of a run, never per shard. A run whose source
    moved underneath it is not a run at a single revision, and saying so is
    the point -- the alternative is a report that quietly claims an
    immutable execution state it did not have.
    """
    try:
        current = capture(repo_root)
    except ProvenanceError as exc:
        return {"checked": False, "violation": None, "error": str(exc)}

    violated = (
        current.git_commit != startup.git_commit
        or current.tracked_worktree_clean != startup.tracked_worktree_clean
        or current.tracked_diff_sha256 != startup.tracked_diff_sha256
    )
    return {
        "checked": True,
        "violation": violated,
        "startup": startup.as_dict(),
        "post_run": current.as_dict(),
    }


__all__ = [
    "DIFF_SERIALIZATION_ID",
    "PROVENANCE_SCHEMA_VERSION",
    "ExecutionProvenance",
    "ProvenanceError",
    "capture",
    "detect_source_mutation",
    "tracked_diff_identity",
]
