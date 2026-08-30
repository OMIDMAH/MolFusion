"""Build/run provenance shared by the corpus builder and its studies.

Every generated corpus artifact -- the reference corpus itself and any
study derived from it -- has to name the MolFusion revision that produced
it, so the two callers share one implementation rather than each growing
their own subtly different subprocess call.
"""

import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def git_commit(repo_marker: Path) -> str | None:
    """Current MolFusion commit, or None outside a working git checkout.

    Best-effort provenance: a build from an exported tarball is still a
    valid build, it just cannot name a commit. Cached because the answer
    cannot change within a process and spawning git per build measurably
    slows a test suite that builds many small corpora.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_marker,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


__all__ = ["git_commit"]
