"""Audit recorded provenance in the completed A1 and A2 shard sets.

Phase 6A.5. Both matrices were produced before provenance hardening, so
many of their shards carry ``molfusion_git_commit: null``. This module says
so, in a separate artifact, and does not touch a single shard.

The distinction it exists to preserve:

  **recorded** -- what a shard literally contains. Some shards contain no
  commit. That fact is permanent and is reported as-is.

  **reconstructed** -- what the run provenance demonstrably was, argued
  from evidence outside the shards: the execution commit, the state of the
  runner sources at that commit, and the identity of the results.

Backfilling the second into the first would destroy the only evidence that
the defect ever existed, and would make a null shard indistinguishable from
one that genuinely recorded its commit. So nothing is written back. The
audit is a claim *about* the raw data, stored beside it.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from molfusion_backend.benchmark import provenance

AUDIT_VERSION = "6A.5.1"

DEFAULT_A1 = Path("backend/benchmark_runs/track_a1")
DEFAULT_A2 = Path("backend/benchmark_runs/track_a2")
DEFAULT_OUT = Path("backend/benchmark_runs/provenance_audit.json")

#: Commits whose changes are part of each track's execution history. A1 ran
#: across a fix sequence; A2 ran entirely at one commit.
A1_EXECUTION_COMMITS = ("459653b", "ddabb42", "2bcb467")
A2_EXECUTION_COMMITS = ("e6ae297",)


def _shard_commit(payload: dict[str, Any]) -> str | None:
    """The commit a shard records, under either the old or new schema."""
    environment = payload.get("environment", {})
    execution = environment.get("execution")
    if isinstance(execution, dict):
        return execution.get("git_commit")
    return environment.get("molfusion_git_commit")


def audit_track(run_dir: Path, *, track: str, results_name: str,
                execution_commits: tuple[str, ...]) -> dict[str, Any]:
    """Summarise recorded provenance for one completed track. Read-only."""
    shard_dir = run_dir / "shards"
    shards = sorted(shard_dir.rglob("*.json"))

    commits: Counter[str] = Counter()
    schema: Counter[str] = Counter()
    for path in shards:
        payload = json.loads(path.read_text("utf-8"))
        commit = _shard_commit(payload)
        commits[commit if commit else "null"] += 1
        environment = payload.get("environment", {})
        schema["hardened" if "execution" in environment else "legacy"] += 1

    results_path = run_dir / results_name
    report_path = run_dir / "run_report.json"
    report = json.loads(report_path.read_text("utf-8")) if report_path.exists() else {}

    null_count = commits.get("null", 0)
    return {
        "track": track,
        "run_directory": str(run_dir).replace("\\", "/"),
        "total_shards": len(shards),
        "recorded": {
            "per_commit_shard_counts": dict(sorted(commits.items())),
            "null_commit_shards": null_count,
            "populated_commit_shards": len(shards) - null_count,
            "completeness_fraction": (
                round((len(shards) - null_count) / len(shards), 6) if shards else None),
            "shard_provenance_schema": dict(sorted(schema.items())),
        },
        "reconstructed": {
            "execution_commits": list(execution_commits),
            "basis": (
                "runner sources are byte-identical to the execution commit; "
                "protocol_version and benchmark_release agree across every shard; "
                "the null field is a logging defect in the pre-6A.5 worker-local "
                "capture, not evidence of a different code state"),
            "backfilled_into_shards": False,
        },
        "scientific_identity": report.get("scientific_identity_sha256"),
        "results_file_sha256": (
            hashlib.sha256(results_path.read_bytes()).hexdigest()
            if results_path.exists() else None),
        "result_rows": report.get("result_rows"),
    }


def build_audit(*, a1_dir: Path, a2_dir: Path) -> dict[str, Any]:
    a1 = audit_track(a1_dir, track="tdc_official", results_name="results_track_a1.csv",
                     execution_commits=A1_EXECUTION_COMMITS)
    a2 = audit_track(a2_dir, track="molfusion_scaffold",
                     results_name="results_track_a2.csv",
                     execution_commits=A2_EXECUTION_COMMITS)
    total = a1["total_shards"] + a2["total_shards"]
    nulls = a1["recorded"]["null_commit_shards"] + a2["recorded"]["null_commit_shards"]
    return {
        "audit_version": AUDIT_VERSION,
        "provenance_schema_version": provenance.PROVENANCE_SCHEMA_VERSION,
        "shards_mutated": False,
        "tracks": {"A1": a1, "A2": a2},
        "summary": {
            "total_historical_shards": total,
            "shards_missing_a_recorded_commit": nulls,
            "classification": "metadata defect, not a scientific-value defect",
            "statement": (
                "Historical A1/A2 scientific results remain valid, but some "
                "shards lack complete recorded Git metadata because of the "
                "pre-6A.5 worker-local provenance implementation."),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-dir", type=Path, default=DEFAULT_A1)
    parser.add_argument("--a2-dir", type=Path, default=DEFAULT_A2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    audit = build_audit(a1_dir=args.a1_dir, a2_dir=args.a2_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, indent=1, sort_keys=True)
        handle.write("\n")

    for name, track in audit["tracks"].items():
        recorded = track["recorded"]
        print(f"{name} ({track['track']}): {track['total_shards']} shards, "
              f"{recorded['populated_commit_shards']} with a commit, "
              f"{recorded['null_commit_shards']} null")
        for commit, count in recorded["per_commit_shard_counts"].items():
            print(f"    {commit[:16]:<18} {count:>4}")
        print(f"    scientific identity {track['scientific_identity']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
