"""Track A1 execution: orchestration, checkpointing, resume, and QC.

The run is roughly a day of single-core work, so the design assumption is
that it *will* be interrupted -- by a crash, a reboot, or a person. Every
guarantee here follows from that:

  * work is checkpointed at one shard per ``endpoint x representation x
    probe``, written atomically, so an interruption costs one cell;
  * a shard is reused only after it validates against the current release
    and the expected seed set, never because a file with the right name
    exists;
  * parallelism is across cells, never inside the scientific logic, so the
    worker count cannot change a single number.

Nothing here imports PyTDC.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy
import rdkit
import sklearn

from molfusion_backend.benchmark import a1, feature_store, protocol, release

SHARD_SCHEMA_VERSION = 1

DEFAULT_FROZEN = Path("backend/benchmark_data/frozen")
DEFAULT_MANIFEST = Path("backend/benchmark_manifests/tdc_admet_group.json")
DEFAULT_OUTPUT = Path("backend/benchmark_runs/track_a1")
DEFAULT_CACHE = Path("backend/benchmark_cache/features")

#: Long-format result columns. Written in this order, always.
RESULT_COLUMNS = (
    "benchmark_release",
    "track",
    "endpoint",
    "task_type",
    "split_id",
    "split_strategy",
    "seed",
    "representation",
    "agent_version",
    "model_family",
    "probe",
    "hyperparameters",
    "metric",
    "metric_value",
    "tdc_official_metric",
    "molfusion_primary_metric",
    "n_train",
    "n_valid",
    "n_test",
    "feature_dim",
    "feature_failures",
    "feature_seconds",
    "feature_cache_hit",
    "fit_seconds",
    "selection_seconds",
    "predict_seconds",
    "validation_predict_seconds",
)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# the experiment matrix
# ---------------------------------------------------------------------------


def experiment_matrix(manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Every ``(endpoint, representation, probe)`` cell Track A1 must produce.

    Derived from the frozen protocol and the frozen manifest, never from
    what happens to be on disk -- so a missing cell is detectable rather
    than invisible.
    """
    endpoints = sorted(n for n, e in manifest["endpoints"].items() if e["included"])
    return [
        (endpoint, representation, probe)
        for endpoint in endpoints
        for representation in protocol.TRACK_A_REPRESENTATIONS
        for probe in protocol.PROBES
    ]


def expected_counts(manifest: dict[str, Any]) -> dict[str, int]:
    """What a complete Track A1 run must contain, computed up front."""
    cells = experiment_matrix(manifest)
    endpoints = len({c[0] for c in cells})
    seeds = len(protocol.TRACK_A1_SEEDS)
    candidates = len(pipelines_grid_size())
    metrics_per = {
        protocol.TASK_CLASSIFICATION: 4,   # auroc, auprc, balanced_accuracy, mcc
        protocol.TASK_REGRESSION: 4,       # mae, rmse, r2, spearman
    }
    rows = 0
    for endpoint, _representation, _probe in cells:
        rows += seeds * metrics_per[manifest["endpoints"][endpoint]["task_type"]]
    return {
        "endpoints": endpoints,
        "representations": len(protocol.TRACK_A_REPRESENTATIONS),
        "probes": len(protocol.PROBES),
        "seeds": seeds,
        "hyperparameter_candidates": candidates,
        "cells": len(cells),
        "selection_fits": len(cells) * seeds * candidates,
        "final_fits": len(cells) * seeds,
        "total_fits": len(cells) * seeds * (candidates + 1),
        "test_evaluations": len(cells) * seeds,
        "result_rows": rows,
    }


def pipelines_grid_size() -> list[Any]:
    from molfusion_backend.benchmark import pipelines

    return pipelines.hyperparameter_grid(
        protocol.PROBE_LINEAR, protocol.TASK_CLASSIFICATION
    )


def cell_identity(
    *, release_identity: str, endpoint: str, representation: str, probe: str, seeds: Sequence[int]
) -> str:
    """A deterministic scientific identity for one completed cell.

    Covers the configuration that determines the numbers and nothing else.
    Timestamps, machine paths, worker counts and wall-clock durations are
    excluded on purpose: rerunning the same cell on another machine next
    year must produce the same identity, or the identity is recording where
    the work happened rather than what was computed.
    """
    payload = "\x1f".join(
        (
            str(SHARD_SCHEMA_VERSION),
            release_identity,
            protocol.PROTOCOL_VERSION,
            a1.TRACK,
            endpoint,
            representation,
            probe,
            ",".join(str(s) for s in sorted(seeds)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------


def shard_path(output_dir: Path, endpoint: str, representation: str, probe: str) -> Path:
    return output_dir / "shards" / endpoint / f"{representation}__{probe}.json"


def write_shard(path: Path, payload: dict[str, Any]) -> None:
    """Write one shard atomically: a partial shard must never look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=1, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_valid_shard(
    path: Path, *, release_identity: str, expected_seeds: Sequence[int], expected_rows: int
) -> dict[str, Any] | None:
    """Return a shard only if it is complete and current, else None.

    Every rejection path here is a cell that gets recomputed. That is the
    cheap direction to be wrong in: recomputing a good cell costs minutes,
    while trusting a truncated or stale one silently corrupts the result
    table.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("shard_schema_version") != SHARD_SCHEMA_VERSION:
        return None
    if payload.get("benchmark_release") != release_identity:
        return None
    if payload.get("protocol_version") != protocol.PROTOCOL_VERSION:
        return None
    if payload.get("status") != "complete":
        return None
    if sorted(payload.get("seeds", [])) != sorted(expected_seeds):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        return None
    identity = cell_identity(
        release_identity=release_identity,
        endpoint=payload["cell"]["endpoint"],
        representation=payload["cell"]["representation"],
        probe=payload["cell"]["probe"],
        seeds=payload["seeds"],
    )
    if payload.get("cell_identity") != identity:
        return None
    return payload


# ---------------------------------------------------------------------------
# one work unit: endpoint x representation, both probes
# ---------------------------------------------------------------------------


def run_unit(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one ``endpoint x representation`` unit and write its shards.

    Both probes share the unit so the feature matrix is computed or loaded
    once rather than twice. Runs in a worker process; everything it needs
    arrives as plain data.
    """
    frozen_dir = Path(job["frozen_dir"])
    output_dir = Path(job["output_dir"])
    manifest = json.loads(Path(job["manifest_path"]).read_text("utf-8"))
    release_identity = manifest["release_identity_sha256"]
    endpoint_name, representation = job["endpoint"], job["representation"]

    started = time.perf_counter()
    endpoint = a1.load_official_endpoint(
        endpoint_name, frozen_dir=frozen_dir, manifest=manifest
    )
    guards = a1.verify_leakage_guards(endpoint, manifest=manifest)
    splits_by_seed = a1.official_splits(endpoint, frozen_dir=frozen_dir)

    store = feature_store.FeatureStore(Path(job["cache_dir"]))
    features = a1.features_for(
        endpoint,
        representation,
        store=store,
        release_identity=release_identity,
        artifact_identity=a1.artifact_identity_for(representation),
    )

    expected_failures = job["expected_feature_failures"]
    if len(features.failures) != expected_failures:
        return {
            "endpoint": endpoint_name,
            "representation": representation,
            "status": "failed",
            "error": (
                f"unexpected representation failures: {len(features.failures)} "
                f"(frozen audit recorded {expected_failures}); refusing to compare "
                "models trained on a different molecule set"
            ),
            "seconds": time.perf_counter() - started,
        }

    written = []
    for probe in protocol.PROBES:
        result = a1.run_cell(
            endpoint=endpoint,
            representation=representation,
            probe=probe,
            splits_by_seed=splits_by_seed,
            features=features,
            release_identity=release_identity,
        )
        payload = {
            "shard_schema_version": SHARD_SCHEMA_VERSION,
            "status": "complete",
            "benchmark_release": release_identity,
            "release_name": manifest["release_name"],
            "protocol_version": protocol.PROTOCOL_VERSION,
            "cell_identity": cell_identity(
                release_identity=release_identity,
                endpoint=endpoint_name,
                representation=representation,
                probe=probe,
                seeds=result["seeds"],
            ),
            "leakage_guards": guards,
            "test_set_sha256": guards["test_set_sha256"],
            "chembl37_exposure": manifest["endpoints"][endpoint_name].get("chembl37_exposure"),
            "environment": _environment(),
            **result,
        }
        write_shard(shard_path(output_dir, endpoint_name, representation, probe), payload)
        written.append(probe)

    return {
        "endpoint": endpoint_name,
        "representation": representation,
        "status": "ok",
        "probes": written,
        "cache_hit": features.cache_hit,
        "feature_seconds": features.seconds,
        "feature_failures": len(features.failures),
        "dimension": features.dimension,
        "seconds": time.perf_counter() - started,
    }


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "molfusion_git_commit": _git("rev-parse", "HEAD"),
        "molfusion_git_working_tree_clean": _git("status", "--porcelain") == "",
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def plan(
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    endpoints: Sequence[str] | None = None,
    representations: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """Split the matrix into work still to do and cells already validated."""
    release_identity = manifest["release_identity_sha256"]
    seeds = list(protocol.TRACK_A1_SEEDS)
    cells = experiment_matrix(manifest)
    if endpoints:
        cells = [c for c in cells if c[0] in set(endpoints)]
    if representations:
        cells = [c for c in cells if c[1] in set(representations)]

    metrics_per = {protocol.TASK_CLASSIFICATION: 4, protocol.TASK_REGRESSION: 4}
    done: list[tuple[str, str, str]] = []
    pending_units: dict[tuple[str, str], None] = {}
    for endpoint, representation, probe in cells:
        expected_rows = len(seeds) * metrics_per[manifest["endpoints"][endpoint]["task_type"]]
        shard = read_valid_shard(
            shard_path(output_dir, endpoint, representation, probe),
            release_identity=release_identity,
            expected_seeds=seeds,
            expected_rows=expected_rows,
        )
        if shard is not None:
            done.append((endpoint, representation, probe))
        else:
            pending_units[(endpoint, representation)] = None

    jobs = [
        {
            "endpoint": endpoint,
            "representation": representation,
            "expected_feature_failures": _frozen_failures(manifest, endpoint, representation),
        }
        for endpoint, representation in pending_units
    ]
    return jobs, done


def _frozen_failures(manifest: dict[str, Any], endpoint: str, representation: str) -> int:
    availability = manifest["endpoints"][endpoint].get("representation_availability") or {}
    entry = availability.get(representation)
    if entry is None:
        return 0
    # The frozen audit counted failures over the *cleaned* universe; Track A1
    # runs over the official rows, whose only invalid molecules are the ones
    # the manifest already records as rdkit_invalid.
    return entry.get("failures", 0) + manifest["endpoints"][endpoint]["ingestion"]["rdkit_invalid"]


def execute(
    *,
    frozen_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
    endpoints: Sequence[str] | None = None,
    representations: Sequence[str] | None = None,
    expected_release_identity: str | None = None,
) -> dict[str, Any]:
    """Run Track A1 to completion, resuming whatever is already valid."""
    manifest = json.loads(manifest_path.read_text("utf-8"))
    release_identity = manifest["release_identity_sha256"]
    if expected_release_identity and release_identity != expected_release_identity:
        raise a1.TrackA1Error(
            f"release identity {release_identity} != expected "
            f"{expected_release_identity}; aborting rather than benchmarking "
            "against unverified data"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs, done = plan(
        manifest, output_dir, endpoints=endpoints, representations=representations
    )
    for job in jobs:
        job.update(
            {
                "frozen_dir": str(frozen_dir),
                "manifest_path": str(manifest_path),
                "output_dir": str(output_dir),
                "cache_dir": str(cache_dir),
            }
        )

    print(
        f"Track A1: {len(done)} cells already valid, {len(jobs)} units to run "
        f"on {workers} worker(s)",
        flush=True,
    )

    started = time.perf_counter()
    outcomes: list[dict[str, Any]] = []
    if workers <= 1:
        for job in jobs:
            outcomes.append(run_unit(job))
            _report(outcomes[-1], len(outcomes), len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_unit, job): job for job in jobs}
            for finished, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                try:
                    outcomes.append(future.result())
                except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                    outcomes.append(
                        {
                            "endpoint": job["endpoint"],
                            "representation": job["representation"],
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                _report(outcomes[-1], finished, len(jobs))

    wall_seconds = time.perf_counter() - started
    failed = [o for o in outcomes if o["status"] != "ok"]
    return {
        "release_identity": release_identity,
        "units_run": len(outcomes),
        "units_failed": len(failed),
        "failures": failed,
        "cells_already_valid": len(done),
        "workers": workers,
        "wall_seconds": wall_seconds,
        "feature_seconds": sum(o.get("feature_seconds", 0.0) for o in outcomes),
        "cache_hits": sum(1 for o in outcomes if o.get("cache_hit")),
    }


def _report(outcome: dict[str, Any], index: int, total: int) -> None:
    tag = "ok " if outcome["status"] == "ok" else "FAIL"
    extra = ""
    if outcome["status"] == "ok":
        extra = (
            f" feat={outcome['feature_seconds']:6.1f}s"
            f" {'cached' if outcome['cache_hit'] else 'cold  '}"
            f" d={outcome['dimension']:>5}"
        )
    else:
        extra = f" {outcome.get('error', '')[:120]}"
    print(
        f"[{index:>4}/{total}] {tag} {outcome['endpoint']:<32}"
        f"{outcome['representation']:<28}{outcome.get('seconds', 0):7.1f}s{extra}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# collection and QC
# ---------------------------------------------------------------------------


def collect(output_dir: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gather every valid shard into long-format rows, and audit the matrix."""
    release_identity = manifest["release_identity_sha256"]
    seeds = list(protocol.TRACK_A1_SEEDS)
    metrics_per = {protocol.TASK_CLASSIFICATION: 4, protocol.TASK_REGRESSION: 4}

    rows: list[dict[str, Any]] = []
    present: set[tuple[str, str, str]] = set()
    missing: list[tuple[str, str, str]] = []
    identities: dict[str, str] = {}
    test_identities: dict[str, set[str]] = {}
    feature_failures: dict[str, int] = {}
    timings: list[dict[str, Any]] = []

    for endpoint, representation, probe in experiment_matrix(manifest):
        expected_rows = len(seeds) * metrics_per[manifest["endpoints"][endpoint]["task_type"]]
        shard = read_valid_shard(
            shard_path(output_dir, endpoint, representation, probe),
            release_identity=release_identity,
            expected_seeds=seeds,
            expected_rows=expected_rows,
        )
        if shard is None:
            missing.append((endpoint, representation, probe))
            continue
        present.add((endpoint, representation, probe))
        rows.extend(shard["rows"])
        identities[f"{endpoint}/{representation}/{probe}"] = shard["cell_identity"]
        test_identities.setdefault(endpoint, set()).add(shard["test_set_sha256"])
        feature_failures[f"{endpoint}/{representation}"] = shard["feature_failures"]
        for timing in shard["timings"]:
            timings.append({**timing, "endpoint": endpoint,
                            "representation": representation, "probe": probe})

    expected = expected_counts(manifest)
    seen_keys = [
        (r["endpoint"], r["representation"], r["probe"], r["seed"], r["metric"]) for r in rows
    ]
    duplicates = len(seen_keys) - len(set(seen_keys))
    bad_values = [
        r for r in rows
        if r["metric_value"] is None
        or (isinstance(r["metric_value"], float) and (r["metric_value"] != r["metric_value"]
                                                      or abs(r["metric_value"]) == float("inf")))
    ]
    unstable_test = {e: sorted(v) for e, v in test_identities.items() if len(v) != 1}

    audit = {
        "expected": expected,
        "observed": {
            "cells": len(present),
            "result_rows": len(rows),
            "endpoints": len({c[0] for c in present}),
            "representations": len({c[1] for c in present}),
            "probes": len({c[2] for c in present}),
            "seeds": len({r["seed"] for r in rows}),
        },
        "missing_cells": [list(c) for c in missing],
        "duplicate_rows": duplicates,
        "nan_or_inf_rows": [
            {"endpoint": r["endpoint"], "representation": r["representation"],
             "probe": r["probe"], "seed": r["seed"], "metric": r["metric"],
             "value": r["metric_value"]}
            for r in bad_values
        ],
        "test_identity_stable_within_endpoint": not unstable_test,
        "endpoints_with_unstable_test_identity": unstable_test,
        "feature_failures": feature_failures,
        "total_feature_failures": sum(feature_failures.values()),
        "cell_identities": identities,
        "complete": (
            not missing
            and duplicates == 0
            and len(rows) == expected["result_rows"]
            and not unstable_test
        ),
    }
    return rows, {"audit": audit, "timings": timings}


#: The columns that determine the science. Everything omitted here -- every
#: duration, the cache-hit flag -- is a property of the machine and the run
#: order, not of the result, and including it would make two identical
#: benchmarks disagree.
SCIENTIFIC_COLUMNS = (
    "benchmark_release",
    "track",
    "endpoint",
    "task_type",
    "split_id",
    "seed",
    "representation",
    "agent_version",
    "model_family",
    "probe",
    "hyperparameters",
    "metric",
    "metric_value",
    "n_train",
    "n_valid",
    "n_test",
    "feature_dim",
    "feature_failures",
)


def scientific_identity(rows: Sequence[dict[str, Any]]) -> str:
    """A digest of the science only, stable across machines and reruns.

    The results CSV digest is not this: it covers fit_seconds and friends,
    so it changes every run even when every number that matters is
    identical. Metric values are formatted with repr() -- the shortest
    round-trip form -- so the digest is exact rather than rounded.
    """
    ordered = sorted(
        rows,
        key=lambda r: (r["endpoint"], r["representation"], r["probe"], r["seed"], r["metric"]),
    )
    digest = hashlib.sha256()
    digest.update(f"scientific_identity_v1rows={len(ordered)}".encode())
    for row in ordered:
        fields = []
        for column in SCIENTIFIC_COLUMNS:
            value = row.get(column)
            fields.append(repr(value) if isinstance(value, float) else str(value))
        digest.update(("" + "".join(fields)).encode("utf-8"))
    return digest.hexdigest()


def write_results(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """Write long-format results and return the file's SHA-256.

    Rows are sorted into a canonical order first, so the digest identifies
    the science rather than the order shards happened to finish in.
    """
    ordered = sorted(
        rows,
        key=lambda r: (
            r["endpoint"], r["representation"], r["probe"], r["seed"], r["metric"]
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(RESULT_COLUMNS), lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)
    return release.sha256_file(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the Track A1 benchmark.")
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--endpoints", nargs="*", default=None)
    parser.add_argument("--representations", nargs="*", default=None)
    parser.add_argument("--expect-release", default=None)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text("utf-8"))

    summary: dict[str, Any] = {}
    if not args.collect_only:
        summary = execute(
            frozen_dir=args.frozen_dir,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            workers=args.workers,
            endpoints=args.endpoints,
            representations=args.representations,
            expected_release_identity=args.expect_release,
        )

    rows, collected = collect(args.output_dir, manifest)
    audit = collected["audit"]
    digest = write_results(args.output_dir / "results_track_a1.csv", rows)

    report = {
        "run_summary": summary,
        "audit": audit,
        "result_rows": len(rows),
        "results_file_sha256": digest,
        "scientific_identity_sha256": scientific_identity(rows),
        "scientific_identity_columns": list(SCIENTIFIC_COLUMNS),
        "feature_cache_contract": feature_store.cache_contract(),
        "environment": _environment(),
    }
    with open(args.output_dir / "run_report.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    with open(args.output_dir / "timings.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(collected["timings"], handle, indent=1, sort_keys=True)
        handle.write("\n")

    print(f"\ncells      {audit['observed']['cells']}/{audit['expected']['cells']}")
    print(f"rows       {len(rows)}/{audit['expected']['result_rows']}")
    print(f"missing    {len(audit['missing_cells'])}")
    print(f"duplicates {audit['duplicate_rows']}")
    print(f"nan/inf    {len(audit['nan_or_inf_rows'])}")
    print(f"complete   {audit['complete']}")
    print(f"file sha   {digest}")
    print(f"science    {scientific_identity(rows)}")
    return 0 if audit["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
