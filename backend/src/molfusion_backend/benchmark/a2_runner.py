"""Track A2 execution: orchestration, checkpointing, resume, and QC.

Same discipline as the A1 runner -- atomic shards, validate-before-reuse,
parallelism across cells only -- against a separate output directory so A1's
frozen artifacts are never touched.

A2 additionally records, per endpoint, what cleaning removed and evidence
that the five scaffold partitions really are distinct. Without the latter
A2 would be A1 with extra steps, so it is checked rather than assumed.

Nothing here imports PyTDC, and nothing here imports the A1 modules.
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

from molfusion_backend.benchmark import a2, feature_store, protocol, release

SHARD_SCHEMA_VERSION = 1

DEFAULT_FROZEN = Path("backend/benchmark_data/frozen")
DEFAULT_MANIFEST = Path("backend/benchmark_manifests/tdc_admet_group.json")
DEFAULT_OUTPUT = Path("backend/benchmark_runs/track_a2")
DEFAULT_CACHE = Path("backend/benchmark_cache/features")

RESULT_COLUMNS = (
    "benchmark_release", "track", "endpoint", "task_type", "split_id",
    "split_strategy", "seed", "representation", "agent_version", "model_family",
    "probe", "hyperparameters", "metric", "metric_value", "tdc_official_metric",
    "molfusion_primary_metric", "n_train", "n_valid", "n_test", "feature_dim",
    "feature_failures", "feature_seconds", "feature_cache_hit", "fit_seconds",
    "selection_seconds", "predict_seconds", "validation_predict_seconds",
)

SCIENTIFIC_COLUMNS = (
    "benchmark_release", "track", "endpoint", "task_type", "split_id", "seed",
    "representation", "agent_version", "model_family", "probe", "hyperparameters",
    "metric", "metric_value", "n_train", "n_valid", "n_test", "feature_dim",
    "feature_failures",
)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "molfusion_git_commit": _git("rev-parse", "HEAD"),
        # --porcelain counts untracked files; the repository permanently
        # carries two unrelated .docx files, so this reads False even with a
        # clean tracked tree. Recorded as-is rather than quietly filtered.
        "molfusion_git_working_tree_clean": _git("status", "--porcelain") == "",
    }


# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------


def experiment_matrix(manifest: dict[str, Any]) -> list[tuple[str, str, str]]:
    endpoints = sorted(n for n, e in manifest["endpoints"].items() if e["included"])
    return [
        (endpoint, representation, probe)
        for endpoint in endpoints
        for representation in protocol.TRACK_A_REPRESENTATIONS
        for probe in protocol.PROBES
    ]


def expected_counts(manifest: dict[str, Any]) -> dict[str, int]:
    from molfusion_backend.benchmark import pipelines

    cells = experiment_matrix(manifest)
    seeds = len(a2.SEEDS)
    candidates = len(pipelines.hyperparameter_grid(
        protocol.PROBE_LINEAR, protocol.TASK_CLASSIFICATION))
    return {
        "endpoints": len({c[0] for c in cells}),
        "representations": len(protocol.TRACK_A_REPRESENTATIONS),
        "probes": len(protocol.PROBES),
        "seeds": seeds,
        "hyperparameter_candidates": candidates,
        "cells": len(cells),
        "selection_fits": len(cells) * seeds * candidates,
        "final_fits": len(cells) * seeds,
        "total_fits": len(cells) * seeds * (candidates + 1),
        "test_evaluations": len(cells) * seeds,
        "result_rows": len(cells) * seeds * 4,
    }


def cell_identity(*, release_identity: str, endpoint: str, representation: str,
                  probe: str, seeds: Sequence[int]) -> str:
    payload = "\x1f".join((
        str(SHARD_SCHEMA_VERSION), release_identity, protocol.PROTOCOL_VERSION,
        a2.TRACK, endpoint, representation, probe,
        ",".join(str(s) for s in sorted(seeds)),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scientific_identity(rows: Sequence[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: (
        r["endpoint"], r["representation"], r["probe"], r["seed"], r["metric"]))
    digest = hashlib.sha256()
    digest.update(f"scientific_identity_v1\x1frows={len(ordered)}".encode())
    for row in ordered:
        fields = [repr(row.get(c)) if isinstance(row.get(c), float) else str(row.get(c))
                  for c in SCIENTIFIC_COLUMNS]
        digest.update(("\x1e" + "\x1f".join(fields)).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------


def shard_path(output_dir: Path, endpoint: str, representation: str, probe: str) -> Path:
    return output_dir / "shards" / endpoint / f"{representation}__{probe}.json"


def write_shard(path: Path, payload: dict[str, Any]) -> None:
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


def read_valid_shard(path: Path, *, release_identity: str,
                     expected_seeds: Sequence[int], expected_rows: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("shard_schema_version") != SHARD_SCHEMA_VERSION:
        return None
    if payload.get("track") != a2.TRACK:
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
        release_identity=release_identity, endpoint=payload["cell"]["endpoint"],
        representation=payload["cell"]["representation"],
        probe=payload["cell"]["probe"], seeds=payload["seeds"])
    if payload.get("cell_identity") != identity:
        return None
    return payload


# ---------------------------------------------------------------------------
# work unit
# ---------------------------------------------------------------------------


def run_unit(job: dict[str, Any]) -> dict[str, Any]:
    """One endpoint x representation unit: both probes, five partitions."""
    frozen_dir = Path(job["frozen_dir"])
    output_dir = Path(job["output_dir"])
    manifest = json.loads(Path(job["manifest_path"]).read_text("utf-8"))
    release_identity = manifest["release_identity_sha256"]
    endpoint_name, representation = job["endpoint"], job["representation"]

    started = time.perf_counter()
    endpoint = a2.load_cleaned_endpoint(endpoint_name, frozen_dir=frozen_dir, manifest=manifest)
    splits_by_seed = a2.build_splits(endpoint)
    distinctness = a2.split_distinctness(splits_by_seed)
    # Non-distinct test sets are recorded, not fatal. The frozen splitter
    # orders scaffold groups largest-first and only permutes within an
    # equal-size tier, so on endpoints whose multi-member groups already
    # cover the train+validation target the seed barely moves the test set.
    # That limits what such an endpoint can say about repartitioning -- but
    # dropping it would shrink A2's endpoint set relative to A1 and weaken
    # the very comparison A2 exists to make. It is flagged per endpoint and
    # excluded from the "genuinely repartitioned" subset at analysis time.
    # Actual partition OVERLAP remains fatal; that is a defect, not a
    # property, and a2.build_splits still raises on it.

    store = feature_store.FeatureStore(Path(job["cache_dir"]))
    features = a2.features_for(endpoint, representation, store=store,
                               release_identity=release_identity)

    written = []
    for probe in protocol.PROBES:
        result = a2.run_cell(
            endpoint=endpoint, representation=representation, probe=probe,
            splits_by_seed=splits_by_seed, features=features,
            release_identity=release_identity)
        payload = {
            "shard_schema_version": SHARD_SCHEMA_VERSION,
            "status": "complete",
            "track": a2.TRACK,
            "benchmark_release": release_identity,
            "release_name": manifest["release_name"],
            "protocol_version": protocol.PROTOCOL_VERSION,
            "cell_identity": cell_identity(
                release_identity=release_identity, endpoint=endpoint_name,
                representation=representation, probe=probe, seeds=result["seeds"]),
            "cleaning": {
                "raw_rows": endpoint.raw_rows,
                "cleaned_molecules": endpoint.size,
                **endpoint.ingestion,
            },
            "split_audits": {str(s): v.audit for s, v in splits_by_seed.items()},
            "split_distinctness": distinctness,
            "chembl37_exposure": manifest["endpoints"][endpoint_name].get("chembl37_exposure"),
            "environment": _environment(),
            **result,
        }
        write_shard(shard_path(output_dir, endpoint_name, representation, probe), payload)
        written.append(probe)

    return {
        "endpoint": endpoint_name, "representation": representation, "status": "ok",
        "probes": written, "cache_hit": features.cache_hit,
        "feature_seconds": features.seconds, "feature_failures": len(features.failures),
        "dimension": features.dimension, "cleaned_molecules": endpoint.size,
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def plan(manifest, output_dir: Path, *, endpoints=None, representations=None):
    release_identity = manifest["release_identity_sha256"]
    seeds = list(a2.SEEDS)
    cells = experiment_matrix(manifest)
    if endpoints:
        cells = [c for c in cells if c[0] in set(endpoints)]
    if representations:
        cells = [c for c in cells if c[1] in set(representations)]

    done, pending = [], {}
    for endpoint, representation, probe in cells:
        shard = read_valid_shard(
            shard_path(output_dir, endpoint, representation, probe),
            release_identity=release_identity, expected_seeds=seeds, expected_rows=len(seeds) * 4)
        if shard is not None:
            done.append((endpoint, representation, probe))
        else:
            pending[(endpoint, representation)] = None
    jobs = [{"endpoint": e, "representation": r} for e, r in pending]
    return jobs, done


def execute(*, frozen_dir: Path, manifest_path: Path, output_dir: Path, cache_dir: Path,
            workers: int, endpoints=None, representations=None,
            expected_release_identity: str | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text("utf-8"))
    release_identity = manifest["release_identity_sha256"]
    if expected_release_identity and release_identity != expected_release_identity:
        raise a2.TrackA2Error(
            f"release identity {release_identity} != expected {expected_release_identity}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs, done = plan(manifest, output_dir, endpoints=endpoints, representations=representations)
    for job in jobs:
        job.update({"frozen_dir": str(frozen_dir), "manifest_path": str(manifest_path),
                    "output_dir": str(output_dir), "cache_dir": str(cache_dir)})

    print(f"Track A2: {len(done)} cells already valid, {len(jobs)} units to run "
          f"on {workers} worker(s)", flush=True)

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
                    outcomes.append({"endpoint": job["endpoint"],
                                     "representation": job["representation"],
                                     "status": "failed",
                                     "error": f"{type(exc).__name__}: {exc}"})
                _report(outcomes[-1], finished, len(jobs))

    failed = [o for o in outcomes if o["status"] != "ok"]
    return {
        "release_identity": release_identity, "units_run": len(outcomes),
        "units_failed": len(failed), "failures": failed,
        "cells_already_valid": len(done), "workers": workers,
        "wall_seconds": time.perf_counter() - started,
        "feature_seconds": sum(o.get("feature_seconds", 0.0) for o in outcomes),
        "cache_hits": sum(1 for o in outcomes if o.get("cache_hit")),
    }


def _report(outcome, index, total) -> None:
    tag = "ok " if outcome["status"] == "ok" else "FAIL"
    if outcome["status"] == "ok":
        extra = (f" feat={outcome['feature_seconds']:6.1f}s"
                 f" {'cached' if outcome['cache_hit'] else 'cold  '}"
                 f" d={outcome['dimension']:>5} n={outcome['cleaned_molecules']:>6}")
    else:
        extra = f" {outcome.get('error', '')[:110]}"
    print(f"[{index:>4}/{total}] {tag} {outcome['endpoint']:<32}"
          f"{outcome['representation']:<28}{outcome.get('seconds', 0):7.1f}s{extra}", flush=True)


# ---------------------------------------------------------------------------
# collection and QC
# ---------------------------------------------------------------------------


def collect(output_dir: Path, manifest):
    release_identity = manifest["release_identity_sha256"]
    seeds = list(a2.SEEDS)
    rows, present, missing = [], set(), []
    identities, cleaning, distinctness, timings = {}, {}, {}, []
    split_overlaps = []

    for endpoint, representation, probe in experiment_matrix(manifest):
        shard = read_valid_shard(
            shard_path(output_dir, endpoint, representation, probe),
            release_identity=release_identity, expected_seeds=seeds, expected_rows=len(seeds) * 4)
        if shard is None:
            missing.append((endpoint, representation, probe))
            continue
        present.add((endpoint, representation, probe))
        rows.extend(shard["rows"])
        identities[f"{endpoint}/{representation}/{probe}"] = shard["cell_identity"]
        cleaning[endpoint] = shard["cleaning"]
        distinctness[endpoint] = shard["split_distinctness"]
        for seed, audit in shard["split_audits"].items():
            split_overlaps.append({
                "endpoint": endpoint, "seed": int(seed), **audit["sizes"],
                "molecule_overlap_train_test": audit["molecule_overlap"]["train_test"],
                "scaffold_overlap_train_test": audit["scaffold_overlap"]["train_test"],
                "molecule_overlap_train_validation": audit["molecule_overlap"]["train_validation"],
                "scaffold_overlap_train_validation": audit["scaffold_overlap"]["train_validation"],
                "molecule_overlap_validation_test": audit["molecule_overlap"]["validation_test"],
                "scaffold_overlap_validation_test": audit["scaffold_overlap"]["validation_test"],
                "distinct_scaffolds": audit["distinct_scaffolds"],
            })
        for timing in shard["timings"]:
            timings.append({**timing, "endpoint": endpoint,
                            "representation": representation, "probe": probe})

    expected = expected_counts(manifest)
    keys = [(r["endpoint"], r["representation"], r["probe"], r["seed"], r["metric"]) for r in rows]
    duplicates = len(keys) - len(set(keys))
    bad = [r for r in rows if r["metric_value"] is None or
           (isinstance(r["metric_value"], float) and
            (r["metric_value"] != r["metric_value"] or abs(r["metric_value"]) == float("inf")))]

    leaky = [s for s in split_overlaps if any(
        s[k] for k in s if k.startswith(("molecule_overlap", "scaffold_overlap")))]

    audit = {
        "expected": expected,
        "observed": {
            "cells": len(present), "result_rows": len(rows),
            "endpoints": len({c[0] for c in present}),
            "representations": len({c[1] for c in present}),
            "probes": len({c[2] for c in present}),
            "seeds": len({r["seed"] for r in rows}),
        },
        "missing_cells": [list(c) for c in missing],
        "duplicate_rows": duplicates,
        "nan_or_inf_rows": [{"endpoint": r["endpoint"], "representation": r["representation"],
                             "probe": r["probe"], "seed": r["seed"], "metric": r["metric"],
                             "value": r["metric_value"]} for r in bad],
        "splits_with_any_overlap": leaky,
        "all_test_sets_distinct": all(d["all_test_sets_distinct"] for d in distinctness.values()),
        "cell_identities": identities,
        "complete": (not missing and duplicates == 0
                     and len(rows) == expected["result_rows"] and not leaky),
    }
    return rows, {"audit": audit, "timings": timings, "cleaning": cleaning,
                  "distinctness": distinctness, "split_overlaps": split_overlaps}


def write_results(path: Path, rows) -> str:
    ordered = sorted(rows, key=lambda r: (
        r["endpoint"], r["representation"], r["probe"], r["seed"], r["metric"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_COLUMNS),
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    return release.sha256_file(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the Track A2 benchmark.")
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
            frozen_dir=args.frozen_dir, manifest_path=args.manifest,
            output_dir=args.output_dir, cache_dir=args.cache_dir, workers=args.workers,
            endpoints=args.endpoints, representations=args.representations,
            expected_release_identity=args.expect_release)

    rows, collected = collect(args.output_dir, manifest)
    audit = collected["audit"]
    digest = write_results(args.output_dir / "results_track_a2.csv", rows)

    report = {
        "run_summary": summary, "audit": audit, "result_rows": len(rows),
        "results_file_sha256": digest,
        "scientific_identity_sha256": scientific_identity(rows),
        "scientific_identity_columns": list(SCIENTIFIC_COLUMNS),
        "cleaning": collected["cleaning"], "split_distinctness": collected["distinctness"],
        "feature_cache_contract": feature_store.cache_contract(),
        "environment": _environment(),
    }
    for name, payload in (("run_report.json", report),
                          ("timings.json", collected["timings"]),
                          ("split_audits.json", collected["split_overlaps"])):
        with open(args.output_dir / name, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True, default=str)
            handle.write("\n")

    print(f"\ncells      {audit['observed']['cells']}/{audit['expected']['cells']}")
    print(f"rows       {len(rows)}/{audit['expected']['result_rows']}")
    print(f"missing    {len(audit['missing_cells'])}")
    print(f"duplicates {audit['duplicate_rows']}")
    print(f"nan/inf    {len(audit['nan_or_inf_rows'])}")
    print(f"leaky splits {len(audit['splits_with_any_overlap'])}")
    print(f"distinct test sets everywhere {audit['all_test_sets_distinct']}")
    print(f"complete   {audit['complete']}")
    print(f"file sha   {digest}")
    print(f"science    {report['scientific_identity_sha256']}")
    return 0 if audit["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
