"""Assemble the tracked provenance manifest for a frozen benchmark release.

The downloaded data is git-ignored; this manifest is what makes those
ignored bytes verifiable. It carries every checksum, count and split
identity needed to answer "is this the data the protocol was frozen
against?" without re-downloading anything -- and without PyTDC, which is not
a MolFusion dependency.

Run from the repository root:

    .\\backend\\.venv\\Scripts\\python.exe -m molfusion_backend.benchmark.manifest_cli
"""

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy
import rdkit

from molfusion_backend.benchmark import protocol, release, tdc

RELEASE_NAME = "TDC-ADMET-2026-09"
DEFAULT_FROZEN = Path("backend/benchmark_data/frozen")
DEFAULT_OUTPUT = Path("backend/benchmark_manifests/tdc_admet_group.json")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_manifest(
    frozen_dir: Path,
    *,
    features_overlap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit every frozen endpoint and assemble the manifest."""
    metadata = json.loads((frozen_dir / "_tdc_metadata.json").read_text("utf-8"))

    endpoints: dict[str, Any] = {}
    for name in sorted(metadata["endpoints"]):
        entry = metadata["endpoints"][name]
        audit = tdc.audit_endpoint(name=name, metadata=entry, frozen_dir=frozen_dir)
        record = asdict(audit)
        record["tdc_identity"] = {
            "dataset_name": entry["tdc_dataset_name"],
            "dataset_id": entry.get("tdc_dataset_id"),
            "split_method": entry["tdc_split_method"],
        }
        record["train_val"] = audit.checksums["train_val"]
        record["test"] = audit.checksums["test"]
        if features_overlap and name in features_overlap:
            extra = features_overlap[name]
            record["representation_availability"] = extra["per_representation"]
            record["common_evaluation_set"] = extra["common_evaluation_set"]
            record["feature_accounting"] = extra["accounting"]
            record["chembl37_exposure"] = extra["chembl_overlap"]
        endpoints[name] = record

    identity = release.release_identity(
        release_name=RELEASE_NAME,
        protocol_version=protocol.PROTOCOL_VERSION,
        endpoints=endpoints,
    )

    included = [n for n, e in endpoints.items() if e["included"]]
    return {
        "release_name": RELEASE_NAME,
        "release_identity_sha256": identity,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "canonicalization_contract": protocol.CANONICALIZATION_ID,
        "serialization_contract": release.SERIALIZATION_CONTRACT,
        "source": {
            "suite": "Therapeutics Data Commons ADMET benchmark group",
            "pytdc_version": metadata["pytdc_version"],
            "group": metadata["group"],
            "acquisition": "single download; audits read frozen files only",
        },
        "official_split_semantics": {
            "test_set": "shipped file; get()/__next__() take no seed",
            "train_valid": "get_train_valid_split(seed) splits train_val only",
            "train_val_fractions": list(tdc.OFFICIAL_TRAIN_VAL_FRACTIONS),
            "split_method": tdc.OFFICIAL_SPLIT_METHOD,
            "seeds": list(tdc.OFFICIAL_SEEDS),
            "scaffold_includes_chirality": tdc.OFFICIAL_SCAFFOLD_INCLUDES_CHIRALITY,
            "test_identity_invariant_across_seeds": all(
                e["split_identity"]["test_identity_invariant_across_seeds"]
                for e in endpoints.values()
            ),
        },
        "evaluation_tracks": protocol.protocol_summary()["evaluation_tracks"],
        "endpoint_count": len(endpoints),
        "included_endpoint_count": len(included),
        "excluded_endpoints": {
            n: e["exclusion_reasons"] for n, e in endpoints.items() if not e["included"]
        },
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "numpy": numpy.__version__,
            "molfusion_git_commit": _git("rev-parse", "HEAD"),
            "molfusion_git_working_tree_clean": _git("status", "--porcelain") == "",
        },
        "endpoints": endpoints,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--features-overlap",
        type=Path,
        default=None,
        help="optional JSON from the representation/ChEMBL availability pass",
    )
    args = parser.parse_args(argv)

    extra = None
    if args.features_overlap and args.features_overlap.exists():
        extra = json.loads(args.features_overlap.read_text("utf-8"))

    manifest = build_manifest(args.frozen_dir, features_overlap=extra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")

    print(f"release        {manifest['release_name']}")
    print(f"identity       {manifest['release_identity_sha256']}")
    print(f"endpoints      {manifest['included_endpoint_count']}/{manifest['endpoint_count']} included")
    print(f"written        {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
