"""Export Supplementary Tables S1-S11 as deterministic CSV data files.

The captions for these tables have existed since Phase 6C.5; the data files
they refer to had not been exported. This copies them out of the frozen
publication evidence package -- it recomputes nothing. Each source table is
read, its columns are projected to the caption's stated content, and the
result is written with a fixed serialisation so the same inputs always
produce the same bytes and the same SHA-256.

A manifest records, per file, the row and column counts, the digest and the
frozen source it came from, so a reader can tie every supplementary file
back to the analysis output that produced it.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE = Path("backend/benchmark_runs/publication")
A1 = Path("backend/benchmark_runs/track_a1/analysis")
A2 = Path("backend/benchmark_runs/track_a2/analysis")
OUT = Path("docs/manuscript/submission/jcheminformatics/supplementary/tables")

#: (id, filename stem, source path, description). S10 and S11 are the former
#: main Tables 6 and 7, demoted under the journal layout; their data is
#: unchanged, only their numbering moved.
EXPORTS: tuple[tuple[str, str, Path, str], ...] = (
    ("S1", "supplementary_table_S1_pairwise_contrasts_a2",
     PACKAGE / "supplementary" / "all_pairwise_contrasts_a2.csv",
     "Complete pairwise contrasts, Track A2"),
    ("S2", "supplementary_table_S2_pairwise_contrasts_a1",
     PACKAGE / "supplementary" / "all_pairwise_contrasts_a1.csv",
     "Complete pairwise contrasts, Track A1"),
    ("S3", "supplementary_table_S3_contrast_reproduction",
     PACKAGE / "supplementary" / "a1_vs_a2_contrasts.csv",
     "Reproduction of Track A1 contrasts under Track A2"),
    ("S4", "supplementary_table_S4_endpoint_ranks",
     None, "Endpoint-level ranks for both tracks (assembled)"),
    ("S5", "supplementary_table_S5_chembl_exposure",
     PACKAGE / "supplementary" / "chembl_exposure.csv",
     "ChEMBL 37 corpus exposure by endpoint"),
    ("S6", "supplementary_table_S6_split_stability",
     PACKAGE / "supplementary" / "split_stability_detail.csv",
     "Per-partition stability detail"),
    ("S7", "supplementary_table_S7_curation_and_distinctness",
     None, "Curation effects and partition distinctness by endpoint (assembled)"),
    ("S8", "supplementary_table_S8_friedman",
     None, "Friedman omnibus results for both tracks (assembled)"),
    ("S9", "supplementary_table_S9_provenance_audit",
     None, "Execution provenance audit (assembled)"),
    ("S10", "supplementary_table_S10_endpoint_stability",
     PACKAGE / "tables" / "table6_endpoint_stability.csv",
     "Endpoint stability (former main Table 6)"),
    ("S11", "supplementary_table_S11_endpoint_subsets",
     PACKAGE / "tables" / "table7_22_vs_19_endpoint_subset.csv",
     "All-endpoint and repartitioned-subset summaries (former main Table 7)"),
)


def read(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Fixed serialisation: UTF-8, LF, given column order, no re-sorting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble_endpoint_ranks(resolve=lambda p: p) -> list[dict[str, Any]]:
    """S4: both tracks' endpoint-level ranks in one table, track-labelled."""
    out = []
    for track, path in (("A1", PACKAGE / "supplementary" / "a1_endpoint_ranks.csv"),
                        ("A2", PACKAGE / "supplementary" / "a2_endpoint_ranks.csv")):
        for row in read(resolve(path)):
            out.append({"track": track, **row})
    return out


def assemble_curation_and_distinctness(resolve=lambda p: p) -> list[dict[str, Any]]:
    """S7: curation counts joined to the partition-distinctness measures."""
    cleaning = {r["endpoint"]: r for r in read(resolve(
        PACKAGE / "supplementary" / "cleaning_effects.csv"))}
    distinct = {r["endpoint"]: r for r in read(resolve(
        PACKAGE / "supplementary" / "split_distinctness.csv"))}
    out = []
    for endpoint in sorted(cleaning):
        clean, dist = cleaning[endpoint], distinct.get(endpoint, {})
        out.append({
            "endpoint": endpoint,
            "raw_rows": clean["raw_rows"],
            "usable": clean["usable"],
            "rdkit_invalid": clean["rdkit_invalid"],
            "duplicates_collapsed": clean["duplicates_collapsed"],
            "conflicting_molecule_count": clean["conflicting_molecule_count"],
            "duplicates_conflicting_dropped": clean["duplicates_conflicting_dropped"],
            "missing_label_dropped": clean["missing_label_dropped"],
            "mean_pairwise_test_jaccard": dist.get("mean_pairwise_test_jaccard", ""),
            "max_pairwise_test_jaccard": dist.get("max_pairwise_test_jaccard", ""),
            "distinct_test_sets": dist.get("distinct_test_sets", ""),
            "genuinely_repartitioned": dist.get("genuinely_repartitioned", ""),
        })
    return out


def assemble_friedman(resolve=lambda p: p) -> list[dict[str, Any]]:
    out = []
    for track, path in (("A1", PACKAGE / "supplementary" / "friedman_a1.csv"),
                        ("A2", PACKAGE / "supplementary" / "friedman_a2.csv")):
        for row in read(resolve(path)):
            out.append({"track": track, **row})
    return out


def assemble_provenance(resolve=lambda p: p) -> list[dict[str, Any]]:
    """S9: what each shard recorded, kept separate from what was reconstructed."""
    audit_path = resolve(Path("backend/benchmark_runs/provenance_audit.json"))
    if not audit_path.exists():
        return []
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    out = []
    for name, track in sorted(audit["tracks"].items()):
        recorded = track["recorded"]
        out.append({
            "track": name,
            "track_identifier": track["track"],
            "total_shards": track["total_shards"],
            "shards_with_recorded_commit": recorded["populated_commit_shards"],
            "shards_without_recorded_commit": recorded["null_commit_shards"],
            "recorded_commit_counts": "; ".join(
                f"{k}={v}" for k, v in recorded["per_commit_shard_counts"].items()),
            "reconstructed_execution_commits": "; ".join(
                track["reconstructed"]["execution_commits"]),
            "backfilled_into_shards": track["reconstructed"]["backfilled_into_shards"],
            "scientific_identity": track["scientific_identity"],
            "reconstruction_basis": track["reconstructed"]["basis"],
        })
    return out


ASSEMBLERS = {
    "S4": assemble_endpoint_ranks,
    "S7": assemble_curation_and_distinctness,
    "S8": assemble_friedman,
    "S9": assemble_provenance,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--root", type=Path, default=None,
        help="repository root; the default source paths are relative to it, "
             "so the exporter works from any working directory")
    args = parser.parse_args(argv)

    # Source paths are repository-relative. Resolve them against --root (or
    # a root derived from this file) so the exporter does not depend on the
    # caller's working directory.
    root = args.root or Path(__file__).resolve().parents[4]
    resolve = lambda path: path if path.is_absolute() else root / path

    manifest: list[dict[str, Any]] = []
    for identifier, stem, source, description in EXPORTS:
        if identifier in ASSEMBLERS:
            rows = ASSEMBLERS[identifier](resolve)
            origin = "assembled from frozen analysis outputs"
        else:
            located = resolve(source) if source else None
            if not located or not located.exists():
                print(f"  {identifier}: SOURCE MISSING {source}")
                continue
            rows = read(located)
            origin = str(source).replace("\\", "/")
        if not rows:
            print(f"  {identifier}: no rows, skipped")
            continue
        columns = list(rows[0])
        path = args.out / f"{stem}.csv"
        digest = write(path, rows, columns)
        manifest.append({
            "table": identifier,
            "filename": path.name,
            "description": description,
            "rows": len(rows),
            "columns": len(columns),
            "column_names": columns,
            "sha256": digest,
            "scientific_source": origin,
        })
        print(f"  {identifier:<4} {path.name:<58} {len(rows):>4} rows  "
              f"{len(columns):>2} cols  {digest[:16]}")

    manifest_path = args.out / "SUPPLEMENTARY_TABLE_MANIFEST.json"
    payload = {
        "schema": "molfusion_supplementary_manifest_v1",
        "note": ("Exported from the frozen publication evidence package. No "
                 "scientific value was recomputed."),
        "publication_evidence_identity":
            "5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18",
        "tables": manifest,
    }
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=1, sort_keys=False)
        handle.write("\n")
    print(f"\nwrote {manifest_path}  ({len(manifest)} tables)")
    return 0 if len(manifest) == len(EXPORTS) else 1


if __name__ == "__main__":
    sys.exit(main())
