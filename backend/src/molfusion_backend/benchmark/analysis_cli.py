"""Produce the Track A1 derived analysis tables.

Reads the immutable raw result matrix, writes derived tables to a separate
directory, and never touches the raw files. Run from the repository root:

    .\\backend\\.venv\\Scripts\\python.exe -m molfusion_backend.benchmark.analysis_cli
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from molfusion_backend.benchmark import analysis, metrics, protocol

RAW_DIR = Path("backend/benchmark_runs/track_a1")
OUT_DIR = Path("backend/benchmark_runs/track_a1/analysis")
MANIFEST = Path("backend/benchmark_manifests/tdc_admet_group.json")

EXPECTED_IDENTITY = "d40ef09b398f47914aa51f99fd6a4f5893f7778b50c0cca04404b575632de868"
EXPECTED_ROWS = 6160

#: Endpoints where Phase 6A.1 measured that MolFusion cleaning would move
#: more than CLEANING_DIVERGENCE_ALERT of an official partition. Used ONLY
#: for a sensitivity check -- the official A1 result stays all 22 endpoints.
HIGH_REPLICATE_ENDPOINTS = ("ppbr_az", "clearance_hepatocyte_az")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text("utf-8"))

    # --- verify, then never write to raw again -------------------------
    raw_path = args.raw_dir / "results_track_a1.csv"
    before = raw_path.stat()
    rows = analysis.load_raw_results(raw_path)
    provenance = analysis.verify_raw_results(
        rows, expected_identity=EXPECTED_IDENTITY, expected_rows=EXPECTED_ROWS
    )
    print(f"raw identity verified: {provenance['scientific_identity']}")
    print(f"  {provenance['rows']} rows, {provenance['endpoints']} endpoints, "
          f"{provenance['representations']} representations, probes {provenance['probes']}, "
          f"seeds {provenance['seeds']}")

    scores = analysis.aggregate_seeds(rows)
    ranks = analysis.rank_table(scores)
    checksums: dict[str, str] = {}

    def emit(name: str, table: list[dict[str, Any]]) -> None:
        checksums[name] = analysis.write_table(out / name, table)
        print(f"  wrote {name:<38} {len(table):>5} rows")

    # --- Table 2: endpoint-level performance ---------------------------
    emit("endpoint_summary.csv", [s.as_row() for s in scores])

    # --- endpoint-level ranks (derived data) ---------------------------
    rank_rows = [
        {"endpoint": e, "probe": p, "representation": r, "rank": v,
         "task_type": next(s.task_type for s in scores if s.endpoint == e)}
        for (e, p), row in sorted(ranks.items()) for r, v in sorted(row.items())
    ]
    emit("endpoint_ranks.csv", rank_rows)

    # --- Table 3: cross-endpoint ranking -------------------------------
    classification = sorted({s.endpoint for s in scores
                             if s.task_type == protocol.TASK_CLASSIFICATION})
    regression = sorted({s.endpoint for s in scores
                         if s.task_type == protocol.TASK_REGRESSION})
    all_endpoints = sorted({s.endpoint for s in scores})
    without_flagged = [e for e in all_endpoints if e not in HIGH_REPLICATE_ENDPOINTS]

    summary: list[dict[str, Any]] = []
    for probe in protocol.PROBES:
        for subset, endpoints in (
            ("all", all_endpoints),
            ("classification", classification),
            ("regression", regression),
            ("excl_high_replicate", without_flagged),
        ):
            for row in analysis.summarise_ranks(ranks, probe=probe, endpoints=endpoints):
                summary.append({"subset": subset, **row})
    emit("representation_ranks.csv", summary)

    # --- pairwise descriptive ------------------------------------------
    pairwise: list[dict[str, Any]] = []
    for probe in protocol.PROBES:
        pairwise += analysis.pairwise_wins(scores, probe=probe)
        pairwise += analysis.pairwise_wins(scores, probe=probe,
                                           task_type=protocol.TASK_CLASSIFICATION)
        pairwise += analysis.pairwise_wins(scores, probe=probe,
                                           task_type=protocol.TASK_REGRESSION)
    emit("pairwise_wins.csv", pairwise)

    # --- nonlinear gain -------------------------------------------------
    gains = analysis.nonlinear_gain(scores)
    emit("nonlinear_gain.csv", gains)
    by_rep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in gains:
        by_rep[g["representation"]].append(g)
    gain_summary = sorted(
        (
            {
                "representation": r,
                "n_endpoints": len(v),
                "mean_normalised_gain": float(np.mean([g["normalised_gain"] for g in v])),
                "median_normalised_gain": float(np.median([g["normalised_gain"] for g in v])),
                "mean_rank_gain": float(np.mean([g["rank_gain"] for g in v])),
                "endpoints_improved": int(sum(1 for g in v if g["normalised_gain"] > 0)),
            }
            for r, v in by_rep.items()
        ),
        key=lambda r: -r["mean_normalised_gain"],
    )
    emit("nonlinear_gain_summary.csv", gain_summary)

    # --- hyperparameter boundary audit ---------------------------------
    emit("hyperparameter_audit.csv", hyperparameter_audit(rows))

    # --- cost -----------------------------------------------------------
    emit("cost_summary.csv", cost_summary(rows, manifest))

    # --- endpoint instability -------------------------------------------
    emit("endpoint_instability.csv", endpoint_instability(rows, scores, manifest))

    # --- ChEMBL exposure -------------------------------------------------
    exposure, exposure_stats = tfidf_exposure(scores, ranks, manifest)
    emit("tfidf_exposure_analysis.csv", exposure)

    # --- statistics -------------------------------------------------------
    tests: list[dict[str, Any]] = []
    omnibus: list[dict[str, Any]] = []
    for probe in protocol.PROBES:
        for task in (None, protocol.TASK_CLASSIFICATION, protocol.TASK_REGRESSION):
            result = analysis.friedman(scores, probe=probe, task_type=task)
            omnibus.append(result)
            # Post-hoc only where the omnibus rejects -- the frozen plan.
            if result.get("runnable") and result.get("reject_at_alpha"):
                tests += analysis.pairwise_tests(scores, probe=probe, task_type=task)
    emit("friedman.csv", omnibus)
    emit("statistical_tests.csv", tests)

    bootstrap: list[dict[str, Any]] = []
    for probe in protocol.PROBES:
        bootstrap += analysis.bootstrap_mean_rank(ranks, probe=probe)
    emit("bootstrap_mean_rank.csv", bootstrap)

    # --- Table 1: representation characteristics -------------------------
    emit("representation_characteristics.csv", representation_characteristics(rows, manifest))

    # --- figure-ready data ------------------------------------------------
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    checksums["figures/figure_a_rank_heatmap.csv"] = analysis.write_table(
        figures / "figure_a_rank_heatmap.csv", rank_rows
    )
    checksums["figures/figure_b_mean_rank_ci.csv"] = analysis.write_table(
        figures / "figure_b_mean_rank_ci.csv", bootstrap
    )
    checksums["figures/figure_c_task_ranks.csv"] = analysis.write_table(
        figures / "figure_c_task_ranks.csv",
        [r for r in summary if r["subset"] in ("classification", "regression")],
    )
    checksums["figures/figure_d_rank_vs_cost.csv"] = analysis.write_table(
        figures / "figure_d_rank_vs_cost.csv", rank_vs_cost(summary, rows, manifest)
    )
    checksums["figures/figure_e_nonlinear_gain.csv"] = analysis.write_table(
        figures / "figure_e_nonlinear_gain.csv", gain_summary
    )
    checksums["figures/figure_f_tfidf_exposure.csv"] = analysis.write_table(
        figures / "figure_f_tfidf_exposure.csv", exposure
    )
    print(f"  wrote 6 figure-ready tables")

    # --- immutability + identity ------------------------------------------
    after = raw_path.stat()
    if (before.st_size, before.st_mtime) != (after.st_size, after.st_mtime):
        raise analysis.AnalysisError("raw results changed during analysis")

    configuration = {
        "seed_aggregation": analysis.SEED_AGGREGATION,
        "ranking": "direction-aware within endpoint, average ties",
        "statistical_unit": "endpoint",
        "omnibus": "friedman",
        "post_hoc": "paired wilcoxon signed-rank, only where omnibus rejects",
        "correction": protocol.MULTIPLE_COMPARISON_CORRECTION,
        "effect_size": protocol.EFFECT_SIZE,
        "alpha": protocol.ALPHA,
        "bootstrap_resamples": protocol.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": analysis.BOOTSTRAP_SEED,
        "bootstrap_unit": "endpoint",
        "high_replicate_endpoints": list(HIGH_REPLICATE_ENDPOINTS),
    }
    identity = analysis.analysis_identity(
        raw_identity=provenance["scientific_identity"], configuration=configuration
    )
    report = {
        "analysis_version": analysis.ANALYSIS_VERSION,
        "analysis_identity_sha256": identity,
        "raw_provenance": provenance,
        "raw_results_immutable": True,
        "configuration": configuration,
        "table_checksums": checksums,
        "exposure_correlation": exposure_stats,
    }
    with open(out / "analysis_report.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
        handle.write("\n")

    print(f"\nanalysis identity: {identity}")
    print(f"raw results unchanged: True")
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def hyperparameter_audit(rows) -> list[dict[str, Any]]:
    """How often each candidate was selected, and how often at a grid edge."""
    edges: dict[str, set] = {}
    for probe in protocol.PROBES:
        for task in protocol.TASK_TYPES:
            from molfusion_backend.benchmark import pipelines

            grid = pipelines.hyperparameter_grid(probe, task)
            for key in grid[0]:
                values = sorted({g[key] for g in grid})
                edges[f"{probe}|{task}|{key}"] = {values[0], values[-1]}

    counts: dict[tuple, int] = defaultdict(int)
    for row in rows:
        if row["metric"] != row["molfusion_primary_metric"]:
            continue
        params = json.loads(row["hyperparameters"])
        for key, value in params.items():
            counts[(row["probe"], row["task_type"], row["representation"], key, value)] += 1

    out = []
    grouped: dict[tuple, int] = defaultdict(int)
    for (probe, task, rep, key, value), n in counts.items():
        grouped[(probe, task, rep, key)] += n
    for (probe, task, rep, key, value), n in sorted(counts.items(), key=lambda x: str(x[0])):
        total = grouped[(probe, task, rep, key)]
        out.append(
            {
                "probe": probe,
                "task_type": task,
                "representation": rep,
                "parameter": key,
                "value": value,
                "selections": n,
                "share": n / total if total else 0.0,
                "is_grid_boundary": value in edges.get(f"{probe}|{task}|{key}", set()),
            }
        )
    return out


def cost_summary(rows, manifest) -> list[dict[str, Any]]:
    """Per representation and probe: fit, selection, prediction, features."""
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    features: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["metric"] != row["molfusion_primary_metric"]:
            continue
        key = (row["representation"], row["probe"])
        agg[key]["selection_seconds"] += row["selection_seconds"] / 4  # 4 metrics per cell-seed
        agg[key]["fit_seconds"] += row["fit_seconds"] / 1
        agg[key]["test_predict_seconds"] += row["predict_seconds"]
        agg[key]["validation_predict_seconds"] += row["validation_predict_seconds"]
        agg[key]["n"] += 1
        features[(row["endpoint"], row["representation"])] = row["feature_seconds"]

    feature_total: dict[str, float] = defaultdict(float)
    for (_endpoint, representation), seconds in features.items():
        feature_total[representation] += seconds

    dims = {r: manifest["endpoints"]["ames"]["representation_availability"][r]["dimension"]
            for r in protocol.TRACK_A_REPRESENTATIONS}
    out = []
    for (representation, probe), values in sorted(agg.items()):
        n = values["n"]
        out.append(
            {
                "representation": representation,
                "probe": probe,
                "dimension": dims[representation],
                "cell_seeds": int(n),
                "selection_seconds_total": values["selection_seconds"],
                "final_fit_seconds_total": values["fit_seconds"],
                "model_seconds_total": values["selection_seconds"] + values["fit_seconds"],
                "mean_final_fit_seconds": values["fit_seconds"] / n if n else 0.0,
                "test_predict_seconds_total": values["test_predict_seconds"],
                "validation_predict_seconds_total": values["validation_predict_seconds"],
                "feature_seconds_total_all_endpoints": feature_total[representation],
            }
        )
    return out


def endpoint_instability(rows, scores, manifest) -> list[dict[str, Any]]:
    """Where the five realizations disagree most, plus known caveats."""
    spread: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        span = score.maximum - score.minimum
        denominator = abs(score.mean) if score.mean else 1.0
        spread[score.endpoint].append(span / denominator)

    out = []
    for endpoint, values in sorted(spread.items()):
        entry = manifest["endpoints"][endpoint]
        ingestion = entry["ingestion"]
        conflicting = ingestion["duplicates_conflicting_dropped"]
        out.append(
            {
                "endpoint": endpoint,
                "task_type": entry["task_type"],
                "usable_molecules": entry["usable"],
                "mean_relative_seed_spread": float(np.mean(values)),
                "max_relative_seed_spread": float(np.max(values)),
                "small_sample": entry["usable"] < 1000,
                "conflicting_records_dropped_by_cleaning": conflicting,
                "high_replicate_flag": endpoint in HIGH_REPLICATE_ENDPOINTS,
                "minority_fraction": entry["label_summary"].get("minority_fraction"),
                "strong_imbalance": (
                    entry["label_summary"].get("minority_fraction", 1.0) < 0.2
                    if entry["task_type"] == protocol.TASK_CLASSIFICATION else False
                ),
            }
        )
    return sorted(out, key=lambda r: -r["mean_relative_seed_spread"])


def tfidf_exposure(scores, ranks, manifest):
    """TF-IDF rank against frozen ChEMBL 37 overlap, descriptively."""
    from scipy.stats import pearsonr, spearmanr

    rows = []
    for endpoint in sorted({s.endpoint for s in scores}):
        exposure = manifest["endpoints"][endpoint].get("chembl37_exposure") or {}
        entry = {
            "endpoint": endpoint,
            "task_type": manifest["endpoints"][endpoint]["task_type"],
            "chembl_overlap_fraction": exposure.get("overlap_fraction"),
            "benchmark_molecules": exposure.get("benchmark_molecules"),
            "present_in_chembl37": exposure.get("present_in_chembl37"),
        }
        for probe in protocol.PROBES:
            entry[f"tfidf_rank_{probe}"] = ranks[(endpoint, probe)]["smiles_tfidf_4096"]
        rows.append(entry)

    stats = {}
    overlap = np.array([r["chembl_overlap_fraction"] for r in rows], dtype=float)
    for probe in protocol.PROBES:
        rank_values = np.array([r[f"tfidf_rank_{probe}"] for r in rows], dtype=float)
        pearson = pearsonr(overlap, rank_values)
        spearman = spearmanr(overlap, rank_values)
        stats[probe] = {
            "n_endpoints": len(rows),
            "pearson_r": float(pearson[0]),
            "pearson_p": float(pearson[1]),
            "spearman_rho": float(spearman[0]),
            "spearman_p": float(spearman[1]),
            "note": (
                "Rank is better when lower, so a NEGATIVE correlation would "
                "mean TF-IDF ranks better on high-exposure endpoints. n=22 and "
                "exposure is unsupervised, so this is descriptive only."
            ),
        }
    return rows, stats


def representation_characteristics(rows, manifest) -> list[dict[str, Any]]:
    """Table 1: type, dimension, sparsity, feature cost."""
    from molfusion_backend.agents import registry

    sparsity: dict[str, list[float]] = defaultdict(list)
    for meta in Path("backend/benchmark_cache/features").glob("**/metadata.json"):
        payload = json.loads(meta.read_text("utf-8"))
        value = payload.get("nonzero_fraction")
        if value is not None:
            sparsity[payload["agent_id"]].append(1.0 - value)

    features: dict[tuple[str, str], float] = {}
    for row in rows:
        features[(row["endpoint"], row["representation"])] = row["feature_seconds"]
    totals: dict[str, float] = defaultdict(float)
    for (_e, representation), seconds in features.items():
        totals[representation] += seconds

    out = []
    for representation in protocol.TRACK_A_REPRESENTATIONS:
        agent = registry.get(representation)
        values = sparsity.get(representation, [])
        out.append(
            {
                "representation": representation,
                "value_type": agent.value_type,
                "dimension": agent.output_dim,
                "dtype": "float64",
                "mean_sparsity": float(np.mean(values)) if values else None,
                "feature_seconds_total": totals[representation],
                "agent_version": agent.version,
            }
        )
    return out


def rank_vs_cost(summary, rows, manifest) -> list[dict[str, Any]]:
    """Figure D data: mean rank against model and feature cost."""
    costs = {(c["representation"], c["probe"]): c for c in cost_summary(rows, manifest)}
    out = []
    for row in summary:
        if row["subset"] != "all":
            continue
        cost = costs.get((row["representation"], row["probe"]))
        if cost is None:
            continue
        out.append(
            {
                "representation": row["representation"],
                "probe": row["probe"],
                "dimension": cost["dimension"],
                "mean_rank": row["mean_rank"],
                "median_rank": row["median_rank"],
                "model_seconds_total": cost["model_seconds_total"],
                "mean_final_fit_seconds": cost["mean_final_fit_seconds"],
                "feature_seconds_total": cost["feature_seconds_total_all_endpoints"],
            }
        )
    return out


if __name__ == "__main__":
    sys.exit(main())
