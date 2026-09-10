"""Build the Phase 6B publication evidence package.

Reads the frozen A1/A2 raw results and their committed analysis outputs,
verifies their identities before using them, and emits tables, figures,
figure data, supplementary material, a claim registry and a single
deterministic package identity.

Computes no new statistics. Every number traces to a Phase 6A.3 or 6A.4
table; this stage decides what is publishable, not what is true.

    .\\backend\\.venv\\Scripts\\python.exe -m molfusion_backend.benchmark.publication_cli
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from molfusion_backend.benchmark import (
    analysis,
    protocol,
    publication,
    publication_claims,
    publication_figures,
)

A1_DIR = Path("backend/benchmark_runs/track_a1")
A2_DIR = Path("backend/benchmark_runs/track_a2")
OUT_DIR = Path("backend/benchmark_runs/publication")
MANIFEST = Path("backend/benchmark_manifests/tdc_admet_group.json")

#: Canonical representation order for every table and figure: by A2
#: nonlinear mean rank, the primary result. Recorded so ordering is a rule
#: rather than a choice made per table.
ORDER_RULE = "ascending A2 nonlinear mean rank, ties broken alphabetically"


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _f(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-dir", type=Path, default=A1_DIR)
    parser.add_argument("--a2-dir", type=Path, default=A2_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    out = args.out_dir
    for sub in ("tables", "figures", "figure_data", "supplementary", "evidence"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # --- verify inputs before trusting them ----------------------------
    a1_rows = analysis.load_raw_results(args.a1_dir / "results_track_a1.csv")
    a1_provenance = analysis.verify_raw_results(
        a1_rows, expected_identity=publication.A1_RAW_IDENTITY, expected_rows=6160)
    a2_report = json.loads((args.a2_dir / "run_report.json").read_text("utf-8"))
    a2_identity = a2_report["scientific_identity_sha256"]
    if a2_identity != publication.A2_RAW_IDENTITY:
        raise analysis.AnalysisError(
            f"A2 raw identity {a2_identity} != frozen {publication.A2_RAW_IDENTITY}")
    a2_analysis = json.loads(
        (args.a2_dir / "analysis" / "analysis_report.json").read_text("utf-8"))
    if a2_analysis["analysis_identity_sha256"] != publication.A2_ANALYSIS_IDENTITY:
        raise analysis.AnalysisError("A2 analysis identity does not match the frozen value")
    print(f"A1 raw identity  {a1_provenance['scientific_identity']}  verified")
    print(f"A2 raw identity  {a2_identity}  verified")
    print(f"A2 analysis id   {a2_analysis['analysis_identity_sha256']}  verified")

    manifest = json.loads(args.manifest.read_text("utf-8"))
    a1a, a2a = args.a1_dir / "analysis", args.a2_dir / "analysis"

    a1_ranks = _read(a1a / "representation_ranks.csv")
    a2_ranks = _read(a2a / "representation_ranks.csv")
    a1_boot = _read(a1a / "bootstrap_mean_rank.csv")
    a2_boot = _read(a2a / "bootstrap_mean_rank.csv")
    a1_tests = _read(a1a / "statistical_tests.csv")
    a2_tests = _read(a2a / "statistical_tests.csv")
    a1_endpoint_ranks = _read(a1a / "endpoint_ranks.csv")
    a2_endpoint_ranks = _read(a2a / "endpoint_ranks.csv")
    characteristics = _read(a1a / "representation_characteristics.csv")
    a1_cost = _read(a1a / "cost_summary.csv")
    comparison = _read(a2a / "a1_vs_a2_ranking.csv")
    contrasts = _read(a2a / "a1_vs_a2_contrasts.csv")
    hypotheses = _read(a2a / "hypotheses.csv")
    kendall = _read(a2a / "kendall_w.csv")
    cleaning = _read(a2a / "cleaning_effects.csv")
    distinctness = _read(a2a / "split_distinctness.csv")
    gain_comparison = _read(a2a / "a1_vs_a2_nonlinear_gain.csv")
    a2_friedman = _read(a2a / "friedman.csv")
    a1_friedman = _read(a1a / "friedman.csv")
    split_stability = _read(a2a / "split_stability.csv")

    for row in contrasts:
        row["reproduced"] = row["reproduced"] == "True"
        row["effect_direction_preserved"] = row["effect_direction_preserved"] == "True"

    # --- canonical ordering --------------------------------------------
    nonlinear_all = {r["representation"]: _f(r["mean_rank"])
                     for r in a2_ranks
                     if r["probe"] == protocol.PROBE_NONLINEAR and r["subset"] == "all"}
    representations = sorted(protocol.TRACK_A_REPRESENTATIONS,
                             key=lambda r: (nonlinear_all.get(r, 99), r))

    checksums: dict[str, str] = {}

    def emit(folder: str, name: str, rows, columns) -> None:
        path = out / folder / name
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns),
                                    extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        checksums[f"{folder}/{name}"] = publication.table_checksum(rows, columns)
        print(f"  {folder}/{name:<44} {len(rows):>5} rows")

    # --- CI separation, checked rather than assumed --------------------
    separation = {probe: publication.ci_separation(a2_boot, probe=probe)
                  for probe in protocol.PROBES}
    a1_separation = {probe: publication.ci_separation(a1_boot, probe=probe)
                     for probe in protocol.PROBES}

    # --- stability ------------------------------------------------------
    stability = publication.stability_table(kendall)
    disagreements = publication.stability_disagreements(stability)
    low_stability = [r["endpoint"] for r in stability if r["pre_registered_low_stability"]]

    # --- Table 1: representation characteristics -----------------------
    feature_seconds = {r["representation"]: _f(r["feature_seconds_total"])
                       for r in characteristics}
    char_by_name = {r["representation"]: r for r in characteristics}
    table1 = [{
        "representation": name,
        "category": publication.REPRESENTATION_CATEGORY[name],
        "dimension": int(char_by_name[name]["dimension"]),
        "value_type": char_by_name[name]["value_type"],
        "mean_sparsity": round(_f(char_by_name[name]["mean_sparsity"]), 4),
        "artifact_dependency": "fitted ChEMBL 37 TF-IDF artifact"
                               if name == "smiles_tfidf_4096" else "none (stateless)",
        "feature_seconds_total": round(feature_seconds.get(name, 0.0), 1),
        "agent_version": char_by_name[name]["agent_version"],
    } for name in representations]
    emit("tables", "table1_representation_characteristics.csv", table1, table1[0].keys())

    # --- Table 2: primary A1 performance -------------------------------
    def rank_row(rows, probe, name, subset):
        for row in rows:
            if (row["probe"] == probe and row.get("subset", "all") == subset
                    and row["representation"] == name):
                return row
        return {}

    table2 = []
    for name in representations:
        lin = rank_row(a1_ranks, protocol.PROBE_LINEAR, name, "all")
        non = rank_row(a1_ranks, protocol.PROBE_NONLINEAR, name, "all")
        lin_c = rank_row(a1_ranks, protocol.PROBE_LINEAR, name, "classification")
        non_c = rank_row(a1_ranks, protocol.PROBE_NONLINEAR, name, "classification")
        lin_r = rank_row(a1_ranks, protocol.PROBE_LINEAR, name, "regression")
        non_r = rank_row(a1_ranks, protocol.PROBE_NONLINEAR, name, "regression")
        table2.append({
            "representation": name,
            "category": publication.REPRESENTATION_CATEGORY[name],
            "linear_mean_rank": round(_f(lin.get("mean_rank")), 2),
            "nonlinear_mean_rank": round(_f(non.get("mean_rank")), 2),
            "linear_classification_rank": round(_f(lin_c.get("mean_rank")), 2),
            "nonlinear_classification_rank": round(_f(non_c.get("mean_rank")), 2),
            "linear_regression_rank": round(_f(lin_r.get("mean_rank")), 2),
            "nonlinear_regression_rank": round(_f(non_r.get("mean_rank")), 2),
            "linear_wins_22": int(lin.get("wins", 0)),
            "nonlinear_wins_22": int(non.get("wins", 0)),
            "linear_top3_22": int(lin.get("top3", 0)),
            "nonlinear_top3_22": int(non.get("top3", 0)),
        })
    emit("tables", "table2_a1_primary_performance.csv", table2, table2[0].keys())

    # --- Table 3: A2 robustness ----------------------------------------
    comp = {(r["probe"], r["subset"], r["representation"]): r for r in comparison}
    table3 = []
    for name in representations:
        for probe in protocol.PROBES:
            row = comp.get((probe, "all", name), {})
            sub = comp.get((probe, "repartitioned_only", name), {})
            cls = comp.get((probe, "classification", name), {})
            reg = comp.get((probe, "regression", name), {})
            a1r, a2r = _f(row.get("a1_mean_rank")), _f(row.get("a2_mean_rank"))
            movement = a2r - a1r
            table3.append({
                "representation": name,
                "probe": probe,
                "a1_mean_rank": round(a1r, 2),
                "a2_mean_rank": round(a2r, 2),
                "rank_displacement": round(movement, 2),
                "a1_position": row.get("a1_position"),
                "a2_position": row.get("a2_position"),
                "position_change": row.get("position_change"),
                "robustness_verdict": (
                    "reproduced" if abs(movement) <= 0.5
                    else "weakened" if movement > 0 else "strengthened"),
                "a2_repartitioned_only_mean_rank": round(_f(sub.get("a2_mean_rank")), 2),
                "a1_classification_rank": round(_f(cls.get("a1_mean_rank")), 2),
                "a2_classification_rank": round(_f(cls.get("a2_mean_rank")), 2),
                "a1_regression_rank": round(_f(reg.get("a1_mean_rank")), 2),
                "a2_regression_rank": round(_f(reg.get("a2_mean_rank")), 2),
            })
    emit("tables", "table3_a2_robustness.csv", table3, table3[0].keys())

    # --- Table 4: key contrasts (main text) ----------------------------
    reproduced_pairs = {(c["probe"], c["task_type"], c["a"], c["b"])
                        for c in contrasts if c["reproduced"]}
    key = []
    for row in a2_tests:
        if row["significant_after_holm"] != "True":
            continue
        pair = (row["probe"], row["task_type"], row["a"], row["b"])
        physchem = "rdkit_physchem_descriptors" in (row["a"], row["b"])
        if not (physchem or pair in reproduced_pairs):
            continue
        effect = _f(row["effect_size_rank_biserial"])
        superior = row["b"] if effect < 0 else row["a"]
        key.append({
            "probe": row["probe"],
            "task_family": row["task_type"],
            "representation_a": row["a"],
            "representation_b": row["b"],
            "p_holm": round(_f(row["p_holm"]), 4),
            "rank_biserial": round(effect, 2),
            "abs_effect": round(abs(effect), 2),
            "superior_representation": superior,
            "direction": f"{superior} ranks better",
            "reproduced_from_a1": pair in reproduced_pairs,
            "n_endpoints": row.get("n_endpoints", ""),
        })
    key.sort(key=lambda r: (r["p_holm"], -r["abs_effect"]))
    emit("tables", "table4_key_statistical_contrasts.csv", key, key[0].keys())

    # --- Table 5: computational cost -----------------------------------
    cost_by = {(r["representation"], r["probe"]): r for r in a1_cost}
    a2_timings = json.loads((args.a2_dir / "timings.json").read_text("utf-8"))
    a2_model = defaultdict(float)
    for entry in a2_timings:
        a2_model[(entry["representation"], entry["probe"])] += (
            entry["fit_seconds"] + entry["selection_seconds"])

    nonlinear_total = sum(
        _f(cost_by[(n, protocol.PROBE_NONLINEAR)]["model_seconds_total"])
        for n in representations)
    cheapest = min(
        _f(cost_by[(n, protocol.PROBE_NONLINEAR)]["model_seconds_total"])
        for n in representations)

    table5 = []
    for name in representations:
        lin = _f(cost_by[(name, protocol.PROBE_LINEAR)]["model_seconds_total"])
        non = _f(cost_by[(name, protocol.PROBE_NONLINEAR)]["model_seconds_total"])
        table5.append({
            "representation": name,
            "dimension": int(char_by_name[name]["dimension"]),
            "feature_seconds": round(feature_seconds.get(name, 0.0), 1),
            "linear_model_seconds": round(lin, 1),
            "nonlinear_model_seconds": round(non, 1),
            "total_model_seconds": round(lin + non, 1),
            "share_of_nonlinear_model_seconds": round(non / nonlinear_total, 4),
            "relative_nonlinear_cost": round(non / cheapest, 2),
            "a2_nonlinear_model_seconds": round(a2_model[(name, protocol.PROBE_NONLINEAR)], 1),
            "nonlinear_mean_rank": round(nonlinear_all.get(name, float("nan")), 2),
            "a1_nonlinear_mean_rank": round(_f(rank_row(
                a1_ranks, protocol.PROBE_NONLINEAR, name, "all").get("mean_rank")), 2),
        })
    emit("tables", "table5_computational_cost.csv", table5, table5[0].keys())

    # --- stability + subset tables --------------------------------------
    emit("tables", "table6_endpoint_stability.csv", stability, stability[0].keys())

    subset_rows = []
    for probe in protocol.PROBES:
        for subset in ("all", "repartitioned_only"):
            for row in a2_ranks:
                if row["probe"] == probe and row["subset"] == subset:
                    subset_rows.append({
                        "probe": probe,
                        "subset": subset,
                        "n_endpoints": row["n_endpoints"],
                        "representation": row["representation"],
                        "mean_rank": round(_f(row["mean_rank"]), 2),
                        "wins": row["wins"],
                        "top3": row["top3"],
                    })
    subset_rows.sort(key=lambda r: (r["probe"], r["subset"], r["mean_rank"]))
    emit("tables", "table7_22_vs_19_endpoint_subset.csv", subset_rows, subset_rows[0].keys())

    # --- ChEMBL exposure -------------------------------------------------
    exposure = []
    tfidf_a1 = {(r["endpoint"], r["probe"]): _f(r["rank"]) for r in a1_endpoint_ranks
                if r["representation"] == "smiles_tfidf_4096"}
    tfidf_a2 = {(r["endpoint"], r["probe"]): _f(r["rank"]) for r in a2_endpoint_ranks
                if r["representation"] == "smiles_tfidf_4096"}
    for endpoint, entry in sorted(manifest["endpoints"].items()):
        block = entry.get("chembl37_exposure") or {}
        for probe in protocol.PROBES:
            exposure.append({
                "endpoint": endpoint,
                "probe": probe,
                "chembl37_overlap_fraction": round(block.get("overlap_fraction", 0.0), 4),
                "molecules_in_chembl37": block.get("present_in_chembl37"),
                "benchmark_molecules": block.get("benchmark_molecules"),
                "tfidf_a1_rank": tfidf_a1.get((endpoint, probe)),
                "tfidf_a2_rank": tfidf_a2.get((endpoint, probe)),
                "exposure_class": "high" if block.get("overlap_fraction", 0) >= 0.90
                                  else "lower",
                "interpretation": "EXPLORATORY - unsupervised corpus exposure; not tested",
            })
    emit("supplementary", "chembl_exposure.csv", exposure, exposure[0].keys())

    # --- H1-H6 publication evidence --------------------------------------
    ROLE = {
        "H1": ("SECONDARY", "C10"), "H2": ("PRIMARY", "C1"), "H3": ("NEGATIVE", "C4"),
        "H4": ("SECONDARY", "C10"), "H5": ("ROBUSTNESS", "C5"), "H6": ("CAVEAT", "C8"),
    }
    WORDING = {
        "H1": ("Which representation ranked highest depended on the probe in both tracks.",
               "TF-IDF wins linear and descriptors win nonlinear"),
        "H2": ("Physicochemical descriptors retained the best nonlinear mean rank "
               "under repartitioning.", "descriptors are universally best"),
        "H3": ("The TF-IDF regression advantage did not reproduce under "
               "repartitioning.", "TF-IDF fails at regression"),
        "H4": ("The direction of nonlinear gain was preserved for six of seven "
               "representations.", "the gain ordering is identical"),
        "H5": ("Nine of eleven Holm-significant contrasts reproduced, with effect "
               "direction preserved throughout.", "all contrasts replicated"),
        "H6": ("Cross-endpoint rankings were unchanged under MolFusion's stricter "
               "cleaning.", "cleaning does not matter"),
    }
    evidence = []
    for row in hypotheses:
        hid = row["hypothesis"]
        role, claim = ROLE[hid]
        recommended, prohibited = WORDING[hid]
        evidence.append({
            "hypothesis": hid,
            "question": row["claim"],
            "a1_finding": row["a1"],
            "a2_finding": row["a2"],
            "a2_repartitioned_only": row.get("a2_repartitioned_only", ""),
            "verdict": row["verdict"],
            "detail": row["detail"],
            "publication_role": role,
            "linked_claim_id": claim,
            "recommended_wording": recommended,
            "prohibited_overclaim": prohibited,
        })
    emit("evidence", "h1_h6_publication_evidence.csv", evidence, evidence[0].keys())

    # --- claim registry ---------------------------------------------------
    registry = publication_claims.build_registry(
        a1_summary=a1_ranks, a2_summary=a2_ranks, hypotheses=hypotheses,
        contrasts=contrasts, separation=separation, cost_rows=table5,
        stability_rows=stability, cleaning_rows=cleaning,
        exposure_rows=exposure, gain_comparison=gain_comparison,
        ranking_rows=table3)
    problems = publication_claims.validate_registry(registry)
    if problems:
        raise analysis.AnalysisError(f"claim registry invalid: {problems}")
    emit("evidence", "claim_registry.csv", registry, publication_claims.REGISTRY_COLUMNS)

    ci_rows = [c for probe in protocol.PROBES for c in separation[probe]["comparisons"]]
    emit("evidence", "ci_separation_check.csv", ci_rows, ci_rows[0].keys())

    # --- supplementary ----------------------------------------------------
    emit("supplementary", "all_pairwise_contrasts_a2.csv", a2_tests, a2_tests[0].keys())
    emit("supplementary", "all_pairwise_contrasts_a1.csv", a1_tests, a1_tests[0].keys())
    emit("supplementary", "a1_endpoint_ranks.csv", a1_endpoint_ranks,
         a1_endpoint_ranks[0].keys())
    emit("supplementary", "a2_endpoint_ranks.csv", a2_endpoint_ranks,
         a2_endpoint_ranks[0].keys())
    emit("supplementary", "a1_vs_a2_contrasts.csv", contrasts, contrasts[0].keys())
    emit("supplementary", "split_stability_detail.csv", split_stability,
         split_stability[0].keys())
    emit("supplementary", "cleaning_effects.csv", cleaning, cleaning[0].keys())
    emit("supplementary", "split_distinctness.csv", distinctness, distinctness[0].keys())
    emit("supplementary", "friedman_a1.csv", a1_friedman, a1_friedman[0].keys())
    emit("supplementary", "friedman_a2.csv", a2_friedman, a2_friedman[0].keys())

    # --- figures ----------------------------------------------------------
    heat = [{"endpoint": r["endpoint"], "probe": r["probe"],
             "representation": r["representation"], "rank": _f(r["rank"]),
             "task_type": r.get("task_type", ""),
             "endpoint_stability_flag": next(
                 (s["endpoint_stability_flag"] for s in stability
                  if s["endpoint"] == r["endpoint"]), "OK")}
            for r in a2_endpoint_ranks]
    heat.sort(key=lambda r: (r["probe"], r["endpoint"], r["representation"]))
    emit("figure_data", "figure_01_data.csv", heat, heat[0].keys())

    boot_rows = [{"probe": r["probe"], "representation": r["representation"],
                  "mean_rank": _f(r["mean_rank"]),
                  "ci_lower_95": _f(r["ci_lower_95"]),
                  "ci_upper_95": _f(r["ci_upper_95"]),
                  "bootstrap_resamples": r["bootstrap_resamples"],
                  "resampling_unit": r["resampling_unit"]}
                 for r in a2_boot]
    boot_rows.sort(key=lambda r: (r["probe"], r["mean_rank"]))
    emit("figure_data", "figure_02_data.csv", boot_rows, boot_rows[0].keys())

    slope = [r for r in comparison if r["subset"] == "all"]
    slope.sort(key=lambda r: (r["probe"], int(r["a1_position"])))
    emit("figure_data", "figure_03_data.csv", slope, slope[0].keys())
    emit("figure_data", "figure_04_data.csv", table5, table5[0].keys())
    emit("figure_data", "figure_05_data.csv", stability, stability[0].keys())

    endpoints = sorted({r["endpoint"] for r in a2_endpoint_ranks})
    figures = {
        "figure_01_rank_heatmap.svg": publication_figures.figure_rank_heatmap(
            heat, probes=protocol.PROBES, representations=representations,
            endpoints=endpoints, low_stability=low_stability),
        "figure_02_mean_rank_ci.svg": publication_figures.figure_mean_rank_ci(
            boot_rows, probes=protocol.PROBES),
        "figure_03_rank_robustness.svg": publication_figures.figure_rank_slopegraph(
            comparison, probes=protocol.PROBES),
        "figure_04_rank_vs_cost.svg": publication_figures.figure_rank_vs_cost(table5),
        "figure_05_endpoint_stability.svg":
            publication_figures.figure_endpoint_stability(stability),
    }
    for name, svg in figures.items():
        (out / "figures" / name).write_text(svg, encoding="utf-8", newline="\n")
        print(f"  figures/{name}")

    # --- package identity --------------------------------------------------
    identity = publication.publication_identity(
        a1_identity=a1_provenance["scientific_identity"],
        a2_identity=a2_identity,
        a2_analysis_identity=a2_analysis["analysis_identity_sha256"],
        table_checksums=checksums)

    report = {
        "publication_version": publication.PUBLICATION_VERSION,
        "figure_script_version": publication_figures.FIGURE_SCRIPT_VERSION,
        "publication_identity_sha256": identity,
        "inputs": {
            "a1_raw_identity": a1_provenance["scientific_identity"],
            "a2_raw_identity": a2_identity,
            "a1_analysis_identity": publication.A1_ANALYSIS_IDENTITY,
            "a2_analysis_identity": a2_analysis["analysis_identity_sha256"],
        },
        "representation_order": representations,
        "representation_order_rule": ORDER_RULE,
        "ci_separation": {
            "a1": {p: {k: v for k, v in a1_separation[p].items() if k != "comparisons"}
                   for p in protocol.PROBES},
            "a2": {p: {k: v for k, v in separation[p].items() if k != "comparisons"}
                   for p in protocol.PROBES},
        },
        "stability": {
            "pre_registered_low_stability": list(publication.PRE_REGISTERED_LOW_STABILITY),
            "threshold": publication.LOW_STABILITY_W_THRESHOLD,
            "rule_pre_registration_disagreements": disagreements,
        },
        "claim_registry": registry,
        "table_checksums": checksums,
        "provenance": {
            "a1_execution_commits": list(publication.A1_EXECUTION_COMMITS),
            "a2_execution_commits": list(publication.A2_EXECUTION_COMMITS),
            "analysis_commits": list(publication.ANALYSIS_COMMITS),
            "provenance_hardening_commit": publication.PROVENANCE_HARDENING_COMMIT,
            "historical_shard_provenance": (
                "338 of 616 historical shards record no git commit; a pre-6A.5 "
                "worker-local logging defect, fixed in 89335dc, which did NOT "
                "produce these results"),
        },
    }
    with open(out / "publication_report.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")

    print(f"\npublication identity {identity}")
    print(f"tables {len(checksums)}  figures {len(figures)}  claims {len(registry)}")
    for probe in protocol.PROBES:
        s = separation[probe]
        print(f"  {probe:<10} leader {s['leader']:<30} "
              f"CI clear of all competitors: {s['separated_from_all']}")
    if disagreements:
        print(f"  stability rule/pre-registration disagreement: {disagreements}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
