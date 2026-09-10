"""Produce the Track A2 derived analysis, and evaluate H1-H6 against A1.

Reads both completed matrices read-only. A1's artifacts are verified against
their frozen identities and never written to. Run from the repository root:

    .\\backend\\.venv\\Scripts\\python.exe -m molfusion_backend.benchmark.analysis_a2_cli
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from molfusion_backend.benchmark import analysis, analysis_a2, protocol

A1_DIR = Path("backend/benchmark_runs/track_a1")
A2_DIR = Path("backend/benchmark_runs/track_a2")
OUT_DIR = Path("backend/benchmark_runs/track_a2/analysis")
MANIFEST = Path("backend/benchmark_manifests/tdc_admet_group.json")

A1_IDENTITY = "d40ef09b398f47914aa51f99fd6a4f5893f7778b50c0cca04404b575632de868"
A1_ANALYSIS_IDENTITY = "2279307bdb30dfe26456e3015b3f4788c522864e46e841a7450ed466ab2d4b76"
EXPECTED_ROWS = 6160

HIGH_REPLICATE_ENDPOINTS = ("ppbr_az", "clearance_hepatocyte_az")

#: Mean-rank movement below this counts as "reproduced". Fixed before the
#: A2 numbers were looked at; roughly half a rank position on a 7-way scale.
RANK_TOLERANCE = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-dir", type=Path, default=A1_DIR)
    parser.add_argument("--a2-dir", type=Path, default=A2_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text("utf-8"))

    # --- A1: verify and re-derive, never write -------------------------
    a1_path = args.a1_dir / "results_track_a1.csv"
    a1_before = a1_path.stat()
    a1_rows = analysis.load_raw_results(a1_path)
    a1_provenance = analysis.verify_raw_results(
        a1_rows, expected_identity=A1_IDENTITY, expected_rows=EXPECTED_ROWS)
    print(f"A1 identity verified: {a1_provenance['scientific_identity']}")

    # --- A2 ------------------------------------------------------------
    a2_path = args.a2_dir / "results_track_a2.csv"
    a2_rows = analysis.load_raw_results(a2_path)
    a2_report = json.loads((args.a2_dir / "run_report.json").read_text("utf-8"))
    print(f"A2 identity: {a2_report['scientific_identity_sha256']}  rows={len(a2_rows)}")

    distinctness = a2_report["split_distinctness"]
    repartitioned = sorted(
        e for e, d in distinctness.items()
        if d["mean_pairwise_test_jaccard"] <= protocol.A2_PARTITION_VARIABILITY_ALERT
    )
    low_variability = sorted(set(distinctness) - set(repartitioned))
    print(f"genuinely repartitioned endpoints: {len(repartitioned)}/22"
          f"  low-variability: {low_variability}")

    checksums: dict[str, str] = {}

    def emit(name: str, table: list[dict[str, Any]]) -> None:
        checksums[name] = analysis.write_table(out / name, table)
        print(f"  wrote {name:<40} {len(table):>5} rows")

    # --- core A2 tables -------------------------------------------------
    a1_scores = analysis.aggregate_seeds(a1_rows)
    a1_ranks = analysis.rank_table(a1_scores)
    a2_scores = analysis.aggregate_seeds(a2_rows)
    a2_ranks = analysis.rank_table(a2_scores)

    emit("endpoint_summary.csv", [s.as_row() for s in a2_scores])
    emit("endpoint_ranks.csv", [
        {"endpoint": e, "probe": p, "representation": r, "rank": v,
         "task_type": next(s.task_type for s in a2_scores if s.endpoint == e)}
        for (e, p), row in sorted(a2_ranks.items()) for r, v in sorted(row.items())])

    classification = sorted({s.endpoint for s in a2_scores
                             if s.task_type == protocol.TASK_CLASSIFICATION})
    regression = sorted({s.endpoint for s in a2_scores
                         if s.task_type == protocol.TASK_REGRESSION})
    all_endpoints = sorted({s.endpoint for s in a2_scores})
    without_flagged = [e for e in all_endpoints if e not in HIGH_REPLICATE_ENDPOINTS]

    subsets = {
        "all": all_endpoints,
        "classification": classification,
        "regression": regression,
        "excl_high_replicate": without_flagged,
        "repartitioned_only": repartitioned,
    }
    a2_summary = [
        {"subset": name, **row}
        for probe in protocol.PROBES
        for name, endpoints in subsets.items()
        for row in analysis.summarise_ranks(a2_ranks, probe=probe, endpoints=endpoints)
    ]
    emit("representation_ranks.csv", a2_summary)

    a1_summary = [
        {"subset": name, **row}
        for probe in protocol.PROBES
        for name, endpoints in (("all", all_endpoints), ("classification", classification),
                                ("regression", regression),
                                ("excl_high_replicate", without_flagged),
                                ("repartitioned_only", repartitioned))
        for row in analysis.summarise_ranks(a1_ranks, probe=probe, endpoints=endpoints)
    ]

    pairwise = [row for probe in protocol.PROBES
                for task in (None, protocol.TASK_CLASSIFICATION, protocol.TASK_REGRESSION)
                for row in analysis.pairwise_wins(a2_scores, probe=probe, task_type=task)]
    emit("pairwise_wins.csv", pairwise)

    gains = analysis.nonlinear_gain(a2_scores)
    emit("nonlinear_gain.csv", gains)
    by_rep = defaultdict(list)
    for g in gains:
        by_rep[g["representation"]].append(g)
    a2_gain_summary = sorted(
        ({"representation": r, "n_endpoints": len(v),
          "mean_normalised_gain": float(np.mean([g["normalised_gain"] for g in v])),
          "median_normalised_gain": float(np.median([g["normalised_gain"] for g in v])),
          "mean_rank_gain": float(np.mean([g["rank_gain"] for g in v])),
          "endpoints_improved": int(sum(1 for g in v if g["normalised_gain"] > 0))}
         for r, v in by_rep.items()),
        key=lambda r: -r["mean_normalised_gain"])
    emit("nonlinear_gain_summary.csv", a2_gain_summary)

    # --- stability, the question A1 could not ask -----------------------
    emit("split_stability.csv", analysis_a2.split_stability(a2_rows))
    kendall = [row for probe in protocol.PROBES
               for row in analysis_a2.kendall_w(a2_rows, probe=probe)]
    emit("kendall_w.csv", kendall)

    # --- statistics ------------------------------------------------------
    omnibus, a2_tests = [], []
    for probe in protocol.PROBES:
        for task in (None, protocol.TASK_CLASSIFICATION, protocol.TASK_REGRESSION):
            result = analysis.friedman(a2_scores, probe=probe, task_type=task)
            omnibus.append(result)
            if result.get("runnable") and result.get("reject_at_alpha"):
                a2_tests += analysis.pairwise_tests(a2_scores, probe=probe, task_type=task)
    emit("friedman.csv", omnibus)
    emit("statistical_tests.csv", a2_tests)

    bootstrap = [row for probe in protocol.PROBES
                 for row in analysis.bootstrap_mean_rank(a2_ranks, probe=probe)]
    emit("bootstrap_mean_rank.csv", bootstrap)

    # --- A1 vs A2 --------------------------------------------------------
    comparison = [row for probe in protocol.PROBES
                  for subset in ("all", "classification", "regression", "repartitioned_only")
                  for row in analysis_a2.compare_rankings(
                      a1_summary, a2_summary, probe=probe, subset=subset)]
    emit("a1_vs_a2_ranking.csv", comparison)

    a1_tests = _load(args.a1_dir / "analysis" / "statistical_tests.csv")
    contrasts = analysis_a2.reproduced_contrasts(a1_tests, a2_tests)
    emit("a1_vs_a2_contrasts.csv", contrasts)

    a1_gain = {r["representation"]: r for r in
               _load(args.a1_dir / "analysis" / "nonlinear_gain_summary.csv")}
    gain_comparison = [
        {"representation": r["representation"],
         "a1_mean_normalised_gain": float(a1_gain[r["representation"]]["mean_normalised_gain"]),
         "a2_mean_normalised_gain": r["mean_normalised_gain"],
         "a1_mean_rank_gain": float(a1_gain[r["representation"]]["mean_rank_gain"]),
         "a2_mean_rank_gain": r["mean_rank_gain"],
         "sign_preserved": bool(
             np.sign(float(a1_gain[r["representation"]]["mean_normalised_gain"]))
             == np.sign(r["mean_normalised_gain"]))}
        for r in a2_gain_summary if r["representation"] in a1_gain]
    emit("a1_vs_a2_nonlinear_gain.csv", gain_comparison)

    # --- hypotheses -------------------------------------------------------
    hypotheses = evaluate_hypotheses(
        a1_summary=a1_summary, a2_summary=a2_summary,
        contrasts=contrasts, gain_comparison=gain_comparison,
        repartitioned=repartitioned)
    emit("hypotheses.csv", hypotheses)

    # --- cleaning effects --------------------------------------------------
    cleaning = [{"endpoint": e, **v} for e, v in sorted(a2_report["cleaning"].items())]
    emit("cleaning_effects.csv", cleaning)
    emit("split_distinctness.csv", [
        {"endpoint": e, "mean_pairwise_test_jaccard": d["mean_pairwise_test_jaccard"],
         "max_pairwise_test_jaccard": d["max_pairwise_test_jaccard"],
         "distinct_test_sets": d["distinct_test_sets"],
         "genuinely_repartitioned": e in repartitioned}
        for e, d in sorted(distinctness.items())])

    # --- immutability + identity -------------------------------------------
    a1_after = a1_path.stat()
    if (a1_before.st_size, a1_before.st_mtime) != (a1_after.st_size, a1_after.st_mtime):
        raise analysis.AnalysisError("A1 raw results changed during A2 analysis")

    configuration = {
        "analysis_version": analysis_a2.ANALYSIS_VERSION,
        "seed_aggregation": analysis.SEED_AGGREGATION,
        "statistical_unit": "endpoint",
        "stability_unit": "endpoint x split (descriptive only, never fed to the omnibus)",
        "rank_tolerance": RANK_TOLERANCE,
        "correction": protocol.MULTIPLE_COMPARISON_CORRECTION,
        "effect_size": protocol.EFFECT_SIZE,
        "alpha": protocol.ALPHA,
        "bootstrap_resamples": protocol.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": analysis.BOOTSTRAP_SEED,
        "partition_variability_alert": protocol.A2_PARTITION_VARIABILITY_ALERT,
        "repartitioned_endpoints": repartitioned,
        "low_variability_endpoints": low_variability,
    }
    identity = analysis.analysis_identity(
        raw_identity=a2_report["scientific_identity_sha256"], configuration=configuration)

    report = {
        "analysis_version": analysis_a2.ANALYSIS_VERSION,
        "analysis_identity_sha256": identity,
        "a1_raw_identity": a1_provenance["scientific_identity"],
        "a1_analysis_identity_expected": A1_ANALYSIS_IDENTITY,
        "a1_raw_results_immutable": True,
        "a2_raw_identity": a2_report["scientific_identity_sha256"],
        "configuration": configuration,
        "table_checksums": checksums,
        "hypotheses": hypotheses,
    }
    with open(out / "analysis_report.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")

    print(f"\nA2 analysis identity: {identity}")
    print(f"A1 raw results unchanged: True")
    for row in hypotheses:
        print(f"  {row['hypothesis']:<4} {row['verdict']:<14} {row['detail'][:100]}")
    return 0


def _load(path: Path) -> list[dict[str, Any]]:
    import csv

    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        rows = [dict(r) for r in csv.DictReader(handle)]
    for row in rows:
        for key, value in list(row.items()):
            if key in ("p_raw", "p_holm", "effect_size_rank_biserial", "median_difference",
                       "wilcoxon_statistic", "mean_rank", "mean_normalised_gain",
                       "mean_rank_gain", "median_normalised_gain"):
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = float("nan")
            if key == "significant_after_holm":
                row[key] = value == "True"
    return rows


def evaluate_hypotheses(*, a1_summary, a2_summary, contrasts, gain_comparison,
                        repartitioned) -> list[dict[str, Any]]:
    """Evaluate the six pre-registered hypotheses. Labels fixed in advance."""
    out = []

    # H1 -- the probe-dependent reversal
    a1_linear = analysis_a2.leader(a1_summary, probe=protocol.PROBE_LINEAR)
    a2_linear = analysis_a2.leader(a2_summary, probe=protocol.PROBE_LINEAR)
    a1_nonlinear = analysis_a2.leader(a1_summary, probe=protocol.PROBE_NONLINEAR)
    a2_nonlinear = analysis_a2.leader(a2_summary, probe=protocol.PROBE_NONLINEAR)
    a2_linear_sub = analysis_a2.leader(a2_summary, probe=protocol.PROBE_LINEAR,
                                       subset="repartitioned_only")
    a2_nonlinear_sub = analysis_a2.leader(a2_summary, probe=protocol.PROBE_NONLINEAR,
                                          subset="repartitioned_only")
    reversal_a1 = a1_linear != a1_nonlinear
    reversal_a2 = a2_linear != a2_nonlinear
    out.append({
        "hypothesis": "H1",
        "claim": "the probe-dependent reversal persists under repartitioning",
        "a1": f"linear={a1_linear}, nonlinear={a1_nonlinear}",
        "a2": f"linear={a2_linear}, nonlinear={a2_nonlinear}",
        "a2_repartitioned_only": f"linear={a2_linear_sub}, nonlinear={a2_nonlinear_sub}",
        "verdict": ("reproduced" if (reversal_a1 and reversal_a2
                                     and a2_linear == a1_linear and a2_nonlinear == a1_nonlinear)
                    else "partially_reproduced" if reversal_a2
                    else "contradicted"),
        "detail": (f"A1 leaders differ by probe: {reversal_a1}. "
                   f"A2 leaders differ by probe: {reversal_a2}."),
    })

    # H2 -- physchem nonlinear consistency
    def row_for(summary, probe, representation, subset="all"):
        for r in summary:
            if (r["probe"] == probe and r.get("subset", "all") == subset
                    and r["representation"] == representation):
                return r
        return {}

    a1_phys = row_for(a1_summary, protocol.PROBE_NONLINEAR, "rdkit_physchem_descriptors")
    a2_phys = row_for(a2_summary, protocol.PROBE_NONLINEAR, "rdkit_physchem_descriptors")
    out.append({
        "hypothesis": "H2",
        "claim": "rdkit_physchem_descriptors retains its strong nonlinear ranking",
        "a1": f"mean_rank={a1_phys.get('mean_rank'):.2f}, top3={a1_phys.get('top3')}/22",
        "a2": f"mean_rank={a2_phys.get('mean_rank'):.2f}, top3={a2_phys.get('top3')}/22",
        "a2_repartitioned_only": "",
        "verdict": analysis_a2.classify(a1_phys.get("mean_rank", 0.0),
                                        a2_phys.get("mean_rank", 0.0),
                                        tolerance=RANK_TOLERANCE),
        "detail": f"mean rank moved {a2_phys.get('mean_rank', 0) - a1_phys.get('mean_rank', 0):+.2f}",
    })

    # H3 -- TF-IDF regression effect under the linear probe
    a1_tfidf = row_for(a1_summary, protocol.PROBE_LINEAR, "smiles_tfidf_4096", "regression")
    a2_tfidf = row_for(a2_summary, protocol.PROBE_LINEAR, "smiles_tfidf_4096", "regression")
    out.append({
        "hypothesis": "H3",
        "claim": "the TF-IDF linear advantage stays concentrated in regression",
        "a1": f"mean_rank={a1_tfidf.get('mean_rank'):.2f}, wins={a1_tfidf.get('wins')}/9, "
              f"top3={a1_tfidf.get('top3')}/9",
        "a2": f"mean_rank={a2_tfidf.get('mean_rank'):.2f}, wins={a2_tfidf.get('wins')}/9, "
              f"top3={a2_tfidf.get('top3')}/9",
        "a2_repartitioned_only": "",
        "verdict": analysis_a2.classify(a1_tfidf.get("mean_rank", 0.0),
                                        a2_tfidf.get("mean_rank", 0.0),
                                        tolerance=RANK_TOLERANCE),
        "detail": f"regression mean rank moved "
                  f"{a2_tfidf.get('mean_rank', 0) - a1_tfidf.get('mean_rank', 0):+.2f}",
    })

    # H4 -- probe x representation interaction
    preserved = [g for g in gain_comparison if g["sign_preserved"]]
    out.append({
        "hypothesis": "H4",
        "claim": "the nonlinear-gain ordering is stable",
        "a1": ", ".join(f"{g['representation']}={g['a1_mean_normalised_gain']:+.3f}"
                        for g in sorted(gain_comparison,
                                        key=lambda x: -x["a1_mean_normalised_gain"])[:3]),
        "a2": ", ".join(f"{g['representation']}={g['a2_mean_normalised_gain']:+.3f}"
                        for g in sorted(gain_comparison,
                                        key=lambda x: -x["a2_mean_normalised_gain"])[:3]),
        "a2_repartitioned_only": "",
        "verdict": "reproduced" if len(preserved) >= 6 else
                   "partially_reproduced" if len(preserved) >= 4 else "contradicted",
        "detail": f"gain sign preserved for {len(preserved)}/7 representations",
    })

    # H5 -- statistical reproducibility
    tested = [c for c in contrasts if c["a2_tested"]]
    reproduced = [c for c in contrasts if c["reproduced"]]
    out.append({
        "hypothesis": "H5",
        "claim": "A1 significant contrasts reproduce under A2",
        "a1": f"{len(contrasts)} contrasts significant after Holm",
        "a2": f"{len(reproduced)} reproduced, {len(tested) - len(reproduced)} not, "
              f"{len(contrasts) - len(tested)} untested (A2 omnibus did not reject)",
        "a2_repartitioned_only": "",
        "verdict": ("reproduced" if contrasts and len(reproduced) == len(contrasts)
                    else "partially_reproduced" if reproduced else "not_reproduced"),
        "detail": ("failing to reject is not evidence of no difference; "
                   "A2's independent partitions carry their own noise"),
    })

    # H6 -- cleaning sensitivity
    a1_all = analysis_a2.leader(a1_summary, probe=protocol.PROBE_NONLINEAR)
    a2_all = analysis_a2.leader(a2_summary, probe=protocol.PROBE_NONLINEAR)
    a2_excl = analysis_a2.leader(a2_summary, probe=protocol.PROBE_NONLINEAR,
                                 subset="excl_high_replicate")
    out.append({
        "hypothesis": "H6",
        "claim": "heavy cleaning of ppbr_az and clearance_hepatocyte_az changes rankings",
        "a1": f"nonlinear leader (uncleaned official rows) = {a1_all}",
        "a2": f"nonlinear leader (fully cleaned) = {a2_all}",
        "a2_repartitioned_only": f"excluding the two heavily-cleaned endpoints = {a2_excl}",
        "verdict": "no_material_change" if a1_all == a2_all == a2_excl else "changed",
        "detail": "compares the uncleaned A1 ranking with the fully cleaned A2 ranking",
    })
    return out


if __name__ == "__main__":
    sys.exit(main())
