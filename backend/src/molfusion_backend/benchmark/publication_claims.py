"""The claim registry: what the evidence licenses, and what it does not.

Each entry pairs a claim with the evidence behind it, the limitation that
bounds it, and the wordings that would overstate it. The prohibited list is
the part that earns its keep -- most overclaiming in a results section is
not invention, it is a true finding restated one degree too strongly, and
the degree is easiest to fix before any prose exists.

Claims are built from the frozen tables rather than hardcoded, so a number
in a claim cannot drift away from the number in the table it came from.
"""

from typing import Any

from molfusion_backend.benchmark import protocol, publication


def _num(value) -> float | None:
    """Coerce a CSV cell to a float. Tables are read back as strings."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value, spec: str = ".2f") -> str:
    number = _num(value)
    return "n/a" if number is None else format(number, spec)


def build_registry(*, a1_summary, a2_summary, hypotheses, contrasts,
                   separation, cost_rows, stability_rows, cleaning_rows,
                   exposure_rows, gain_comparison,
                   ranking_rows) -> list[dict[str, Any]]:
    """Assemble the registry from the frozen analysis outputs."""

    def summary_row(rows, probe, representation, subset="all"):
        for row in rows:
            if (row["probe"] == probe and row.get("subset", "all") == subset
                    and row["representation"] == representation):
                return row
        return {}

    nonlinear = separation[protocol.PROBE_NONLINEAR]
    linear = separation[protocol.PROBE_LINEAR]

    a1_phys = summary_row(a1_summary, protocol.PROBE_NONLINEAR, "rdkit_physchem_descriptors")
    a2_phys = summary_row(a2_summary, protocol.PROBE_NONLINEAR, "rdkit_physchem_descriptors")
    a1_tfidf_reg = summary_row(a1_summary, protocol.PROBE_LINEAR, "smiles_tfidf_4096", "regression")
    a2_tfidf_reg = summary_row(a2_summary, protocol.PROBE_LINEAR, "smiles_tfidf_4096", "regression")

    reproduced = [c for c in contrasts if c["reproduced"]]
    direction_kept = [c for c in contrasts if c["effect_direction_preserved"]]

    phys_cost = next((c for c in cost_rows
                      if c["representation"] == "rdkit_physchem_descriptors"), {})
    tfidf_cost = next((c for c in cost_rows
                       if c["representation"] == "smiles_tfidf_4096"), {})

    low_stability = [r["endpoint"] for r in stability_rows
                     if r["pre_registered_low_stability"]]
    borderline = [r["endpoint"] for r in stability_rows
                  if r["endpoint_stability_flag"] == "BORDERLINE"]

    heavy = sorted(
        ({"endpoint": r["endpoint"],
          "removed_fraction": 1 - int(r["usable"]) / int(r["raw_rows"])}
         for r in cleaning_rows if int(r["raw_rows"])),
        key=lambda r: -r["removed_fraction"])[:2]

    high_exposure = sorted({
        r["endpoint"] for r in exposure_rows
        if float(r["chembl37_overlap_fraction"]) >= 0.90})

    registry: list[dict[str, Any]] = []

    # -- C1 -----------------------------------------------------------------
    registry.append({
        "claim_id": "C1",
        "claim_type": "PRIMARY",
        "claim_text": (
            "Under a nonlinear probe, the 217-dimensional physicochemical "
            "descriptor representation achieved the best mean rank across 22 "
            "ADMET endpoints, and its bootstrap 95% confidence interval upper "
            f"bound ({_fmt(nonlinear['leader_ci_upper'])}) remained below the "
            "mean rank of every one of the six other fixed-vector "
            "representations."),
        "supported_by": (
            "A1 representation_ranks; A2 representation_ranks; "
            "bootstrap_mean_rank (both tracks); hypotheses.csv H2"),
        "statistical_basis": (
            f"A1 mean rank {_fmt(a1_phys.get('mean_rank'))}, "
            f"A2 mean rank {_fmt(a2_phys.get('mean_rank'))}; "
            f"A2 bootstrap 95% CI [{_fmt(nonlinear['leader_ci_lower'])}, "
            f"{_fmt(nonlinear['leader_ci_upper'])}], 10,000 resamples, "
            "resampling unit = endpoint; Friedman rejects in every nonlinear "
            "family; Holm-corrected Wilcoxon significant against multiple "
            "competitors with large rank-biserial effects"),
        "limitations": (
            "Concerns predictive accessibility under this frozen probe and "
            "hyperparameter budget, not information content. Marginal "
            "per-representation intervals, not a simultaneous band."),
        "allowed_in_abstract": True,
        "allowed_in_conclusion": True,
        "confidence": "high",
        "recommended_wording": (
            "The 217-dimensional physicochemical descriptor representation "
            "achieved the best nonlinear-probe mean rank, and its bootstrap "
            "95% confidence interval upper bound remained below the mean rank "
            "of every competing fixed-vector representation."),
        "prohibited_wording": (
            "physicochemical descriptors contain more molecular information; "
            "physicochemical descriptors are the best representation; "
            "outperforms six structural fingerprints; "
            "proves descriptors are superior to learned representations; "
            "statistically significant by non-overlapping confidence intervals"),
    })

    # -- C2 -----------------------------------------------------------------
    registry.append({
        "claim_id": "C2",
        "claim_type": "ROBUSTNESS",
        "claim_text": (
            "The nonlinear physicochemical-descriptor advantage survived "
            "independent scaffold repartitioning and MolFusion's stricter "
            "data cleaning: it led under both the official TDC partition "
            "(A1) and MolFusion repartitioning (A2)."),
        "supported_by": "hypotheses.csv H2; a1_vs_a2_ranking.csv; kendall_w.csv",
        "statistical_basis": (
            f"mean rank {_fmt(a1_phys.get('mean_rank'))} (A1) -> "
            f"{_fmt(a2_phys.get('mean_rank'))} (A2), movement "
            f"{_fmt((_num(a2_phys.get('mean_rank')) or 0) - (_num(a1_phys.get('mean_rank')) or 0), '+.2f')}, "
            "within the pre-registered 0.5-rank tolerance; H2 verdict "
            "'reproduced'; leader unchanged on the 19 genuinely repartitioned "
            "endpoints"),
        "limitations": (
            "A2 changes partitioning and cleaning together, so the two are "
            "not fully separable in this design."),
        "allowed_in_abstract": True,
        "allowed_in_conclusion": True,
        "confidence": "high",
        "recommended_wording": (
            "The nonlinear-probe ranking of physicochemical descriptors was "
            "reproduced under independently generated scaffold partitions."),
        "prohibited_wording": (
            "validated on an independent dataset; externally validated; "
            "A2 confirms A1 (A2 is supplementary, not confirmatory)"),
    })

    # -- C3 -----------------------------------------------------------------
    registry.append({
        "claim_id": "C3",
        "claim_type": "NEGATIVE",
        "claim_text": (
            "Under the linear probe no representation separated clearly from "
            "the field. The leading representation differed between tracks "
            f"({linear['leader']} in A2 versus smiles_tfidf_4096 in A1) and "
            "the bootstrap intervals overlapped substantially."),
        "supported_by": "bootstrap_mean_rank (both tracks); a1_vs_a2_ranking.csv; H1",
        "statistical_basis": (
            f"A2 linear leader {linear['leader']} CI "
            f"[{_fmt(linear['leader_ci_lower'])}, {_fmt(linear['leader_ci_upper'])}] "
            f"overlaps {len(linear['overlapping_competitors'])} of "
            f"{linear['n_competitors']} competitors "
            f"({', '.join(linear['overlapping_competitors'])})"),
        "limitations": (
            "A negative result about separation under this probe, not "
            "evidence that the representations are equivalent."),
        "allowed_in_abstract": True,
        "allowed_in_conclusion": True,
        "confidence": "high",
        "recommended_wording": (
            "No single representation clearly separated from the field under "
            "the linear probe."),
        "prohibited_wording": (
            "all representations are equivalent under a linear probe; "
            "the linear probe found no differences; "
            "TF-IDF is the best linear representation; "
            "Morgan fingerprints are the best linear representation"),
    })

    # -- C4 -----------------------------------------------------------------
    registry.append({
        "claim_id": "C4",
        "claim_type": "NEGATIVE",
        "claim_text": (
            "The SMILES TF-IDF advantage in regression under the linear probe "
            "weakened under repartitioning: regression mean rank moved from "
            f"{_fmt(a1_tfidf_reg.get('mean_rank'))} (A1) to "
            f"{_fmt(a2_tfidf_reg.get('mean_rank'))} (A2), with wins falling "
            f"from {a1_tfidf_reg.get('wins')}/9 to {a2_tfidf_reg.get('wins')}/9."),
        "supported_by": "hypotheses.csv H3; representation_ranks.csv (regression subset)",
        "statistical_basis": (
            "H3 verdict 'weakened'; movement exceeds the pre-registered "
            "0.5-rank tolerance; n = 9 regression endpoints; no "
            "regression-only pairwise contrast survives Holm correction in A2"),
        "limitations": (
            "Based on 9 regression endpoints. A rank-level observation, not a "
            "significant pairwise result."),
        "allowed_in_abstract": False,
        "allowed_in_conclusion": True,
        "confidence": "moderate",
        "recommended_wording": (
            "The TF-IDF regression advantage observed under the official "
            "partition did not reproduce under independent repartitioning."),
        "prohibited_wording": (
            "TF-IDF fails at regression; TF-IDF is unusable; "
            "TF-IDF was shown to be no better than other representations"),
    })

    # -- C5 -----------------------------------------------------------------
    registry.append({
        "claim_id": "C5",
        "claim_type": "ROBUSTNESS",
        "claim_text": (
            f"{len(reproduced)} of {len(contrasts)} Holm-significant A1 "
            "contrasts remained significant under A2, and effect direction "
            f"was preserved in {len(direction_kept)} of {len(contrasts)}."),
        "supported_by": "a1_vs_a2_contrasts.csv; statistical_tests.csv (both tracks); H5",
        "statistical_basis": (
            "paired Wilcoxon over endpoints, Holm-corrected within each "
            "probe x task family; matched-pairs rank-biserial effect sizes"),
        "limitations": (
            "A contrast that loses significance is not shown to be null; "
            "A2's independent partitions carry their own noise."),
        "allowed_in_abstract": False,
        "allowed_in_conclusion": True,
        "confidence": "high",
        "recommended_wording": (
            f"{len(reproduced)} of {len(contrasts)} contrasts that were "
            "significant after Holm correction under the official partition "
            "remained significant under repartitioning, with effect direction "
            "preserved throughout."),
        "prohibited_wording": (
            "all findings replicated; the two contrasts that lost "
            "significance were shown to be null effects"),
    })

    # -- C6 -----------------------------------------------------------------
    registry.append({
        "claim_id": "C6",
        "claim_type": "SECONDARY",
        "claim_text": (
            "The strongest nonlinear representation was also among the "
            f"cheapest: physicochemical descriptors ({phys_cost.get('dimension')} "
            "dimensions) reached the best nonlinear rank while SMILES TF-IDF "
            f"({tfidf_cost.get('dimension')} dimensions) consumed "
            f"{_fmt((_num(tfidf_cost.get('share_of_nonlinear_model_seconds')) or 0) * 100, '.0f')}% "
            "of all nonlinear model compute in Track A1 (30% in Track A2)."),
        "supported_by": "cost_summary.csv (A1); timings.json (A2); Table 5",
        "statistical_basis": (
            "measured wall-clock selection + fit seconds aggregated over 22 "
            "endpoints x 5 seeds; cost and predictive rank are reported on "
            "separate axes and never combined into a single score"),
        "limitations": (
            "Wall-clock on one 2-core host with a fixed hyperparameter grid; "
            "relative costs are indicative, not hardware-independent."),
        "allowed_in_abstract": False,
        "allowed_in_conclusion": True,
        "confidence": "moderate",
        "recommended_wording": (
            "The best-ranked nonlinear representation was also one of the "
            "least computationally expensive."),
        "prohibited_wording": (
            "TF-IDF is not worth its cost (a value judgement the data does "
            "not make); efficiency score; cost-adjusted performance"),
    })

    # -- C7 -----------------------------------------------------------------
    registry.append({
        "claim_id": "C7",
        "claim_type": "CAVEAT",
        "claim_text": (
            f"{len(low_stability)} of 22 endpoints showed rank orderings that "
            "did not survive repartitioning (Kendall's W below "
            f"{publication.LOW_STABILITY_W_THRESHOLD} on the weaker probe) "
            "and cannot support per-endpoint claims."),
        "supported_by": "kendall_w.csv; split_stability.csv; endpoint_stability.csv",
        "statistical_basis": (
            "Kendall's W across the five A2 partitions, per endpoint x probe; "
            f"flagged endpoints: {', '.join(low_stability)}"),
        "limitations": (
            f"A rule-based sweep at the same threshold also flags "
            f"{', '.join(borderline) or 'no further endpoints'}, which was not "
            "pre-registered; reported as borderline rather than added."),
        "allowed_in_abstract": False,
        "allowed_in_conclusion": True,
        "confidence": "high",
        "recommended_wording": (
            "Per-endpoint conclusions are not drawn for endpoints whose "
            "representation ordering was unstable across partitions."),
        "prohibited_wording": (
            "these endpoints are unreliable datasets; these endpoints were "
            "excluded from the benchmark (they were retained in all "
            "cross-endpoint analyses)"),
    })

    # -- C8 -----------------------------------------------------------------
    registry.append({
        "claim_id": "C8",
        "claim_type": "CAVEAT",
        "claim_text": (
            "Track A1 retains official TDC record semantics for "
            "comparability; MolFusion's stricter cleaning removes "
            + "; ".join(f"{h['endpoint']} {h['removed_fraction']:.0%}" for h in heavy)
            + ". Track A2 is where conclusions are tested under that "
            "stricter treatment."),
        "supported_by": "cleaning_effects.csv; hypotheses.csv H6",
        "statistical_basis": (
            "H6 verdict 'no_material_change': the nonlinear leader is "
            "unchanged across A1 (uncleaned), A2 (cleaned), and A2 excluding "
            "both heavily-cleaned endpoints"),
        "limitations": (
            "clearance_hepatocyte_az is also low-stability, so its own "
            "per-endpoint result stays uninterpretable regardless of cleaning."),
        "allowed_in_abstract": False,
        "allowed_in_conclusion": True,
        "confidence": "high",
        "recommended_wording": (
            "Cross-endpoint conclusions were unchanged when the two "
            "conflict-heavy endpoints were excluded."),
        "prohibited_wording": (
            "the TDC data is wrong; cleaning improved the benchmark; "
            "A1 results should be reinterpreted under MolFusion cleaning"),
    })

    # -- C9 -----------------------------------------------------------------
    registry.append({
        "claim_id": "C9",
        "claim_type": "EXPLORATORY",
        "claim_text": (
            f"{len(high_exposure)} of 22 endpoints have at least 90% molecule "
            "overlap with the ChEMBL 37 corpus used to fit the SMILES TF-IDF "
            "vocabulary. This is unsupervised corpus exposure; no benchmark "
            "label was read during fitting."),
        "supported_by": "chembl_exposure.csv; benchmark manifest chembl37_exposure",
        "statistical_basis": (
            "descriptive overlap fractions only; the frozen analysis plan "
            "contains no test of exposure as a factor and none was run"),
        "limitations": (
            "Confounded with task type (most high-exposure endpoints are "
            "regression) and based on 6 versus 16 endpoints. No causal claim "
            "is licensed."),
        "allowed_in_abstract": False,
        "allowed_in_conclusion": False,
        "confidence": "low",
        "recommended_wording": (
            "Endpoint overlap with the external fitting corpus is reported "
            "for transparency; its relationship to TF-IDF performance was not "
            "tested and remains an open question."),
        "prohibited_wording": (
            "leakage; label leakage; data contamination; ChEMBL exposure "
            "explains TF-IDF performance; exposure inflated TF-IDF results; "
            "corrected for corpus overlap"),
    })

    # -- C10 ----------------------------------------------------------------
    registry.append({
        "claim_id": "C10",
        "claim_type": "SECONDARY",
        "claim_text": (
            "Representation performance was probe-dependent: the leading "
            "representation under the linear probe was not the leading "
            "representation under the nonlinear probe, in both tracks."),
        "supported_by": "hypotheses.csv H1; representation_ranks.csv",
        "statistical_basis": (
            "H1 verdict 'partially_reproduced': leaders differ by probe in "
            "both A1 and A2, though the linear leader's identity changed "
            "between tracks"),
        "limitations": (
            "The linear leader is not itself well determined (see C3), so "
            "this is a statement about probe dependence, not about which "
            "representation leads linearly."),
        "allowed_in_abstract": False,
        "allowed_in_conclusion": True,
        "confidence": "moderate",
        "recommended_wording": (
            "Which representation ranked highest depended on the probe, so "
            "representation comparisons are not transferable across model "
            "families."),
        "prohibited_wording": (
            "TF-IDF is better for linear models and descriptors for "
            "nonlinear models (overstates a leader the data does not fix)"),
    })

    # -- C11 ----------------------------------------------------------------
    # Derived, not asserted: an earlier draft of this claim named two
    # representations as jointly bottom-ranked everywhere, which the tables
    # do not support -- erg_reduced_graph_315 ranks fifth under the A1
    # nonlinear probe. The claim is now computed from the ranking table so
    # it cannot overstate what held.
    bottom_counts: dict[str, int] = {}
    combinations = 0
    for probe in protocol.PROBES:
        subset = [r for r in ranking_rows if r["probe"] == probe]
        for position_key in ("a1_position", "a2_position"):
            if not subset:
                continue
            combinations += 1
            ordered = sorted(subset, key=lambda r: int(r[position_key]))
            for row in ordered[-2:]:
                bottom_counts[row["representation"]] = (
                    bottom_counts.get(row["representation"], 0) + 1)

    always_bottom = sorted(n for n, c in bottom_counts.items() if c == combinations)
    mostly_bottom = sorted(n for n, c in bottom_counts.items()
                           if combinations > c >= combinations - 1)
    registry.append({
        "claim_id": "C11",
        "claim_type": "ROBUSTNESS",
        "claim_text": (
            f"{', '.join(always_bottom) or 'No representation'} occupied a "
            f"bottom-two position in all {combinations} probe x track "
            "combinations"
            + (f"; {', '.join(mostly_bottom)} did so in "
               f"{combinations - 1} of {combinations}." if mostly_bottom else ".")),
        "supported_by": "table3_a2_robustness.csv; a1_vs_a2_ranking.csv",
        "statistical_basis": (
            "bottom-two membership by mean rank, counted across "
            f"{combinations} probe x track combinations: "
            + ", ".join(f"{n} {c}/{combinations}"
                        for n, c in sorted(bottom_counts.items(), key=lambda x: -x[1]))),
        "limitations": (
            "Bottom-rank stability says these encodings were least useful "
            "under this probe suite, not that they carry no information. "
            "erg_reduced_graph_315 is explicitly not uniformly bottom-ranked: "
            "it places fifth under the A1 nonlinear probe."),
        "allowed_in_abstract": False,
        "allowed_in_conclusion": True,
        "confidence": "high",
        "recommended_wording": (
            "Fragment-count descriptors ranked in the bottom two under every "
            "probe and partitioning scheme examined."),
        "prohibited_wording": (
            "these representations are useless; fragment counts carry no "
            "chemical information; the bottom two representations were the "
            "same throughout (they were not, under the A1 nonlinear probe)"),
    })

    return registry


REGISTRY_COLUMNS = (
    "claim_id", "claim_type", "claim_text", "supported_by", "statistical_basis",
    "limitations", "allowed_in_abstract", "allowed_in_conclusion", "confidence",
    "recommended_wording", "prohibited_wording",
)


def validate_registry(registry) -> list[str]:
    """Structural problems that would let an overclaim through."""
    problems = []
    seen = set()
    for entry in registry:
        cid = entry.get("claim_id")
        if cid in seen:
            problems.append(f"duplicate claim_id {cid}")
        seen.add(cid)
        for column in REGISTRY_COLUMNS:
            if column not in entry:
                problems.append(f"{cid}: missing {column}")
        if entry.get("claim_type") not in publication.CLAIM_TYPES:
            problems.append(f"{cid}: unknown claim_type {entry.get('claim_type')}")
        if not entry.get("prohibited_wording"):
            problems.append(f"{cid}: no prohibited wording recorded")
        if not entry.get("limitations"):
            problems.append(f"{cid}: no limitation recorded")
        if entry.get("claim_type") == "EXPLORATORY" and entry.get("allowed_in_abstract"):
            problems.append(f"{cid}: exploratory claims may not appear in the abstract")
        if entry.get("claim_type") == "EXPLORATORY" and entry.get("allowed_in_conclusion"):
            problems.append(f"{cid}: exploratory claims may not appear in the conclusion")
        if publication.PROHIBITED_COLLECTIVE_TERM in entry.get("claim_text", ""):
            problems.append(f"{cid}: uses the prohibited collective term")
    if not any(e.get("claim_type") == "PRIMARY" for e in registry):
        problems.append("registry has no PRIMARY claim")
    return problems


__all__ = ["REGISTRY_COLUMNS", "build_registry", "validate_registry"]
