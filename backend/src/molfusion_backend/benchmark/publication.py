"""Phase 6B: turn the frozen benchmark results into a publication package.

This module computes nothing scientific. Every number it emits already
exists in the Phase 6A.3 and 6A.4 analysis outputs; what is added here is
*organisation* -- which findings are strong enough to lead with, which are
supporting, which are negative, and which must never leave the
supplementary material.

That framing is the point. The risk at this stage is not a wrong
calculation, it is a true number stated more strongly than it was earned.
So the claim registry carries, for every claim, the evidence that supports
it, the limitation that bounds it, and an explicit list of wordings that
the evidence does *not* license. A later drafting phase can then be
checked against the registry rather than against memory.

Two conventions worth stating once.

**"Six other fixed-vector representations", never "structural
fingerprints".** The Track A competitors are a mixed set -- circular and
substructure-key fingerprints, physicochemical and fragment-count
descriptors, a reduced-graph encoding and a SMILES n-gram TF-IDF. Calling
them all fingerprints would be inaccurate.

**Predictive accessibility, not information content.** Physicochemical
descriptors leading the nonlinear probe means those 217 numbers put more
signal within reach of this frozen probe than 4096 TF-IDF dimensions did.
It does not mean they *contain* more chemistry.
"""

import hashlib
from collections.abc import Sequence
from typing import Any

from molfusion_backend.benchmark import protocol

PUBLICATION_VERSION = "6B.1"

#: Frozen inputs. Phase 6B reads these and never recomputes them.
A1_RAW_IDENTITY = "d40ef09b398f47914aa51f99fd6a4f5893f7778b50c0cca04404b575632de868"
A2_RAW_IDENTITY = "9dd5dfa6067c8a760b0bb8fb39648f71f662f2fa1bbf4cc5d7cb0cd495a69f14"
A2_ANALYSIS_IDENTITY = "bda6bd23db77c08a49f8529db609dbc02ecc2136982b8153c6a57cb60c100217"
A1_ANALYSIS_IDENTITY = "2279307bdb30dfe26456e3015b3f4788c522864e46e841a7450ed466ab2d4b76"

#: Execution and analysis history, kept distinct in the reproducibility
#: statement because the hardened provenance code did NOT produce these
#: results -- it was written afterwards, in Phase 6A.5.
A1_EXECUTION_COMMITS = ("459653b", "ddabb42", "2bcb467")
A2_EXECUTION_COMMITS = ("e6ae297",)
ANALYSIS_COMMITS = ("fe4bc60", "15b78a2")
PROVENANCE_HARDENING_COMMIT = "89335dc"

#: Accurate per-representation categories. Deliberately not one label.
REPRESENTATION_CATEGORY = {
    "morgan_ecfp4_1024": "circular fingerprint",
    "avalon_1024": "substructure fingerprint",
    "maccs_keys_167": "substructure key fingerprint",
    "rdkit_physchem_descriptors": "physicochemical descriptors",
    "rdkit_fragment_descriptors": "fragment counts",
    "erg_reduced_graph_315": "reduced-graph features",
    "smiles_tfidf_4096": "SMILES n-gram TF-IDF",
}

#: Collective noun for the Track A set. Used verbatim in claim wording.
COLLECTIVE_TERM = "fixed-vector representations"
PROHIBITED_COLLECTIVE_TERM = "structural fingerprints"

#: Endpoints whose representation ordering does not survive repartitioning.
#: Pre-registered in the Phase 6B brief; the rule below is what selects
#: them, and any disagreement is reported rather than silently resolved.
PRE_REGISTERED_LOW_STABILITY = (
    "herg",
    "cyp2c9_substrate_carbonmangels",
    "clearance_hepatocyte_az",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "bioavailability_ma",
)
LOW_STABILITY_W_THRESHOLD = 0.35

#: Cleaning removes an unusually large fraction of these endpoints, so A1
#: (official rows as shipped) and A2 (fully cleaned) differ most here.
CONFLICT_SENSITIVE_ENDPOINTS = ("ppbr_az", "clearance_hepatocyte_az")

CLAIM_TYPES = (
    "PRIMARY", "ROBUSTNESS", "SECONDARY", "NEGATIVE", "CAVEAT", "EXPLORATORY",
)

STABILITY_RECOMMENDED = "per-endpoint interpretation: RECOMMENDED"
STABILITY_NOT_RECOMMENDED = "per-endpoint interpretation: NOT RECOMMENDED"


# ---------------------------------------------------------------------------
# stability classification
# ---------------------------------------------------------------------------


def stability_table(kendall_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-endpoint rank stability, with the flag publication tables carry.

    An endpoint qualifies as low-stability when its *weaker* probe falls
    below the threshold: a representation ordering that dissolves under
    either probe cannot support a per-endpoint claim, and taking the
    minimum avoids one strong probe masking the other.
    """
    by_endpoint: dict[str, dict[str, float]] = {}
    for row in kendall_rows:
        by_endpoint.setdefault(row["endpoint"], {})[row["probe"]] = float(row["kendall_w"])

    out = []
    for endpoint, per_probe in sorted(by_endpoint.items()):
        weakest = min(per_probe.values())
        rule_flag = weakest < LOW_STABILITY_W_THRESHOLD
        pre_registered = endpoint in PRE_REGISTERED_LOW_STABILITY
        out.append({
            "endpoint": endpoint,
            "kendall_w_linear": per_probe.get(protocol.PROBE_LINEAR),
            "kendall_w_nonlinear": per_probe.get(protocol.PROBE_NONLINEAR),
            "kendall_w_min": weakest,
            "rule_low_stability": rule_flag,
            "pre_registered_low_stability": pre_registered,
            "endpoint_stability_flag": (
                "LOW" if pre_registered else "BORDERLINE" if rule_flag else "OK"),
            "per_endpoint_interpretation": (
                STABILITY_NOT_RECOMMENDED if pre_registered else STABILITY_RECOMMENDED),
            "agrees_with_pre_registration": rule_flag == pre_registered,
        })
    return out


def stability_disagreements(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Endpoints the rule flags that pre-registration did not, or vice versa.

    Reported, never auto-resolved. Quietly widening a pre-registered
    exclusion after seeing the data is how a caveat list becomes a
    post-hoc filter.
    """
    return sorted(r["endpoint"] for r in rows if not r["agrees_with_pre_registration"])


# ---------------------------------------------------------------------------
# confidence-interval separation
# ---------------------------------------------------------------------------


def ci_separation(bootstrap_rows: Sequence[dict[str, Any]], *, probe: str) -> dict[str, Any]:
    """Is the leader's bootstrap CI clear of every competitor's?

    The Phase 6B brief forbids asserting separation without checking it, so
    this returns the comparison rather than an assumption. Note what the
    answer is and is not: these are marginal per-representation intervals,
    not a simultaneous band, so non-overlap is supporting evidence for a
    difference and not a test of one. The test is the Holm-corrected
    Wilcoxon.
    """
    rows = [r for r in bootstrap_rows if r["probe"] == probe]
    ordered = sorted(rows, key=lambda r: float(r["mean_rank"]))
    leader = ordered[0]
    upper = float(leader["ci_upper_95"])

    comparisons = []
    for row in ordered[1:]:
        lower = float(row["ci_lower_95"])
        comparisons.append({
            "probe": probe,
            "leader": leader["representation"],
            "competitor": row["representation"],
            "leader_ci_upper": upper,
            "competitor_ci_lower": lower,
            "competitor_mean_rank": float(row["mean_rank"]),
            "separated": lower > upper,
            "leader_upper_below_competitor_mean": upper < float(row["mean_rank"]),
        })

    return {
        "probe": probe,
        "leader": leader["representation"],
        "leader_mean_rank": float(leader["mean_rank"]),
        "leader_ci_lower": float(leader["ci_lower_95"]),
        "leader_ci_upper": upper,
        "n_competitors": len(comparisons),
        "separated_from_all": all(c["separated"] for c in comparisons),
        "below_every_competitor_mean": all(
            c["leader_upper_below_competitor_mean"] for c in comparisons),
        "overlapping_competitors": [
            c["competitor"] for c in comparisons if not c["separated"]],
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def table_checksum(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    """Content digest of an emitted table, independent of file layout."""
    digest = hashlib.sha256()
    digest.update(f"publication_table_v1\x1frows={len(rows)}".encode())
    for row in rows:
        fields = [repr(row.get(c)) if isinstance(row.get(c), float) else str(row.get(c))
                  for c in columns]
        digest.update(("\x1e" + "\x1f".join(fields)).encode("utf-8"))
    return digest.hexdigest()


def publication_identity(*, a1_identity: str, a2_identity: str,
                         a2_analysis_identity: str,
                         table_checksums: dict[str, str]) -> str:
    """Deterministic identity for the whole evidence package.

    Covers the inputs it was derived from and the content of every table it
    produced. Excludes timestamps, absolute paths and machine metadata, so
    two machines running the same code over the same frozen results agree.
    """
    payload = "\x1f".join((
        f"publication_identity_v1:{PUBLICATION_VERSION}",
        f"a1={a1_identity}",
        f"a2={a2_identity}",
        f"a2_analysis={a2_analysis_identity}",
        *(f"{name}={table_checksums[name]}" for name in sorted(table_checksums)),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "A1_ANALYSIS_IDENTITY",
    "A1_EXECUTION_COMMITS",
    "A1_RAW_IDENTITY",
    "A2_ANALYSIS_IDENTITY",
    "A2_EXECUTION_COMMITS",
    "A2_RAW_IDENTITY",
    "ANALYSIS_COMMITS",
    "CLAIM_TYPES",
    "COLLECTIVE_TERM",
    "CONFLICT_SENSITIVE_ENDPOINTS",
    "LOW_STABILITY_W_THRESHOLD",
    "PRE_REGISTERED_LOW_STABILITY",
    "PROHIBITED_COLLECTIVE_TERM",
    "PROVENANCE_HARDENING_COMMIT",
    "PUBLICATION_VERSION",
    "REPRESENTATION_CATEGORY",
    "ci_separation",
    "publication_identity",
    "stability_disagreements",
    "stability_table",
    "table_checksum",
]
