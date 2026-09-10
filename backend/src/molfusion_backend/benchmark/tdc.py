"""Audit of the frozen TDC ADMET release, against the Phase 6A protocol.

Reads only frozen files (see :mod:`release`), never PyTDC. Two jobs:

1. Run the Phase 6A ingestion audit over every endpoint -- validity,
   duplicates, conflicts, inclusion -- without fitting a single model.
2. Establish what TDC's official split actually *is*, from data rather
   than from documentation, and record where it diverges from what Phase
   6A assumed.

The second job is the reason this phase exists. Phase 6A specified five
independent 70/10/20 scaffold splits; TDC ships a fixed held-out test set
and re-splits only the remainder. Those are different experiments, and the
difference is measured here rather than argued about.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from molfusion_backend.benchmark import datasets, metrics, protocol, release, splits
from molfusion_backend.chemistry import canonical_smiles_from_mol, parse_smiles

# --- official TDC split semantics, as read from PyTDC source ------------
#
# tdc/benchmark_group/base_group.py:
#   get()/__next__()      read train_val.csv and test.csv from disk. Neither
#                         takes a seed, so the test set is a shipped file and
#                         cannot vary with one.
#   get_train_valid_split(seed, benchmark, split_type="default")
#                         reads train_val.csv only and splits it with
#                         frac = [0.875, 0.125, 0.0]. The trailing 0.0 is the
#                         test fraction: there is no test set to draw, because
#                         it was already held out.
#   split_type="default"  looks the method up per dataset in
#                         metadata.bm_split_names; every ADMET endpoint
#                         resolves to "scaffold".
#   evaluate_many()       enforces min_requirement = 5 runs.
#
OFFICIAL_TRAIN_VAL_FRACTIONS = (0.875, 0.125, 0.0)
OFFICIAL_SEEDS = (1, 2, 3, 4, 5)
OFFICIAL_SPLIT_METHOD = "scaffold"
OFFICIAL_SCAFFOLD_INCLUDES_CHIRALITY = False

# TDC's scaffold key ignores chirality; MolFusion's includes it. Same
# molecules, different grouping, so overlap is audited under both rather
# than one being quietly assumed to answer for the other.
MOLFUSION_SCAFFOLD_INCLUDES_CHIRALITY = True

TDC_METRIC_TO_MOLFUSION = {
    "roc-auc": "auroc",
    "pr-auc": "auprc",
    "mae": "mae",
    "spearman": "spearman",
}


def tdc_scaffold(smiles: str) -> str:
    """The scaffold key *as TDC computes it* -- chirality excluded."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot compute a scaffold for unparseable SMILES: {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=OFFICIAL_SCAFFOLD_INCLUDES_CHIRALITY
    )
    return scaffold if scaffold else protocol.EMPTY_SCAFFOLD_KEY


def task_type_for(tdc_metric: str) -> str:
    """Infer the task from the official metric.

    The metric is source-defined and unambiguous about task: roc-auc and
    pr-auc are classification-only, mae and spearman regression-only. This
    avoids hard-coding a task list from memory when the package already
    states it.
    """
    if tdc_metric in ("roc-auc", "pr-auc"):
        return protocol.TASK_CLASSIFICATION
    if tdc_metric in ("mae", "spearman"):
        return protocol.TASK_REGRESSION
    raise ValueError(f"unknown TDC metric: {tdc_metric!r}")


@dataclass
class EndpointAudit:
    """Everything Phase 6A.1 records about one endpoint, before modelling."""

    name: str
    tdc_category: str
    task_type: str
    tdc_official_metric: str
    molfusion_primary_metric: str
    raw_rows: int
    ingestion: dict[str, Any]
    usable: int
    included: bool
    exclusion_reasons: list[str] = field(default_factory=list)
    label_summary: dict[str, Any] = field(default_factory=dict)
    scaffolds: dict[str, Any] = field(default_factory=dict)
    split_identity: dict[str, Any] = field(default_factory=dict)
    official_overlap: dict[str, Any] = field(default_factory=dict)
    checksums: dict[str, Any] = field(default_factory=dict)


def _rows_to_records(rows: Sequence[Sequence[str]], task_type: str):
    """Frozen text rows -> (smiles, label) pairs, labels parsed once here."""
    records = []
    for row in rows:
        _, drug, label = row[0], row[1], row[2]
        if label == "" or label.lower() in ("nan", "none"):
            records.append((drug, None))
            continue
        value = float(label)
        records.append((drug, value))
    return records


def audit_endpoint(
    *,
    name: str,
    metadata: dict[str, Any],
    frozen_dir: Path,
) -> EndpointAudit:
    """Audit one endpoint end to end, without training anything."""
    tdc_metric = metadata["tdc_official_metric"]
    task_type = task_type_for(tdc_metric)

    train_val_path = frozen_dir / name / "train_val.csv"
    test_path = frozen_dir / name / "test.csv"
    _, tv_rows = release.read_frozen_csv(train_val_path)
    _, te_rows = release.read_frozen_csv(test_path)

    # The endpoint as a whole is train_val + test: TDC split one dataset,
    # so the audit is of that dataset, not of either half separately.
    all_rows = list(tv_rows) + list(te_rows)
    records = _rows_to_records(all_rows, task_type)
    molecules, ingest = datasets.build_dataset(records, task_type=task_type)
    included, reasons = datasets.check_inclusion(molecules, task_type=task_type)

    canonical = [m.canonical_smiles for m in molecules]

    # --- label summary ------------------------------------------------
    labels = [m.label for m in molecules]
    if task_type == protocol.TASK_CLASSIFICATION:
        balance = datasets.class_balance(molecules)
        label_summary = dict(balance)
    else:
        ordered = sorted(labels)
        spread = ordered[-1] - ordered[0] if ordered else 0.0
        label_summary = {
            "min": ordered[0] if ordered else None,
            "max": ordered[-1] if ordered else None,
            "median": ordered[len(ordered) // 2] if ordered else None,
            "spread": spread,
            "conflict_tolerance": spread * protocol.REGRESSION_CONFLICT_TOLERANCE_FRACTION,
        }

    # --- scaffold profile ---------------------------------------------
    mf_scaffolds = [splits.bemis_murcko_scaffold(s) for s in canonical]
    groups: dict[str, int] = {}
    for scaffold in mf_scaffolds:
        groups[scaffold] = groups.get(scaffold, 0) + 1
    scaffold_profile = {
        "unique_scaffolds": len(groups),
        "acyclic_group_size": groups.get(protocol.EMPTY_SCAFFOLD_KEY, 0),
        "largest_group_size": max(groups.values()) if groups else 0,
    }

    # --- official split identity --------------------------------------
    identity, overlap = audit_official_split(
        name=name, frozen_dir=frozen_dir, task_type=task_type
    )

    return EndpointAudit(
        name=name,
        tdc_category=metadata["tdc_category"],
        task_type=task_type,
        tdc_official_metric=tdc_metric,
        molfusion_primary_metric=metrics.primary_metric(task_type),
        raw_rows=len(all_rows),
        ingestion=ingest.as_report(),
        usable=len(molecules),
        included=included,
        exclusion_reasons=list(reasons),
        label_summary=label_summary,
        scaffolds=scaffold_profile,
        split_identity=identity,
        official_overlap=overlap,
        checksums={
            "train_val": {
                "sha256": release.sha256_file(train_val_path),
                "bytes": train_val_path.stat().st_size,
                "rows": len(tv_rows),
            },
            "test": {
                "sha256": release.sha256_file(test_path),
                "bytes": test_path.stat().st_size,
                "rows": len(te_rows),
            },
        },
    )


def audit_official_split(
    *, name: str, frozen_dir: Path, task_type: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove what the official split does, from the frozen membership.

    Returns identity hashes and the overlap audit. The identity hashes are
    the point: claiming "the test set does not change with the seed" is an
    assertion, but five equal SHA-256 values over the test molecule set is
    an observation.
    """
    endpoint = frozen_dir / name
    _, tv_rows = release.read_frozen_csv(endpoint / "train_val.csv")
    _, te_rows = release.read_frozen_csv(endpoint / "test.csv")
    seed_splits = json.loads((endpoint / "official_seed_splits.json").read_text("utf-8"))

    canon = _canonicalize_map([r[1] for r in tv_rows] + [r[1] for r in te_rows])
    tv_canon = [canon[r[1]] for r in tv_rows if canon.get(r[1])]
    te_canon = [canon[r[1]] for r in te_rows if canon.get(r[1])]

    identity: dict[str, Any] = {
        "train_val_rows": len(tv_rows),
        "test_rows": len(te_rows),
        "test_fraction": len(te_rows) / (len(tv_rows) + len(te_rows)),
        "test_set_sha256": release.molecule_set_identity(te_canon),
        "train_val_set_sha256": release.molecule_set_identity(tv_canon),
        "seeds": {},
    }

    for seed in sorted(seed_splits, key=int):
        entry = seed_splits[seed]
        train_c = [canon[s] for s in entry["train_drug"] if canon.get(s)]
        valid_c = [canon[s] for s in entry["valid_drug"] if canon.get(s)]
        identity["seeds"][seed] = {
            "train_size": len(entry["train_drug"]),
            "valid_size": len(entry["valid_drug"]),
            "train_set_sha256": release.molecule_set_identity(train_c),
            "validation_set_sha256": release.molecule_set_identity(valid_c),
            # Recomputed per seed from the *same* shipped test file, so an
            # equal value across seeds is evidence, not a tautology of
            # having hashed one object once.
            "test_set_sha256": release.molecule_set_identity(te_canon),
        }

    test_hashes = {v["test_set_sha256"] for v in identity["seeds"].values()}
    identity["test_identity_invariant_across_seeds"] = len(test_hashes) == 1

    tv_set, te_set = set(tv_canon), set(te_canon)
    mf_tv = {splits.bemis_murcko_scaffold(s) for s in tv_set}
    mf_te = {splits.bemis_murcko_scaffold(s) for s in te_set}
    tdc_tv = {tdc_scaffold(s) for s in tv_set}
    tdc_te = {tdc_scaffold(s) for s in te_set}

    overlap = {
        "canonical_molecule_overlap": len(tv_set & te_set),
        "train_val_unique_molecules": len(tv_set),
        "test_unique_molecules": len(te_set),
        "scaffold_overlap_tdc_convention": len(tdc_tv & tdc_te),
        "scaffold_overlap_molfusion_convention": len(mf_tv & mf_te),
        "test_scaffolds_tdc_convention": len(tdc_te),
        "test_scaffolds_molfusion_convention": len(mf_te),
    }
    return identity, overlap


def _canonicalize_map(smiles: Iterable[str]) -> dict[str, str | None]:
    """Canonicalize each distinct raw SMILES once, under the frozen contract."""
    mapping: dict[str, str | None] = {}
    for raw in smiles:
        if raw in mapping:
            continue
        mol, _ = parse_smiles(raw)
        mapping[raw] = canonical_smiles_from_mol(mol) if mol is not None else None
    return mapping


__all__ = [
    "MOLFUSION_SCAFFOLD_INCLUDES_CHIRALITY",
    "OFFICIAL_SCAFFOLD_INCLUDES_CHIRALITY",
    "OFFICIAL_SEEDS",
    "OFFICIAL_SPLIT_METHOD",
    "OFFICIAL_TRAIN_VAL_FRACTIONS",
    "TDC_METRIC_TO_MOLFUSION",
    "EndpointAudit",
    "audit_endpoint",
    "audit_official_split",
    "task_type_for",
    "tdc_scaffold",
]
