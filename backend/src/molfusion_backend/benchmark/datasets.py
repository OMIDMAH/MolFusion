"""Dataset ingestion: validity, canonical identity, duplicates, labels.

Everything here happens *before* a split exists, because every decision it
makes changes what the split is drawn from. The audit is deliberately
explicit and counted: a benchmark that quietly drops records produces
numbers nobody can reconstruct.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from molfusion_backend.benchmark import protocol
from molfusion_backend.chemistry import canonical_smiles_from_mol, parse_smiles


@dataclass(frozen=True)
class LabelledMolecule:
    """One usable record: a canonical molecule and its label."""

    canonical_smiles: str
    label: float
    source_smiles: tuple[str, ...] = ()


@dataclass
class DatasetAudit:
    """Full accounting of what happened to every input record.

    Every input row lands in exactly one category, and the identity is
    checked, so a reader can see the cost of each rule rather than inferring
    it from a shrinking row count.
    """

    input_records: int = 0
    missing_label_dropped: int = 0
    rdkit_invalid: int = 0
    duplicates_collapsed: int = 0
    duplicates_conflicting_dropped: int = 0
    usable: int = 0
    conflicting_molecules: list[str] = field(default_factory=list)

    def validate(self) -> None:
        accounted = (
            self.missing_label_dropped
            + self.rdkit_invalid
            + self.duplicates_collapsed
            + self.duplicates_conflicting_dropped
            + self.usable
        )
        if accounted != self.input_records:
            raise ValueError(
                "Dataset accounting does not balance: "
                f"{accounted} categorised != {self.input_records} input records."
            )

    def as_report(self) -> dict[str, Any]:
        return {
            "input_records": self.input_records,
            "missing_label_dropped": self.missing_label_dropped,
            "rdkit_invalid": self.rdkit_invalid,
            "duplicates_collapsed": self.duplicates_collapsed,
            "duplicates_conflicting_dropped": self.duplicates_conflicting_dropped,
            "usable": self.usable,
            "conflicting_molecule_count": len(self.conflicting_molecules),
        }


def _labels_agree(values: Sequence[float], task_type: str, spread: float) -> bool:
    """Whether repeated measurements of one molecule are the same label.

    Classification labels must match exactly. Regression labels are compared
    against a tolerance derived from the endpoint's own label spread, because
    two measurements of the same compound rarely agree to the last float and
    an absolute tolerance would mean something different for an endpoint in
    log units than for one in percent.
    """
    if task_type == protocol.TASK_CLASSIFICATION:
        return len(set(values)) == 1
    tolerance = protocol.REGRESSION_CONFLICT_TOLERANCE_FRACTION * spread
    return (max(values) - min(values)) <= tolerance


def build_dataset(
    records: Iterable[tuple[str, float | None]], *, task_type: str
) -> tuple[list[LabelledMolecule], DatasetAudit]:
    """Turn raw (smiles, label) rows into a deduplicated, audited dataset.

    Order of operations is fixed and matters: drop missing labels, then
    parse, then canonicalize, then deduplicate. Deduplicating before
    canonicalizing would miss two spellings of one molecule, which is the
    leakage this exists to prevent.
    """
    if task_type not in protocol.TASK_TYPES:
        raise ValueError(f"unknown task_type: {task_type!r}")

    audit = DatasetAudit()
    grouped: dict[str, list[float]] = {}
    sources: dict[str, list[str]] = {}

    rows = list(records)
    audit.input_records = len(rows)

    for smiles, label in rows:
        if label is None:
            # Never imputed: a target is the thing being predicted, and
            # inventing one would fabricate the ground truth.
            audit.missing_label_dropped += 1
            continue
        mol, _ = parse_smiles(smiles)
        if mol is None:
            audit.rdkit_invalid += 1
            continue
        canonical = canonical_smiles_from_mol(mol)
        grouped.setdefault(canonical, []).append(float(label))
        sources.setdefault(canonical, []).append(smiles)

    numeric = [value for values in grouped.values() for value in values]
    spread = (max(numeric) - min(numeric)) if numeric else 0.0

    molecules: list[LabelledMolecule] = []
    for canonical in sorted(grouped):
        values = grouped[canonical]
        if len(values) > 1:
            if _labels_agree(values, task_type, spread):
                audit.duplicates_collapsed += len(values) - 1
            else:
                # Both copies go. Averaging would assert a value the source
                # never did; keeping one would make the dataset depend on row
                # order.
                audit.duplicates_conflicting_dropped += len(values)
                audit.conflicting_molecules.append(canonical)
                continue
        molecules.append(
            LabelledMolecule(
                canonical_smiles=canonical,
                label=values[0],
                source_smiles=tuple(sources[canonical]),
            )
        )

    audit.usable = len(molecules)
    audit.validate()
    return molecules, audit


def check_inclusion(
    molecules: Sequence[LabelledMolecule], *, task_type: str
) -> tuple[bool, list[str]]:
    """Whether an endpoint meets the frozen inclusion criteria.

    Returns the reasons rather than raising, so an excluded endpoint can be
    reported as excluded-with-cause instead of vanishing.
    """
    reasons: list[str] = []
    if len(molecules) < protocol.MINIMUM_MOLECULES:
        reasons.append(
            f"only {len(molecules)} usable molecules, minimum is {protocol.MINIMUM_MOLECULES}"
        )
    if task_type == protocol.TASK_CLASSIFICATION and molecules:
        labels = [molecule.label for molecule in molecules]
        distinct = set(labels)
        if len(distinct) < 2:
            reasons.append("only one class present")
        else:
            minority = min(labels.count(value) for value in distinct)
            if minority < protocol.MINIMUM_MINORITY_CLASS:
                reasons.append(
                    f"minority class has {minority} molecules, minimum is "
                    f"{protocol.MINIMUM_MINORITY_CLASS}"
                )
    return (not reasons), reasons


def class_balance(molecules: Sequence[LabelledMolecule]) -> dict[str, Any]:
    """Per-class counts and the minority fraction, for imbalance reporting."""
    counts: dict[str, int] = {}
    for molecule in molecules:
        key = str(molecule.label)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    return {
        "counts": {key: counts[key] for key in sorted(counts)},
        "total": total,
        "minority_fraction": (min(counts.values()) / total) if total and counts else None,
    }


__all__ = [
    "DatasetAudit",
    "LabelledMolecule",
    "build_dataset",
    "check_inclusion",
    "class_balance",
]
