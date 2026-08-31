"""Deterministic Bemis-Murcko scaffold splits.

A random split asks whether a model can interpolate among scaffolds it has
already seen. That is not the question a representation benchmark should be
answering: the interesting generalisation is to chemistry the training set
does not contain, and a random split reliably overstates it.

Scaffold splitting is therefore the default. Molecules are grouped by
Bemis-Murcko scaffold and whole groups are assigned to one partition, so no
scaffold ever straddles the train/test boundary.

Determinism is structural rather than incidental: groups are ordered by a
hash of the scaffold together with the seed, so the assignment depends only
on (scaffold, seed) and never on input order, dictionary iteration, or the
molecules' positions in the source file.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from molfusion_backend.benchmark import protocol
from molfusion_backend.benchmark.datasets import LabelledMolecule


@dataclass(frozen=True)
class Split:
    """One train/validation/test partition, by index into the molecule list."""

    split_id: str
    seed: int
    strategy: str
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def bemis_murcko_scaffold(smiles: str) -> str:
    """The molecule's Bemis-Murcko scaffold, or the acyclic group key.

    An acyclic molecule has no ring system and RDKit returns an empty
    scaffold for it. Those are collected under one explicit key rather than
    each becoming a singleton group -- the conventional treatment, and the
    group's size is reported because it can be substantial.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot compute a scaffold for unparseable SMILES: {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    return scaffold if scaffold else protocol.EMPTY_SCAFFOLD_KEY


def _group_order_key(scaffold: str, seed: int) -> str:
    """A stable pseudo-random ordering key for one scaffold group.

    Hashing (seed, scaffold) rather than shuffling a list means the order
    depends only on those two values: it cannot drift with Python's hash
    randomisation, the order groups were discovered in, or the number of
    groups.
    """
    return hashlib.sha256(f"{seed}:{scaffold}".encode("utf-8")).hexdigest()


def scaffold_split(
    molecules: Sequence[LabelledMolecule],
    *,
    seed: int,
    train_fraction: float = protocol.TRAIN_FRACTION,
    validation_fraction: float = protocol.VALIDATION_FRACTION,
) -> Split:
    """Assign whole scaffold groups to train/validation/test.

    Groups are filled largest-first within the hash order so the biggest
    scaffold families land in training, which is the standard behaviour and
    keeps the small partitions from being dominated by one series. Sizes are
    therefore approximate: a partition boundary never splits a group, because
    doing so would put the same scaffold on both sides of the evaluation.
    """
    if not molecules:
        raise ValueError("cannot split an empty dataset")
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions leave no test partition")

    groups: dict[str, list[int]] = {}
    for index, molecule in enumerate(molecules):
        groups.setdefault(bemis_murcko_scaffold(molecule.canonical_smiles), []).append(index)

    # Largest groups first; ties broken by the seeded hash so the order is
    # deterministic without being an artefact of insertion order.
    ordered = sorted(
        groups.items(), key=lambda item: (-len(item[1]), _group_order_key(item[0], seed))
    )

    total = len(molecules)
    train_target = train_fraction * total
    validation_target = (train_fraction + validation_fraction) * total

    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for _scaffold, indices in ordered:
        if len(train) < train_target:
            train.extend(indices)
        elif len(train) + len(validation) < validation_target:
            validation.extend(indices)
        else:
            test.extend(indices)

    return Split(
        split_id=f"{protocol.SPLIT_SCAFFOLD}:seed={seed}",
        seed=seed,
        strategy=protocol.SPLIT_SCAFFOLD,
        train=tuple(sorted(train)),
        validation=tuple(sorted(validation)),
        test=tuple(sorted(test)),
    )


def generate_splits(
    molecules: Sequence[LabelledMolecule], *, seeds: Sequence[int] = protocol.SPLIT_SEEDS
) -> list[Split]:
    return [scaffold_split(molecules, seed=seed) for seed in seeds]


def audit_split(molecules: Sequence[LabelledMolecule], split: Split) -> dict[str, Any]:
    """Evidence that a split is disjoint and scaffold-clean.

    Checked rather than assumed: an overlap between train and test is the one
    defect that would invalidate every number derived from the run, and it is
    cheap to prove absent.
    """
    partitions = {"train": split.train, "validation": split.validation, "test": split.test}
    indices = [index for part in partitions.values() for index in part]
    if sorted(indices) != list(range(len(molecules))):
        raise ValueError("split does not partition the dataset exactly once")

    canonical = {
        name: {molecules[index].canonical_smiles for index in part}
        for name, part in partitions.items()
    }
    scaffolds = {
        name: {bemis_murcko_scaffold(molecules[index].canonical_smiles) for index in part}
        for name, part in partitions.items()
    }

    return {
        "split_id": split.split_id,
        "seed": split.seed,
        "strategy": split.strategy,
        "sizes": split.sizes,
        "molecule_overlap": {
            "train_test": len(canonical["train"] & canonical["test"]),
            "train_validation": len(canonical["train"] & canonical["validation"]),
            "validation_test": len(canonical["validation"] & canonical["test"]),
        },
        "scaffold_overlap": {
            "train_test": len(scaffolds["train"] & scaffolds["test"]),
            "train_validation": len(scaffolds["train"] & scaffolds["validation"]),
            "validation_test": len(scaffolds["validation"] & scaffolds["test"]),
        },
        "distinct_scaffolds": len(set().union(*scaffolds.values())) if molecules else 0,
        "acyclic_group_size": sum(
            1
            for molecule in molecules
            if bemis_murcko_scaffold(molecule.canonical_smiles)
            == protocol.EMPTY_SCAFFOLD_KEY
        ),
    }


__all__ = [
    "Split",
    "audit_split",
    "bemis_murcko_scaffold",
    "generate_splits",
    "scaffold_split",
]
