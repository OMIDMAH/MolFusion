"""Representation extraction for a fixed set of molecules.

Uses the production agents unchanged, through the same registry the API
uses, so the benchmark measures what MolFusion actually ships rather than a
reimplementation of it.

The output that matters is not only the matrix but the accounting beside it:
which molecules every representation succeeded on. Comparing representations
on different molecule sets would let one look better by having failed on the
hard compounds, so the evaluation universe is the intersection, computed
once and reported.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from molfusion_backend.agents import registry as agent_registry
from molfusion_backend.benchmark import protocol
from molfusion_backend.chemistry import parse_smiles


@dataclass
class ExtractionResult:
    """One representation's matrix over a molecule list, plus its failures."""

    representation: str
    version: str
    matrix: np.ndarray
    succeeded: tuple[int, ...]
    failures: dict[int, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.size else 0


def extract(
    canonical_smiles: Sequence[str], representation: str
) -> ExtractionResult:
    """Compute one representation for every molecule, isolating failures.

    A failure is recorded per molecule rather than aborting: Phase 5H
    established that one molecule's representation failing is a normal
    outcome, and the benchmark needs the count, not an exception.
    """
    agent = agent_registry.get(representation)
    if agent.output_structure != "vector":
        raise ValueError(
            f"{representation!r} is a {agent.output_structure!r} agent; Track A "
            "covers fixed-length vectors only (see protocol.TRACK_B_DECISION)"
        )

    vectors: list[np.ndarray] = []
    succeeded: list[int] = []
    failures: dict[int, str] = {}

    started = time.perf_counter()
    for index, smiles in enumerate(canonical_smiles):
        mol, error = parse_smiles(smiles)
        if mol is None:
            failures[index] = error or "RDKit could not parse the molecule"
            continue
        try:
            vectors.append(np.asarray(agent.compute(mol), dtype=np.float64))
            succeeded.append(index)
        except ValueError as exc:
            failures[index] = str(exc)
    seconds = time.perf_counter() - started

    matrix = np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float64)
    return ExtractionResult(
        representation=representation,
        version=agent.version,
        matrix=matrix,
        succeeded=tuple(succeeded),
        failures=failures,
        seconds=seconds,
    )


def common_evaluation_set(
    results: Sequence[ExtractionResult], total: int
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Indices every representation handled, plus the accounting.

    The intersection is the primary evaluation universe. Its cost is
    reported, and an endpoint losing more than the alert threshold is flagged
    so the loss is argued about rather than absorbed silently.
    """
    if not results:
        return tuple(range(total)), {"representations": 0}

    common = set(results[0].succeeded)
    for result in results[1:]:
        common &= set(result.succeeded)

    lost = total - len(common)
    fraction = (lost / total) if total else 0.0
    return tuple(sorted(common)), {
        "representations": len(results),
        "input_molecules": total,
        "common_evaluation_set": len(common),
        "lost_to_intersection": lost,
        "lost_fraction": fraction,
        "exceeds_alert_threshold": fraction > protocol.COMMON_SET_LOSS_ALERT,
        "per_representation_failures": {
            result.representation: len(result.failures) for result in results
        },
    }


def sparsity(matrix: np.ndarray) -> float | None:
    """Fraction of entries that are exactly zero, or None for an empty matrix."""
    if matrix.size == 0:
        return None
    return float(1.0 - np.count_nonzero(matrix) / matrix.size)


def representation_profile(result: ExtractionResult) -> dict[str, Any]:
    """The Table A row for one representation: shape, dtype, sparsity, cost."""
    agent = agent_registry.get(result.representation)
    return {
        "representation": result.representation,
        "version": result.version,
        "value_type": agent.value_type,
        "output_structure": agent.output_structure,
        "dimension": result.dimension,
        "dtype": str(result.matrix.dtype) if result.matrix.size else None,
        "sparsity": sparsity(result.matrix),
        "molecules": len(result.succeeded),
        "failures": len(result.failures),
        "feature_seconds": result.seconds,
        "seconds_per_molecule": (
            result.seconds / len(result.succeeded) if result.succeeded else None
        ),
    }


__all__ = [
    "ExtractionResult",
    "common_evaluation_set",
    "extract",
    "representation_profile",
    "sparsity",
]
