"""A matrix-level feature cache for benchmark execution.

Phase 6A froze a *per-molecule* cache key (:mod:`cache`). That contract is
unchanged and still describes what a cached vector depends on. What the
execution phase actually needs is coarser: one matrix per
``endpoint x representation``, reused across five seeds, two probe families
and four hyperparameter candidates -- forty model fits that would otherwise
recompute identical features forty times.

The danger with a matrix cache is subtler than with a per-molecule one. A
matrix is only meaningful together with the row order it was built for, and
"these two files probably have the same rows" is exactly the kind of
assumption that produces a silently wrong benchmark. So the row identity is
hashed *order-sensitively* and stored with the matrix, and a load that
cannot prove row i corresponds to molecule i fails rather than guesses.

Nothing here imports PyTDC.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = 1

#: Matrices are stored float64 -- the dtype the agents emit and the dtype the
#: pipelines consume. Storing float32 would make the cache lossy in a way no
#: downstream check could detect.
MATRIX_DTYPE = np.dtype("<f8")

MATRIX_NPY_VERSION = (1, 0)


class FeatureCacheError(RuntimeError):
    """A cache entry could not be produced, read, or trusted."""


def row_identity(canonical_smiles: Sequence[str]) -> str:
    """An ORDER-SENSITIVE hash of the molecule rows a matrix was built for.

    Deliberately not :func:`release.molecule_set_identity`, which hashes a
    set. Here order is the whole point: the claim being protected is "matrix
    row i is molecule i", and a set hash would happily accept a permuted
    matrix. Duplicated molecules are also preserved rather than collapsed --
    Track A1 consumes the official rows as shipped, duplicates included.
    """
    digest = hashlib.sha256()
    digest.update(f"{CACHE_SCHEMA_VERSION}\x1frows={len(canonical_smiles)}".encode())
    for index, smiles in enumerate(canonical_smiles):
        digest.update(f"\x1e{index}\x1f{smiles}".encode())
    return digest.hexdigest()


def matrix_cache_key(
    *,
    release_identity: str,
    endpoint: str,
    agent_id: str,
    agent_version: str,
    output_dim: int,
    normalization_id: str,
    row_identity_sha256: str,
    artifact_identity: str | None = None,
) -> str:
    """A key covering every input the cached matrix depends on.

    Each component earns its place:

      release_identity     which frozen benchmark release the rows came from
      endpoint             which endpoint within it
      agent_id/version     the code that computed the vectors
      output_dim           the declared width; a changed dimension is a
                           changed representation even at the same version
      normalization_id     the canonicalization contract behind the inputs
      row_identity_sha256  the exact ordered molecule list
      artifact_identity    the fitted payload for artifact-backed agents

    Notably absent: any filename, directory, timestamp, split, seed, probe,
    label or hyperparameter. A file path is not an identity -- a cache keyed
    on one would serve a matrix computed by a superseded artifact with
    nothing downstream able to notice -- and features do not depend on the
    model that will consume them.
    """
    payload = "\x1f".join(
        (
            str(CACHE_SCHEMA_VERSION),
            release_identity,
            endpoint,
            agent_id,
            agent_version,
            str(output_dim),
            normalization_id,
            row_identity_sha256,
            artifact_identity or "",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedMatrix:
    """A feature matrix together with everything needed to trust it."""

    matrix: np.ndarray
    succeeded: tuple[int, ...]
    failures: dict[int, str]
    metadata: dict[str, Any]

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.size else 0


class FeatureStore:
    """An on-disk store of ``endpoint x representation`` feature matrices.

    One directory per entry, named by the cache key, holding the matrix and
    a metadata sidecar. Writes stage into a sibling temporary directory and
    are finalized by an atomic rename, so a killed process leaves either a
    complete entry or nothing -- never a half-written matrix that validates.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def entry_dir(self, key: str) -> Path:
        return self.root / key[:2] / key

    # -- reading --------------------------------------------------------

    def load(self, key: str, *, expect: dict[str, Any]) -> CachedMatrix | None:
        """Return the cached matrix, or None if absent.

        Raises :class:`FeatureCacheError` if an entry exists but disagrees
        with ``expect``. That asymmetry is deliberate: a missing entry is a
        normal cold cache, but a *present* entry that does not match is
        evidence of a stale or corrupt store, and silently recomputing over
        it would hide the problem instead of surfacing it.
        """
        directory = self.entry_dir(key)
        meta_path = directory / "metadata.json"
        matrix_path = directory / "matrix.npy"
        if not meta_path.exists() or not matrix_path.exists():
            return None

        try:
            metadata = json.loads(meta_path.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            raise FeatureCacheError(f"{key}: metadata is not readable JSON: {exc}") from exc

        for field, expected in expect.items():
            actual = metadata.get(field)
            if actual != expected:
                raise FeatureCacheError(
                    f"{key}: cached {field!r} is {actual!r}, expected {expected!r}; "
                    "refusing to reuse a stale entry"
                )

        matrix = np.load(matrix_path, allow_pickle=False)
        if matrix.dtype != MATRIX_DTYPE:
            raise FeatureCacheError(
                f"{key}: cached dtype is {matrix.dtype}, expected {MATRIX_DTYPE}"
            )
        expected_shape = tuple(metadata["matrix_shape"])
        if matrix.shape != expected_shape:
            raise FeatureCacheError(
                f"{key}: matrix shape {matrix.shape} != recorded {expected_shape}"
            )
        succeeded = tuple(metadata["succeeded"])
        if len(succeeded) != matrix.shape[0]:
            raise FeatureCacheError(
                f"{key}: {len(succeeded)} row indices for {matrix.shape[0]} matrix rows"
            )
        return CachedMatrix(
            matrix=matrix,
            succeeded=succeeded,
            failures={int(k): v for k, v in metadata.get("failures", {}).items()},
            metadata=metadata,
        )

    # -- writing --------------------------------------------------------

    def store(
        self,
        key: str,
        *,
        matrix: np.ndarray,
        succeeded: Sequence[int],
        failures: dict[int, str],
        metadata: dict[str, Any],
    ) -> Path:
        """Write one entry atomically, and return its directory."""
        array = np.ascontiguousarray(matrix, dtype=MATRIX_DTYPE)
        if array.shape[0] != len(succeeded):
            raise FeatureCacheError(
                f"{key}: {array.shape[0]} matrix rows for {len(succeeded)} row indices"
            )

        payload = dict(metadata)
        payload.update(
            {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "cache_key": key,
                "matrix_shape": list(array.shape),
                "matrix_dtype": str(MATRIX_DTYPE),
                "succeeded": [int(i) for i in succeeded],
                "failures": {str(k): v for k, v in failures.items()},
            }
        )

        final = self.entry_dir(key)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=final.parent))
        try:
            with open(staging / "matrix.npy", "wb") as handle:
                np.lib.format.write_array(
                    handle, array, version=MATRIX_NPY_VERSION, allow_pickle=False
                )
            with open(staging / "metadata.json", "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=1, sort_keys=True)
                handle.write("\n")
            _atomic_replace_dir(staging, final)
        except BaseException:
            _remove_tree(staging)
            raise
        return final


def _atomic_replace_dir(staging: Path, final: Path) -> None:
    """Move a completed staging directory into place.

    On Windows ``os.replace`` refuses a non-empty destination directory, so
    an existing entry is retired first. Losing the race here is benign:
    another worker producing the same key produces the same bytes, because
    the key covers everything the matrix depends on.
    """
    if final.exists():
        retired = final.with_name(final.name + ".retired")
        _remove_tree(retired)
        try:
            os.replace(final, retired)
        except OSError:
            _remove_tree(staging)
            return
        _remove_tree(retired)
    try:
        os.replace(staging, final)
    except OSError:
        if final.exists():
            _remove_tree(staging)
            return
        raise


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def cache_contract() -> dict[str, Any]:
    """The matrix-cache contract, for the run manifest."""
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "granularity": "one matrix per endpoint x representation",
        "digest": "sha256",
        "key_components": [
            "cache_schema_version",
            "release_identity",
            "endpoint",
            "agent_id",
            "agent_version",
            "output_dim",
            "normalization_id",
            "row_identity_sha256 (order-sensitive)",
            "artifact_identity (empty for stateless agents)",
        ],
        "not_keyed_on": [
            "filename",
            "directory",
            "timestamp",
            "split",
            "seed",
            "probe",
            "hyperparameters",
            "labels",
        ],
        "row_identity": (
            "order-sensitive sha256 over (index, canonical_smiles) pairs, so a "
            "permuted or truncated row list cannot match; duplicates are "
            "preserved, because Track A1 consumes official rows as shipped"
        ),
        "matrix_dtype": str(MATRIX_DTYPE),
        "atomicity": "staging directory finalized by os.replace",
        "validated_on_load": [
            "cache_schema_version",
            "release_identity",
            "endpoint",
            "agent_id",
            "agent_version",
            "output_dim",
            "row_identity_sha256",
            "artifact_identity",
            "matrix dtype",
            "matrix shape",
            "row-index count vs matrix rows",
        ],
        "on_mismatch": "raise FeatureCacheError; never silently recompute over a stale entry",
    }


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "MATRIX_DTYPE",
    "CachedMatrix",
    "FeatureCacheError",
    "FeatureStore",
    "cache_contract",
    "matrix_cache_key",
    "row_identity",
]
