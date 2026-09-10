"""A minimal end-to-end evaluation path, to validate the protocol wiring.

Deliberately small. Phase 6A's job is to freeze the protocol, not to build
the execution framework, so this is the thinnest code that proves the pieces
connect: dataset -> split -> features -> pipeline -> tuned on validation ->
scored on test -> long-format rows.

Scores produced by this function on fixture data carry no scientific meaning
and must not be read as results. The full matrix is a later phase.
"""

import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from molfusion_backend.benchmark import metrics, pipelines, protocol
from molfusion_backend.benchmark.datasets import LabelledMolecule
from molfusion_backend.benchmark.features import ExtractionResult
from molfusion_backend.benchmark.results import ResultRow
from molfusion_backend.benchmark.splits import Split


def _partition(matrix: np.ndarray, labels: np.ndarray, rows: Sequence[int]):
    index = np.asarray(rows, dtype=int)
    return matrix[index], labels[index]


def evaluate_representation(
    *,
    dataset: str,
    endpoint: str,
    task_type: str,
    molecules: Sequence[LabelledMolecule],
    extraction: ExtractionResult,
    evaluation_rows: Sequence[int],
    split: Split,
    probe: str,
) -> list[ResultRow]:
    """Tune on validation, score once on test, and emit one row per metric.

    The test partition is read exactly once, after the hyperparameter is
    chosen on validation. That ordering is the whole guard against tuning on
    the evaluation data, so it is expressed as straight-line code here rather
    than left to a caller's discipline.
    """
    if task_type not in protocol.TASK_TYPES:
        raise ValueError(f"unknown task_type: {task_type!r}")

    # Map dataset indices onto rows of the extracted matrix, restricted to
    # the common evaluation set so every representation is scored on the same
    # molecules.
    row_of = {index: position for position, index in enumerate(extraction.succeeded)}
    keep = [index for index in evaluation_rows if index in row_of]
    usable = set(keep)

    matrix = extraction.matrix[[row_of[index] for index in keep]]
    labels = np.asarray([molecules[index].label for index in keep], dtype=np.float64)
    local = {index: position for position, index in enumerate(keep)}

    def rows_for(partition: Sequence[int]) -> list[int]:
        return [local[index] for index in partition if index in usable]

    train_rows, validation_rows, test_rows = (
        rows_for(split.train),
        rows_for(split.validation),
        rows_for(split.test),
    )
    if not train_rows or not test_rows:
        raise ValueError(
            f"{endpoint}/{extraction.representation}: split leaves an empty "
            "train or test partition after restricting to the common set"
        )

    x_train, y_train = _partition(matrix, labels, train_rows)
    x_valid, y_valid = _partition(matrix, labels, validation_rows or train_rows)
    x_test, y_test = _partition(matrix, labels, test_rows)

    if task_type == protocol.TASK_CLASSIFICATION:
        y_train, y_valid, y_test = (a.astype(int) for a in (y_train, y_valid, y_test))

    primary = metrics.primary_metric(task_type)

    # --- selection: validation only -----------------------------------
    best_params: dict[str, Any] = {}
    best_score = -np.inf
    for candidate in pipelines.hyperparameter_grid(probe, task_type):
        model = pipelines.build_pipeline(
            representation=extraction.representation,
            probe=probe,
            task_type=task_type,
            hyperparameters=candidate,
        )
        model.fit(x_train, y_train)
        try:
            scored = _score(model, x_valid, y_valid, task_type)[primary]
        except ValueError:
            # e.g. a validation fold with a single class: that candidate
            # cannot be scored, so it is skipped rather than silently ranked.
            continue
        oriented = metrics.orient(primary, scored)
        if oriented > best_score:
            best_score, best_params = oriented, candidate

    # --- final fit and the single look at test -------------------------
    model = pipelines.build_pipeline(
        representation=extraction.representation,
        probe=probe,
        task_type=task_type,
        hyperparameters=best_params,
    )
    started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scores = _score(model, x_test, y_test, task_type)
    predict_seconds = time.perf_counter() - started

    return [
        ResultRow(
            dataset=dataset,
            endpoint=endpoint,
            task_type=task_type,
            split_id=split.split_id,
            split_strategy=split.strategy,
            seed=split.seed,
            representation=extraction.representation,
            representation_version=extraction.version,
            model=pipelines.model_name(probe, task_type),
            probe=probe,
            metric=metric,
            value=value,
            n_train=len(train_rows),
            n_valid=len(validation_rows),
            n_test=len(test_rows),
            feature_dim=extraction.dimension,
            feature_failures=len(extraction.failures),
            hyperparameters=json.dumps(best_params, sort_keys=True),
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            feature_seconds=extraction.seconds,
        )
        for metric, value in sorted(scores.items())
    ]


def _score(model, x, y, task_type: str) -> dict[str, float]:
    if task_type == protocol.TASK_CLASSIFICATION:
        probabilities = model.predict_proba(x)[:, 1]
        return metrics.classification_metrics(y, probabilities, model.predict(x))
    return metrics.regression_metrics(y, model.predict(x))


__all__ = ["evaluate_representation"]
