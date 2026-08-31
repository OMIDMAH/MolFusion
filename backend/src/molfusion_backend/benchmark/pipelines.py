"""Probe pipelines: the scaling policy, made executable.

Two rules govern everything here.

Preprocessing is chosen per representation and per probe. Standardizing a
1024-bit fingerprint would destroy its sparsity and give an absent bit a
nonzero value, which is not what an absent bit means; leaving physicochemical
descriptors unscaled would let one descriptor's units dominate a penalized
linear model. A single uniform rule would be wrong for one of them.

Every fitted step -- scaler, imputer -- sees the training split only.
scikit-learn's `Pipeline` enforces this structurally: `fit` on the training
data fits the steps, and `predict` on validation or test only transforms.
Fitting a scaler on the full dataset is the classic silent leak, and it is
prevented by construction here rather than by remembering not to.
"""

from typing import Any

from molfusion_backend.benchmark import protocol


def scaling_for(representation: str, probe: str) -> str:
    if probe == protocol.PROBE_LINEAR:
        table = protocol.LINEAR_SCALING
    elif probe == protocol.PROBE_NONLINEAR:
        table = protocol.NONLINEAR_SCALING
    else:
        raise ValueError(f"unknown probe: {probe!r}")
    if representation not in table:
        raise ValueError(f"no scaling policy for representation {representation!r}")
    return table[representation]


def build_pipeline(
    *,
    representation: str,
    probe: str,
    task_type: str,
    hyperparameters: dict[str, Any] | None = None,
    seed: int = protocol.MODEL_SEED,
    class_weight: str | None = None,
):
    """The pipeline for one (representation, probe, task) combination."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if task_type not in protocol.TASK_TYPES:
        raise ValueError(f"unknown task_type: {task_type!r}")
    params = dict(hyperparameters or {})
    steps: list[tuple[str, Any]] = []

    if probe == protocol.PROBE_LINEAR:
        # RDKit descriptors legitimately emit NaN where a descriptor cannot
        # be computed. A linear model cannot consume it, so it is imputed --
        # from the training split's medians, never the full column.
        steps.append(("impute", SimpleImputer(strategy="median")))
        if scaling_for(representation, probe) == protocol.SCALING_STANDARD:
            steps.append(("scale", StandardScaler()))
        if task_type == protocol.TASK_CLASSIFICATION:
            estimator = LogisticRegression(
                max_iter=5000, random_state=seed, class_weight=class_weight, **params
            )
        else:
            estimator = Ridge(random_state=seed, **params)
    elif probe == protocol.PROBE_NONLINEAR:
        # No scaler and no imputer: the model is scale-invariant and consumes
        # NaN natively, so adding either would fit a step that changes
        # nothing while introducing a place for leakage to hide.
        if task_type == protocol.TASK_CLASSIFICATION:
            estimator = HistGradientBoostingClassifier(
                random_state=seed, class_weight=class_weight, **params
            )
        else:
            estimator = HistGradientBoostingRegressor(random_state=seed, **params)
    else:
        raise ValueError(f"unknown probe: {probe!r}")

    steps.append(("model", estimator))
    return Pipeline(steps)


def hyperparameter_grid(probe: str, task_type: str) -> list[dict[str, Any]]:
    """The frozen grid, expanded into explicit candidate dictionaries.

    Identical for every representation by construction: the function does not
    take one, so a per-representation budget cannot be introduced by accident.
    """
    if probe == protocol.PROBE_LINEAR:
        grid = (
            protocol.LINEAR_CLASSIFIER_GRID
            if task_type == protocol.TASK_CLASSIFICATION
            else protocol.LINEAR_REGRESSOR_GRID
        )
    elif probe == protocol.PROBE_NONLINEAR:
        grid = protocol.NONLINEAR_GRID
    else:
        raise ValueError(f"unknown probe: {probe!r}")

    candidates: list[dict[str, Any]] = [{}]
    for name, values in grid.items():
        candidates = [{**candidate, name: value} for candidate in candidates for value in values]
    return candidates


def model_name(probe: str, task_type: str) -> str:
    if probe == protocol.PROBE_LINEAR:
        return (
            protocol.LINEAR_CLASSIFIER
            if task_type == protocol.TASK_CLASSIFICATION
            else protocol.LINEAR_REGRESSOR
        )
    return (
        protocol.NONLINEAR_CLASSIFIER
        if task_type == protocol.TASK_CLASSIFICATION
        else protocol.NONLINEAR_REGRESSOR
    )


__all__ = ["build_pipeline", "hyperparameter_grid", "model_name", "scaling_for"]
