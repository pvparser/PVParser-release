"""Model wrappers and evaluation metrics for the RAID 2024 reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SUPPORTED_ALGORITHMS = ("envelope", "iforest", "ocsvm")
SUPPORTED_SCALING = ("none", "standard")


@dataclass(frozen=True)
class EvaluationMetrics:
    """Paper-aligned metrics plus time-window confusion counts."""

    tacc: float
    vacc: float
    bacc: float
    train_true_normal: int
    train_false_alarm: int
    valid_true_normal: int
    valid_false_alarm: int
    valid_true_attack: int
    valid_missed_attack: int
    sample_tp: int
    sample_fp: int
    sample_fn: int
    sample_tn: int
    sample_precision: float
    sample_recall: float
    sample_tnr: float
    sample_fpr: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_metrics(
    train_predictions: np.ndarray,
    valid_predictions: np.ndarray,
    valid_labels: np.ndarray,
) -> EvaluationMetrics:
    """Compute the paper's tacc, vacc, and bacc metrics.

    Predictions are expected as binary integers with `0` for normal and `1` for
    anomaly/outlier.
    """

    train_predictions = np.asarray(train_predictions, dtype=np.int8)
    valid_predictions = np.asarray(valid_predictions, dtype=np.int8)
    valid_labels = np.asarray(valid_labels, dtype=np.int8)

    train_true_normal = int(np.sum(train_predictions == 0))
    train_false_alarm = int(np.sum(train_predictions == 1))
    tacc = _safe_ratio(train_true_normal, train_true_normal + train_false_alarm)

    valid_normal_mask = valid_labels == 0
    valid_attack_mask = valid_labels == 1

    valid_true_normal = int(np.sum((valid_predictions == 0) & valid_normal_mask))
    valid_false_alarm = int(np.sum((valid_predictions == 1) & valid_normal_mask))
    valid_true_attack = int(np.sum((valid_predictions == 1) & valid_attack_mask))
    valid_missed_attack = int(np.sum((valid_predictions == 0) & valid_attack_mask))

    valid_normal_acc = _safe_ratio(valid_true_normal, valid_true_normal + valid_false_alarm)
    valid_attack_acc = _safe_ratio(valid_true_attack, valid_true_attack + valid_missed_attack)
    vacc = 0.5 * (valid_normal_acc + valid_attack_acc)
    bacc = 0.5 * (tacc + vacc)

    sample_tp = valid_true_attack
    sample_fp = valid_false_alarm
    sample_fn = valid_missed_attack
    sample_tn = valid_true_normal
    sample_precision = _safe_ratio(sample_tp, sample_tp + sample_fp)
    sample_recall = _safe_ratio(sample_tp, sample_tp + sample_fn)
    sample_tnr = _safe_ratio(sample_tn, sample_tn + sample_fp)
    sample_fpr = _safe_ratio(sample_fp, sample_fp + sample_tn)

    return EvaluationMetrics(
        tacc=tacc,
        vacc=vacc,
        bacc=bacc,
        train_true_normal=train_true_normal,
        train_false_alarm=train_false_alarm,
        valid_true_normal=valid_true_normal,
        valid_false_alarm=valid_false_alarm,
        valid_true_attack=valid_true_attack,
        valid_missed_attack=valid_missed_attack,
        sample_tp=sample_tp,
        sample_fp=sample_fp,
        sample_fn=sample_fn,
        sample_tn=sample_tn,
        sample_precision=sample_precision,
        sample_recall=sample_recall,
        sample_tnr=sample_tnr,
        sample_fpr=sample_fpr,
    )


def _base_estimator(algorithm: str, params: dict[str, Any], random_state: int) -> Any:
    if algorithm == "envelope":
        return EllipticEnvelope(
            contamination=float(params["contamination"]),
            support_fraction=params["support_fraction"],
            assume_centered=False,
            random_state=random_state,
        )
    if algorithm == "iforest":
        return IsolationForest(
            n_estimators=int(params["n_estimators"]),
            max_samples=params["max_samples"],
            contamination=float(params["contamination"]),
            random_state=random_state,
            n_jobs=-1,
        )
    if algorithm == "ocsvm":
        return OneClassSVM(
            kernel=str(params["kernel"]),
            gamma=params["gamma"],
            nu=float(params["nu"]),
        )
    raise ValueError(f"Unsupported algorithm: {algorithm!r}")


def build_estimator(
    algorithm: str,
    params: dict[str, Any],
    *,
    scaling: str = "standard",
    random_state: int = 42,
) -> Any:
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm!r}")
    if scaling not in SUPPORTED_SCALING:
        raise ValueError(f"Unsupported scaling: {scaling!r}")

    estimator = _base_estimator(algorithm, params=params, random_state=random_state)
    if scaling == "standard":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    return estimator


def effective_fit_params(
    algorithm: str,
    params: dict[str, Any],
    *,
    sample_count: int,
) -> dict[str, Any]:
    """Resolve fit-time parameters that depend on the available sample count."""

    resolved = dict(params)
    if algorithm == "iforest" and isinstance(resolved.get("max_samples"), int):
        resolved["max_samples"] = min(int(resolved["max_samples"]), max(1, sample_count))
    return resolved


def predict_outliers(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Convert sklearn one-class predictions to 0=normal / 1=anomaly."""

    raw = estimator.predict(features)
    raw = np.asarray(raw)
    return np.where(raw == -1, 1, 0).astype(np.int8)


def parameter_grid(algorithm: str, grid_size: str = "small") -> Iterable[dict[str, Any]]:
    """Return a modest hyperparameter grid for the paper's classical models."""

    if grid_size not in {"tiny", "small"}:
        raise ValueError(f"Unsupported grid size: {grid_size!r}")

    if algorithm == "envelope":
        contamination_values = [0.001, 0.005] if grid_size == "tiny" else [0.001, 0.005, 0.01]
        support_fraction_values = [0.95, None] if grid_size == "tiny" else [0.90, 0.95, None]
        for contamination, support_fraction in product(contamination_values, support_fraction_values):
            yield {
                "contamination": contamination,
                "support_fraction": support_fraction,
            }
        return

    if algorithm == "iforest":
        contamination_values = [0.001, 0.005] if grid_size == "tiny" else [0.001, 0.005, 0.01]
        max_samples_values = [256, "auto"] if grid_size == "tiny" else [256, 512, "auto"]
        n_estimators_values = [100] if grid_size == "tiny" else [100, 200]
        for n_estimators, max_samples, contamination in product(
            n_estimators_values,
            max_samples_values,
            contamination_values,
        ):
            yield {
                "n_estimators": n_estimators,
                "max_samples": max_samples,
                "contamination": contamination,
            }
        return

    if algorithm == "ocsvm":
        nu_values = [0.001, 0.005] if grid_size == "tiny" else [0.001, 0.005, 0.01]
        gamma_values = ["scale"] if grid_size == "tiny" else ["scale", "auto"]
        kernel_values = ["rbf"] if grid_size == "tiny" else ["rbf", "sigmoid"]
        for kernel, gamma, nu in product(kernel_values, gamma_values, nu_values):
            yield {
                "kernel": kernel,
                "gamma": gamma,
                "nu": nu,
            }
        return

    raise ValueError(f"Unsupported algorithm: {algorithm!r}")
