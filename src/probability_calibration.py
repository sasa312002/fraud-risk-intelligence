"""Leakage-aware probability calibration for the fraud risk project.

Base pipelines include all learned preprocessing and (if used) any resampling.
Cross-fitting produces calibration targets using held-out training folds that
retain the natural prevalence; the external validation and test sets remain
untouched during fitting.
"""
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def make_calibrator(base_estimator, method, seed=42, n_splits=3):
    if method not in ("sigmoid", "isotonic"):
        raise ValueError("method must be 'sigmoid' or 'isotonic'")
    if n_splits < 2:
        raise ValueError("at least two folds are required")
    return CalibratedClassifierCV(
        estimator=clone(base_estimator),
        method=method,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        ensemble=True,
    )


def expected_calibration_error(y_true, probabilities, n_bins=10):
    """Frequency-weighted absolute gap between mean prediction and outcome.

    Equal-width bins are used; includes p=1 in the last interval.
    ECE is a diagnostic with sample-size and binning sensitivity.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("y_true and probabilities must be nonempty equal-length vectors")
    if n_bins < 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid n_bins or probabilities")
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1]), n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        inside = bins == b
        if inside.any():
            error += inside.mean() * abs(y[inside].mean() - p[inside].mean())
    return float(error)


def evaluate_probabilities(y_true, probabilities):
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    if not np.isfinite(p).all():
        raise ValueError("nonfinite model probabilities")
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "ece_10": expected_calibration_error(y, p, n_bins=10),
        "mean_predicted_fraud": float(p.mean()),
        "observed_fraud": float(y.mean()),
    }
