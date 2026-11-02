import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.probability_calibration import (
    make_calibrator, expected_calibration_error, evaluate_probabilities,
)


def test_probability_calibrator_sigmoid_and_isotonic():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"feature": rng.normal(size=120), "other": rng.normal(size=120)})
    y = ((X["feature"] + .5 * X["other"] + rng.normal(size=120)) > .5).astype(int)
    base = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    for method in ("sigmoid", "isotonic"):
        calibrated = make_calibrator(base, method, seed=42, n_splits=3).fit(X, y)
        p = calibrated.predict_proba(X.iloc[:12])[:, 1]
        assert len(p) == 12 and np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()


def test_ece_perfect_binary_predictions():
    assert expected_calibration_error([0, 1, 0, 1], [0., 1., 0., 1.]) == 0
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [1.5, -.5])


def test_probability_metrics_sane():
    metrics = evaluate_probabilities([0, 1, 0, 1], [0.05, 0.9, 0.1, 0.95])
    assert metrics["average_precision"] == 1.0
    assert metrics["brier"] < 0.02
    assert metrics["ece_10"] >= 0
