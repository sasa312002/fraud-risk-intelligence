"""Tests for class weighting and fold-safe oversampling pipelines."""
import numpy as np
import pandas as pd
import pytest
from src.imbalance_learning import build_imbalance_pipeline, VALID_STRATEGIES


def sample_data():
    X = pd.DataFrame({
        "transaction_amount": [10, 12, 14, 11, 13, 15, 8, 9, 100, 120],
        "merchant_risk_score": [0.1, 0.2, 0.3, 0.2, 0.4, 0.5, 0.1, np.nan, 0.8, 0.9],
        "channel": ["POS", "POS", "ATM", "POS", "ATM", "POS", "POS", "ATM", "E-commerce", "Mobile App"],
    })
    y = pd.Series([0]*8 + [1]*2)
    return X, y


def test_all_strategies_fit_predict_and_preserve_input():
    X, y = sample_data()
    original = X.copy(deep=True)
    for strategy in VALID_STRATEGIES:
        model = build_imbalance_pipeline(X, strategy)
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        assert proba.shape == (len(X),)
        assert np.isfinite(proba).all()
        assert ((0 <= proba) & (proba <= 1)).all()
    pd.testing.assert_frame_equal(X, original)


def test_oversampling_only_occurs_during_fit():
    X, y = sample_data()
    model = build_imbalance_pipeline(X, "Random oversampling")
    assert "sampler" in model.named_steps
    model.fit(X, y)
    assert len(model.named_steps["sampler"].sample_indices_) == 16
    unseen = pd.DataFrame({"transaction_amount": [5.0], "merchant_risk_score": [0.4], "channel": ["New Category"]})
    assert model.predict_proba(unseen).shape == (1, 2)


def test_invalid_strategy_and_target_id_rejected():
    X, _ = sample_data()
    with pytest.raises(ValueError):
        build_imbalance_pipeline(X, "SMOTE")
    with pytest.raises(ValueError):
        build_imbalance_pipeline(X.assign(fraud=0), "Class-weighted")
    with pytest.raises(ValueError):
        build_imbalance_pipeline(X.assign(transaction_id="T0"), "Unweighted")
