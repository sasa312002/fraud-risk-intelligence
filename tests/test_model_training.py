"""Lightweight tests for Section D pipeline assembly and leakage prevention."""
import numpy as np
import pandas as pd
import pytest
from src.model_training import build_fraud_pipeline, make_logistic, make_random_forest


def sample_frame():
    return pd.DataFrame({
        "transaction_amount": [10.0, np.nan, 100.0, 50.0],
        "distance_from_usual_location_km": [1.0, 2.0, np.nan, 4.0],
        "transactions_last_1h": [0, 1, 3, 2],
        "transactions_last_24h": [1, 3, 8, 4],
        "merchant_risk_score": [0.1, 0.2, 0.7, 0.3],
        "ip_risk_score": [0.1, 0.5, 0.2, 0.3],
        "new_device": [0, 1, 0, 1],
        "authentication_level": ["Strong", None, "Standard", "Strong"],
        "channel": ["POS", "E-commerce", "ATM", "Mobile App"],
        "account_age_days": [100, 200, 300, 50],
        "device_age_days": [10, 20, 40, 5],
        "failed_logins_24h": [0, 1, 2, 0],
        "billing_address_mismatch": [0, 1, 0, 0],
        "hour": [10, 23, 6, 12],
    })


def test_raw_and_engineered_pipelines_can_fit_and_predict():
    X = sample_frame()
    y = pd.Series([0, 0, 1, 1])
    original_copy = X.copy(deep=True)
    for engineered in [False, True]:
        pipe = build_fraud_pipeline(X, make_logistic(), engineered=engineered)
        pipe.fit(X, y)
        scores = pipe.predict_proba(X)[:, 1]
        assert scores.shape == (4,)
        assert np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()
    pd.testing.assert_frame_equal(X, original_copy)


def test_refuses_target_and_id():
    X = sample_frame()
    X["fraud"] = 0
    with pytest.raises(ValueError, match="Remove target"):
        build_fraud_pipeline(X, make_random_forest())


def test_unseen_category_is_accepted():
    X = sample_frame()
    y = pd.Series([0, 0, 1, 1])
    pipe = build_fraud_pipeline(X, make_logistic(), engineered=True).fit(X, y)
    new = X.iloc[[0]].copy()
    new["channel"] = "UNSEEN_CHANNEL"
    assert np.isfinite(pipe.predict_proba(new)).all()
