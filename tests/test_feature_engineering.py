import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import FraudFeatureEngineer, ENGINEERED_FEATURES

@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        "transaction_amount": [100.0, np.nan],
        "distance_from_usual_location_km": [50.0, 0.0],
        "transactions_last_1h": [5, 0],
        "transactions_last_24h": [10, 0],
        "merchant_risk_score": [0.8, np.nan],
        "ip_risk_score": [0.3, 0.2],
        "new_device": [1, 0],
        "authentication_level": [np.nan, "Strong"],
        "channel": ["E-commerce", "POS"],
        "account_age_days": [100, 200],
        "device_age_days": [10, 20],
        "failed_logins_24h": [2, 0],
        "billing_address_mismatch": [1, 0],
        "hour": [23, 13],
    })


def test_engineered_columns_and_no_input_mutation(sample_transactions):
    original = sample_transactions.copy(deep=True)
    result = FraudFeatureEngineer().fit_transform(sample_transactions)
    assert set(ENGINEERED_FEATURES).issubset(result.columns)
    assert len(ENGINEERED_FEATURES) >= 10
    pd.testing.assert_frame_equal(sample_transactions, original)
    assert result.shape[0] == original.shape[0]


def test_math_and_missingness(sample_transactions):
    result = FraudFeatureEngineer().fit_transform(sample_transactions)
    assert np.isclose(result.loc[0, "velocity_ratio_1h_24h"], 5/11)
    assert result.loc[1, "velocity_ratio_1h_24h"] == 0
    assert result.loc[0, "authentication_missing_indicator"] == 1
    assert result.loc[1, "authentication_missing_indicator"] == 0
    assert result.loc[0, "authentication_channel_interaction"] == "Missing__E-commerce"
    assert result.loc[0, "night_transaction_indicator"] == 1
    assert result.loc[1, "night_transaction_indicator"] == 0
    assert pd.isna(result.loc[1, "combined_risk_score"])
    assert pd.isna(result.loc[1, "amount_log1p"])


def test_target_identifier_rejected(sample_transactions):
    for forbidden in ("fraud", "transaction_id"):
        with pytest.raises(ValueError):
            FraudFeatureEngineer().fit_transform(sample_transactions.assign(**{forbidden: 0}))
