"""Leakage-aware, stateless transaction feature engineering."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create fraud features from attributes known at authorization time.

    Does not impute, encode, learn quantile thresholds, or use the label.
    The input must already exclude transaction_id and fraud.
    """

    REQUIRED_COLUMNS = (
        "transaction_amount",
        "distance_from_usual_location_km",
        "transactions_last_1h",
        "transactions_last_24h",
        "merchant_risk_score",
        "ip_risk_score",
        "new_device",
        "authentication_level",
        "channel",
        "account_age_days",
        "device_age_days",
        "failed_logins_24h",
        "billing_address_mismatch",
        "hour",
    )

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FraudFeatureEngineer expects a pandas DataFrame")
        forbidden = {"fraud", "transaction_id"} & set(X.columns)
        if forbidden:
            raise ValueError(f"Remove target/identifier before transforming: {sorted(forbidden)}")
        missing = sorted(set(self.REQUIRED_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def fit(self, X, y=None):
        self._validate(X)
        return self

    def transform(self, X):
        self._validate(X)
        result = X.copy()

        amount = result["transaction_amount"]
        distance = result["distance_from_usual_location_km"]
        hour_count = result["transactions_last_1h"]
        day_count = result["transactions_last_24h"]
        merchant_risk = result["merchant_risk_score"]
        ip_risk = result["ip_risk_score"]

        result["amount_log1p"] = np.log1p(amount)
        result["distance_log1p"] = np.log1p(distance)
        result["velocity_ratio_1h_24h"] = hour_count / (day_count + 1.0)
        result["hourly_velocity_intensity"] = hour_count / (day_count / 24.0 + 1.0)
        result["high_velocity_indicator"] = (hour_count >= 5).astype("int8")
        result["amount_per_recent_transaction"] = amount / (day_count + 1.0)

        # Keep missing numeric inputs missing. Imputation comes later in the CV pipeline.
        result["combined_risk_score"] = merchant_risk + ip_risk
        result["risk_score_product"] = merchant_risk * ip_risk
        result["amount_merchant_risk_interaction"] = amount * merchant_risk
        result["new_device_distance_interaction"] = result["new_device"] * distance
        result["failed_login_ip_risk_interaction"] = result["failed_logins_24h"] * ip_risk
        result["billing_new_device_interaction"] = (
            result["billing_address_mismatch"] * result["new_device"]
        )

        result["authentication_missing_indicator"] = (
            result["authentication_level"].isna().astype("int8")
        )
        auth = result["authentication_level"].astype("string").fillna("Missing")
        channel = result["channel"].astype("string").fillna("Missing")
        result["authentication_channel_interaction"] = (auth + "__" + channel).astype(str)

        result["device_account_age_ratio"] = (
            result["device_age_days"] / (result["account_age_days"] + 1.0)
        )
        result["night_transaction_indicator"] = (
            (result["hour"] >= 22) | (result["hour"] < 6)
        ).astype("int8")

        return result


ENGINEERED_FEATURES = (
    "amount_log1p", "distance_log1p", "velocity_ratio_1h_24h",
    "hourly_velocity_intensity", "high_velocity_indicator",
    "amount_per_recent_transaction", "combined_risk_score",
    "risk_score_product", "amount_merchant_risk_interaction",
    "new_device_distance_interaction", "failed_login_ip_risk_interaction",
    "billing_new_device_interaction", "authentication_missing_indicator",
    "authentication_channel_interaction", "device_account_age_ratio",
    "night_transaction_indicator",
)
