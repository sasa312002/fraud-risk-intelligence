import numpy as np
import pandas as pd
import pytest
from src.explainability import explain_logistic_margins, local_reason_table
from src.model_training import build_fraud_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


def sample_calibrator():
    rng = np.random.default_rng(42)
    n = 96
    X = pd.DataFrame({
        'transaction_amount': rng.uniform(5, 300, n),
        'new_device': rng.integers(0, 2, n),
        'channel': rng.choice(['POS', 'E-commerce'], n),
    })
    y = (X['transaction_amount'].to_numpy() + 85 * X['new_device'].to_numpy() + rng.normal(0, 60, n) > 175).astype(int)
    pipe = build_fraud_pipeline(X, LogisticRegression(max_iter=1000), engineered=False)
    cal = CalibratedClassifierCV(
        estimator=pipe, method='sigmoid',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    ).fit(X, y)
    return X, cal


def test_shap_grouped_values_reconstruct_mean_base_margin():
    X, cal = sample_calibrator()
    result = explain_logistic_margins(cal, X.iloc[:48], X.iloc[48:54])
    assert result.shap_values.shape == (6, 3)
    assert result.fold_count == 3
    assert result.max_residual < 1e-5
    assert np.allclose(result.base_values + result.shap_values.sum(axis=1), result.mean_margin, atol=1e-5)
    assert set(result.global_importance.feature) == set(X.columns)


def test_shap_local_reasons_have_metadata_and_directions():
    X, cal = sample_calibrator()
    result = explain_logistic_margins(cal, X.iloc[:48], X.iloc[48:50])
    p = cal.predict_proba(X.iloc[48:50])[:, 1]
    table = local_reason_table(result, ['TX1', 'TX2'], p, ['REVIEW', 'BLOCK'], top_k=2)
    assert table.shape[0] == 4
    assert set(table.base_risk_direction) <= {'RAISES', 'LOWERS', 'NEUTRAL'}
    assert np.allclose(table.groupby('transaction_id').calibrated_fraud_probability.first().to_numpy(), p)


def test_rejects_leakage_and_mismatched_columns():
    X, cal = sample_calibrator()
    with pytest.raises(ValueError, match='Target/ID'):
        explain_logistic_margins(cal, X.assign(fraud=0), X.iloc[:2].assign(fraud=0))
    with pytest.raises(ValueError, match='identical raw feature'):
        explain_logistic_margins(cal, X, X.iloc[:3].drop(columns='channel'))
