import numpy as np
import pandas as pd
import pytest
from src.robustness_testing import SCENARIOS, apply_stress_scenario, evaluate_stress_scenarios
from src.cost_optimization import CostAssumptions


@pytest.fixture
def X():
    rng = np.random.default_rng(123)
    n = 60
    return pd.DataFrame({
        'transaction_amount': rng.uniform(20, 150, n),
        'distance_from_usual_location_km': rng.uniform(0, 90, n),
        'merchant_risk_score': rng.uniform(0, 1, n),
        'ip_risk_score': rng.uniform(0, 1, n),
        'transactions_last_1h': rng.integers(0, 6, n),
        'authentication_level': rng.choice(['Standard', 'Strong'], n),
        'channel': rng.choice(['POS', 'E-commerce'], n),
    }, index=pd.Index(range(500, 560)))


class DummyModel:
    def predict_proba(self, X):
        p = (0.05 + .10 * np.asarray(X['merchant_risk_score'].fillna(.5)) +
             .07 * np.asarray(X['ip_risk_score'].fillna(.5)) +
             .012 * np.asarray(X['transactions_last_1h'].fillna(2)) +
             .0004 * np.asarray(X['transaction_amount'].fillna(50)))
        p = np.clip(p, 0.001, .99)
        return np.c_[1-p, p]


def test_scenarios_deterministic_preserve_schema_and_input(X):
    original = X.copy(deep=True)
    for name in SCENARIOS:
        first = apply_stress_scenario(X, name, seed=42)
        second = apply_stress_scenario(X, name, seed=42)
        pd.testing.assert_frame_equal(first, second)
        assert first.index.equals(X.index)
        assert list(first.columns) == list(X.columns)
    pd.testing.assert_frame_equal(X, original)


def test_perturbations_actually_change_values_and_reject_leakage(X):
    for name in SCENARIOS:
        if name != 'baseline':
            assert not apply_stress_scenario(X, name).equals(X), name
    with pytest.raises(ValueError, match='Target/ID'):
        apply_stress_scenario(X.assign(fraud=0), 'baseline')
    with pytest.raises(ValueError, match='Unknown scenario'):
        apply_stress_scenario(X, 'bogus')


def test_fixed_policy_stress_report_and_baseline(X):
    model = DummyModel()
    y = np.r_[np.zeros(45, dtype=int), np.ones(15, dtype=int)]
    summary = evaluate_stress_scenarios(
        model, X, .17, .25, CostAssumptions(), y_validation=y,
    )
    assert len(summary) == len(SCENARIOS)
    assert summary.iloc[0].scenario == 'baseline'
    assert summary.iloc[0].mean_abs_score_change == 0
    assert summary.iloc[0].action_flip_rate == 0
    assert summary['validation_ap_diagnostic'].notna().all()
    assert summary['review_rate'].between(0, 1).all()
    assert summary['block_rate'].between(0, 1).all()
    assert summary['projected_cost_per_transaction'].ge(0).all()


def test_no_refitting_and_capacity_exceedance_reported(X):
    class AlwaysHigh:
        def predict_proba(self, data):
            p = np.full(len(data), .9)
            return np.c_[1-p, p]
    summary = evaluate_stress_scenarios(AlwaysHigh(), X, .12, .20, CostAssumptions())
    assert summary.capacity_block_exceeded.all()
    assert not summary.capacity_review_exceeded.any()
    assert 'validation_ap_diagnostic' not in summary


def test_psi_is_small_for_matching_distribution_and_larger_for_shift():
    from src.robustness_testing import population_stability_index
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 2000)
    same = rng.normal(0, 1, 2000)
    shifted = rng.normal(3, 1, 2000)
    assert population_stability_index(ref, same) < .15
    assert population_stability_index(ref, shifted) > 1.0
    assert np.isnan(population_stability_index([np.nan], [1]))
