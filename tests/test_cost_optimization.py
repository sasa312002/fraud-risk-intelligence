import numpy as np
import pytest
from src.cost_optimization import (
    CostAssumptions, expected_cost_by_action, optimize_thresholds,
    policy_actions, summarize_policy, cost_sensitivity,
)


def test_actions_and_costs_are_well_defined():
    p = np.array([0.01, 0.18, 0.91])
    c = CostAssumptions(max_review_rate=1, max_block_rate=0)
    actions = policy_actions(p, .1, .7)
    assert actions.tolist() == ["APPROVE", "REVIEW", "BLOCK"]
    expected = expected_cost_by_action(p, c)
    assert np.allclose(expected["APPROVE"], p * c.fraud_approved_loss)
    assert (expected["REVIEW"] >= c.manual_review_cost).all()
    assert summarize_policy(p, actions, c)["transactions"] == 3


def test_optimizer_respects_capacities_and_never_needs_labels():
    p = np.linspace(.01, .9, 100)
    c = CostAssumptions(max_review_rate=.20, max_block_rate=.05)
    best, grid = optimize_thresholds(p, c)
    assert len(grid) > 0
    assert best["review_rate"] <= .20 + 1e-12
    assert best["block_rate"] <= .05 + 1e-12
    assert best["review_threshold"] <= best["block_threshold"]
    assert best["projected_cost_total"] <= summarize_policy(
        p, np.full(len(p), "APPROVE"), c
    )["projected_cost_total"]


def test_outcome_cost_counts_and_input_validation():
    c = CostAssumptions()
    p = [.01, .2, .8, .8]
    actions = policy_actions(p, .1, .7)
    result = summarize_policy(p, actions, c, y_true=[0, 1, 1, 0])
    assert result["fraud_review_count"] == 1
    assert result["fraud_blocked_count"] == 1
    assert result["legitimate_blocked_count"] == 1
    assert result["validation_outcome_cost_total"] > 0
    with pytest.raises(ValueError):
        policy_actions([1.5], .1, .3)
    with pytest.raises(ValueError):
        policy_actions([.5], .8, .1)


def test_cost_sensitivity_is_computed():
    p = np.linspace(.001, .8, 150)
    out = cost_sensitivity(p, CostAssumptions(), fraud_losses=(20_000,),
                           block_losses=(1_500,), review_detection_rates=(.9,))
    assert len(out) == 1
    assert 0 <= out.iloc[0]["review_rate"] <= .20
