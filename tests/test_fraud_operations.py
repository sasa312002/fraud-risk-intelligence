import numpy as np
import pandas as pd
import pytest

from src.cost_optimization import CostAssumptions
from src.fraud_operations import (
    ReviewCapacity, build_casebook, operations_summary,
    fixed_policy_bootstrap_costs,
)


def test_decisions_and_queue_do_not_depend_on_truth():
    p = np.array([.02, .20, .25, .70, .17, .27])
    cases = build_casebook(list(range(6)), p, .15, .5,
                           ReviewCapacity(analyst_count=2,
                                          reviews_per_analyst_per_batch=1))
    assert list(cases.action) == ['APPROVE', 'REVIEW', 'REVIEW', 'BLOCK', 'REVIEW', 'REVIEW']
    assert (cases.operational_status == 'REVIEW_ASSIGNED').sum() == 2
    assert (cases.operational_status == 'PENDING_REVIEW_BACKLOG').sum() == 2
    assert cases.reviewer_id.notna().sum() == 2
    assert cases.loc[cases.action == 'APPROVE', 'reviewer_id'].isna().all()
    assert cases.loc[cases.action == 'BLOCK', 'reviewer_id'].isna().all()


def test_operations_summary_cost_and_capacity():
    p = [0.05, .19, .25, .9]
    cases = build_casebook(['a','b','c','d'], p, .15, .5, ReviewCapacity(1, 2))
    s = operations_summary(cases, CostAssumptions(), actual_fraud=[0, 1, 0, 1])
    assert (s['approve_count'], s['review_count'], s['block_count']) == (1,2,1)
    assert s['review_backlog_count'] == 0
    assert s['fraud_blocked_count'] == 1
    assert s['estimated_fraud_prevented'] == pytest.approx(1.9)


def test_invalid_input_and_bootstrap_determinism():
    with pytest.raises(ValueError):
        build_casebook(['x', 'x'], [.2, .3], .1, .5)
    c = build_casebook(['a','b','c','d'], [.10,.20,.4,.9], .15, .5)
    a = fixed_policy_bootstrap_costs(c, CostAssumptions(), [0,1,0,1], seed=4, n_boot=40)
    b = fixed_policy_bootstrap_costs(c, CostAssumptions(), [0,1,0,1], seed=4, n_boot=40)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        fixed_policy_bootstrap_costs(c, CostAssumptions(), [0,1], n_boot=3)
