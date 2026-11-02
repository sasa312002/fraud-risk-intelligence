"""Offline fraud-operations replay on a fixed, provisional policy.

Uses ONLY scores and transaction IDs to assign actions/review queues.
Actual validation fraud labels, if provided to a separate auditor, are used
only to evaluate the fixed policy after the fact (never for decisions).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cost_optimization import CostAssumptions, policy_actions, summarize_policy


@dataclass(frozen=True)
class ReviewCapacity:
    analyst_count: int = 4
    reviews_per_analyst_per_batch: int = 30

    def validate(self):
        for name in ('analyst_count', 'reviews_per_analyst_per_batch'):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')

    @property
    def total_capacity(self):
        self.validate()
        return int(self.analyst_count * self.reviews_per_analyst_per_batch)


def build_casebook(transaction_ids, probabilities, review_threshold,
                   block_threshold, capacity=ReviewCapacity()):
    """Assign actions and reviewer slots with no label/ground-truth access.

    Priority is descending *calibrated fraud probability* among review cases.
    Each analyst handles <= reviews_per_analyst_per_batch in the simulated
    batch. Overflow remains PENDING_REVIEW_BACKLOG, never auto-approved.
    Without timestamps, 'per batch' is NOT a measured hourly/daily throughput.
    """
    capacity.validate()
    ids = np.asarray(transaction_ids, dtype=object)
    p = np.asarray(probabilities, dtype=float)
    if ids.ndim != 1 or p.ndim != 1 or len(ids) != len(p):
        raise ValueError('Transaction IDs and scores must be aligned 1D vectors')
    if pd.isna(ids).any() or len(pd.unique(ids)) != len(ids):
        raise ValueError('IDs must be nonmissing and unique')
    actions = policy_actions(p, review_threshold, block_threshold)
    cases = pd.DataFrame({
        'transaction_id': ids, 'fraud_probability': p,
        'action': actions,
        'review_priority_rank': pd.Series(pd.NA, index=range(len(p)), dtype='Int64'),
        'reviewer_id': pd.Series(pd.NA, index=range(len(p)), dtype='string'),
        'operational_status': np.where(
            actions == 'BLOCK', 'BLOCK_PROPOSED',
            np.where(actions == 'APPROVE', 'APPROVE_PROPOSED', 'REVIEW_PENDING')
        ),
    })
    review_idx = cases.loc[cases.action == 'REVIEW'].sort_values(
        ['fraud_probability', 'transaction_id'], ascending=[False, True],
        key=lambda s: s.astype(str) if s.name == 'transaction_id' else s,
        kind='stable',
    ).index.to_numpy()
    for rank, idx in enumerate(review_idx, start=1):
        cases.at[idx, 'review_priority_rank'] = rank
        if rank <= capacity.total_capacity:
            analyst_number = ((rank - 1) % capacity.analyst_count) + 1
            cases.at[idx, 'reviewer_id'] = f'ANALYST_{analyst_number:02d}'
            cases.at[idx, 'operational_status'] = 'REVIEW_ASSIGNED'
        else:
            cases.at[idx, 'operational_status'] = 'PENDING_REVIEW_BACKLOG'
    return cases


def operations_summary(cases, costs: CostAssumptions, actual_fraud=None):
    """Summarize score-based decisions; hindsight costs optional and separate."""
    required = {'transaction_id', 'fraud_probability', 'action', 'reviewer_id', 'operational_status'}
    if not required.issubset(cases.columns):
        raise ValueError(f'Missing operational fields: {required - set(cases.columns)}')
    summary = summarize_policy(
        cases.fraud_probability.to_numpy(), cases.action.to_numpy(),
        costs, y_true=actual_fraud,
    )
    review_count = summary['review_count']
    assigned = int((cases.operational_status == 'REVIEW_ASSIGNED').sum())
    backlog = int((cases.operational_status == 'PENDING_REVIEW_BACKLOG').sum())
    summary.update({
        'review_assigned_count': assigned,
        'review_backlog_count': backlog,
        'review_assignment_rate': assigned / review_count if review_count else 1.0,
        'model_review_decisions_are_not_final': True,
    })
    return summary


def fixed_policy_bootstrap_costs(cases, costs, actual_fraud, n_boot=400, seed=42):
    """Bootstrap retrospective outcome-based costs of already-fixed decisions.

    For diagnostics only. Policy and score thresholds are not re-optimized.
    Reviewer effectiveness is assumed. These are not verified savings.
    """
    y = np.asarray(actual_fraud)
    if y.ndim != 1 or len(y) != len(cases) or not np.isin(y, [0, 1]).all():
        raise ValueError('actual_fraud must be aligned binary outcomes')
    if not isinstance(n_boot, int) or n_boot < 2:
        raise ValueError('n_boot must be at least 2')
    rng = np.random.default_rng(seed)
    n = len(cases)
    p = cases.fraud_probability.to_numpy()
    actions = cases.action.to_numpy()
    rows = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        row = summarize_policy(p[ix], actions[ix], costs, y_true=y[ix])
        rows.append(row['validation_outcome_cost_per_transaction'])
    return pd.DataFrame([{
        'bootstrap_replicates': n_boot,
        'cost_per_transaction_2p5': float(np.percentile(rows, 2.5)),
        'cost_per_transaction_median': float(np.median(rows)),
        'cost_per_transaction_97p5': float(np.percentile(rows, 97.5)),
    }])
