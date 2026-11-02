"""Section J: label-free input perturbations + fixed-model / fixed-policy stress audit.

No model fitting, threshold re-optimization, or reading of test data takes place.
If y_validation is supplied, label-based metrics are diagnostics on the
existing validation sample only, not deployable stress-test guarantees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from src.cost_optimization import policy_actions, summarize_policy

SCENARIOS = (
    "baseline",
    "numeric_missing_10pct",
    "numeric_missing_25pct",
    "authentication_missing_20pct",
    "unseen_channel_10pct",
    "amount_up_25pct",
    "distance_up_50pct",
    "velocity_plus2",
    "risk_scores_noise_0p05",
)

NUMERIC_MISSING_COLS = (
    "transaction_amount", "distance_from_usual_location_km",
    "merchant_risk_score", "ip_risk_score",
)


def _random_nonnull_rows(frame: pd.DataFrame, col: str, share: float, rng):
    if col not in frame.columns:
        raise ValueError(f"Missing required feature: {col}")
    candidate = np.flatnonzero(frame[col].notna().to_numpy())
    n = int(round(len(frame) * share))
    n = min(n, len(candidate))
    if n <= 0:
        return np.array([], dtype=int)
    return rng.choice(candidate, size=n, replace=False)


def apply_stress_scenario(X: pd.DataFrame, scenario: str, seed: int = 42) -> pd.DataFrame:
    """Return a new DataFrame with a deterministic, explicitly defined perturbation.

    Neither target nor transaction_id may be passed in. The original frame,
    original column schema, row order and index are preserved.
    """
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a nonempty pandas DataFrame")
    if {'fraud', 'transaction_id'} & set(X.columns):
        raise ValueError("Target/ID must be excluded from model inputs")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    result = X.copy(deep=True)
    rng = np.random.default_rng(seed)

    if scenario in {'numeric_missing_10pct', 'numeric_missing_25pct'}:
        share = .10 if scenario == 'numeric_missing_10pct' else .25
        for col in NUMERIC_MISSING_COLS:
            rows = _random_nonnull_rows(result, col, share, rng)
            result.iloc[rows, result.columns.get_loc(col)] = np.nan
    elif scenario == 'authentication_missing_20pct':
        col = 'authentication_level'
        rows = _random_nonnull_rows(result, col, .20, rng)
        result.iloc[rows, result.columns.get_loc(col)] = np.nan
    elif scenario == 'unseen_channel_10pct':
        col = 'channel'
        rows = _random_nonnull_rows(result, col, .10, rng)
        result.iloc[rows, result.columns.get_loc(col)] = '__UNSEEN_STRESS_CHANNEL__'
    elif scenario == 'amount_up_25pct':
        col = 'transaction_amount'
        if col not in result.columns:
            raise ValueError(f"Missing required feature: {col}")
        result[col] = result[col] * 1.25
    elif scenario == 'distance_up_50pct':
        col = 'distance_from_usual_location_km'
        if col not in result.columns:
            raise ValueError(f"Missing required feature: {col}")
        result[col] = result[col] * 1.5
    elif scenario == 'velocity_plus2':
        col = 'transactions_last_1h'
        if col not in result.columns:
            raise ValueError(f"Missing required feature: {col}")
        result[col] = result[col] + 2
    elif scenario == 'risk_scores_noise_0p05':
        for col in ('merchant_risk_score', 'ip_risk_score'):
            if col not in result.columns:
                raise ValueError(f"Missing required feature: {col}")
            result[col] = (result[col] + rng.normal(0, .05, len(result))).clip(0, 1)

    assert result.shape == X.shape and result.index.equals(X.index)
    return result


def evaluate_stress_scenarios(model, X_validation: pd.DataFrame, review_threshold: float,
                              block_threshold: float, costs, y_validation=None,
                              scenarios=SCENARIOS, seed=42) -> pd.DataFrame:
    """Score fixed model/thresholds under perturbation; never refit or retune.

    Changes in labels after counterfactual feature edits are unknown; label-
    based metrics are shown *only* as stability diagnostics on validation.
    Capacity may be exceeded under stress: it is reported, not silently fixed.
    """
    if 'baseline' not in scenarios:
        raise ValueError("Scenarios must include baseline")
    if not (0 <= review_threshold <= block_threshold <= 1):
        raise ValueError("Invalid thresholds")
    baseline_p = np.asarray(model.predict_proba(X_validation)[:, 1], dtype=float)
    if baseline_p.ndim != 1 or len(baseline_p) != len(X_validation):
        raise ValueError('Unexpected model probability output shape')
    if not np.isfinite(baseline_p).all() or not ((0 <= baseline_p) & (baseline_p <= 1)).all():
        raise ValueError('Invalid model probabilities')
    baseline_actions = policy_actions(baseline_p, review_threshold, block_threshold)
    if y_validation is not None:
        y_validation = np.asarray(y_validation)
        if y_validation.shape != baseline_p.shape or not np.isin(y_validation, [0, 1]).all():
            raise ValueError('y_validation must be aligned binary validation outcomes')
    rows = []
    for scenario in scenarios:
        stressed = apply_stress_scenario(X_validation, scenario, seed)
        p = np.asarray(model.predict_proba(stressed)[:, 1], dtype=float)
        if p.shape != baseline_p.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError(f'Invalid model probabilities for scenario {scenario}')
        actions = policy_actions(p, review_threshold, block_threshold)
        summary = summarize_policy(p, actions, costs)
        row = {
            'scenario': scenario,
            'n_transactions': len(p),
            'mean_probability': float(p.mean()),
            'mean_abs_score_change': float(np.abs(p - baseline_p).mean()),
            'p95_abs_score_change': float(np.quantile(np.abs(p - baseline_p), .95)),
            'action_flip_rate': float((actions != baseline_actions).mean()),
            'review_rate': summary['review_rate'],
            'block_rate': summary['block_rate'],
            'capacity_review_exceeded': bool(summary['review_rate'] > costs.max_review_rate + 1e-10),
            'capacity_block_exceeded': bool(summary['block_rate'] > costs.max_block_rate + 1e-10),
            'projected_cost_per_transaction': summary['projected_cost_per_transaction'],
            'projected_cost_total': summary['projected_cost_total'],
        }
        if y_validation is not None:
            row['validation_ap_diagnostic'] = float(average_precision_score(y_validation, p))
            row['validation_roc_auc_diagnostic'] = float(roc_auc_score(y_validation, p))
            row['validation_brier_diagnostic'] = float(brier_score_loss(y_validation, p))
        rows.append(row)
    return pd.DataFrame(rows)


def population_stability_index(train, comparison, n_bins=10):
    """Training-quantile PSI, including a missingness bin.

    Descriptive only. Not a hypothesis test or evidence of live temporal drift.
    """
    if n_bins < 2:
        raise ValueError('n_bins must be >= 2')
    ref = np.asarray(train, dtype=float)
    cur = np.asarray(comparison, dtype=float)
    ref_valid = ref[np.isfinite(ref)]
    cur_valid = cur[np.isfinite(cur)]
    if not len(ref_valid) or not len(cur_valid):
        return np.nan
    cutpoints = np.unique(np.quantile(ref_valid, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = np.r_[-np.inf, cutpoints, np.inf]
    ref_freq = np.r_[np.histogram(ref_valid, bins=edges)[0], len(ref) - len(ref_valid)]
    cur_freq = np.r_[np.histogram(cur_valid, bins=edges)[0], len(cur) - len(cur_valid)]
    # Add tiny mass to avoid division by zero when categories have zero samples.
    ref_dist = (ref_freq + 1e-6) / (ref_freq.sum() + 1e-6 * len(ref_freq))
    cur_dist = (cur_freq + 1e-6) / (cur_freq.sum() + 1e-6 * len(cur_freq))
    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))
