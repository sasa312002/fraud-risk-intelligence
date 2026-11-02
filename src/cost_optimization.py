"""Cost-sensitive fraud policy search using calibrated probabilities.

The optimizer uses predicted probabilities and stated business assumptions,
never validation labels. Validation labels are used ONLY for diagnostic
realized (review-outcome-averaged) costs after policy choice.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostAssumptions:
    fraud_approved_loss: float = 20_000.0
    legitimate_blocked_loss: float = 1_500.0
    manual_review_cost: float = 150.0
    review_fraud_detection_rate: float = 0.90
    review_false_rejection_rate: float = 0.02
    max_review_rate: float = 0.20
    max_block_rate: float = 0.05

    def validate(self):
        for field in ("fraud_approved_loss", "legitimate_blocked_loss", "manual_review_cost"):
            if getattr(self, field) < 0 or not np.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be non-negative and finite")
        for field in ("review_fraud_detection_rate", "review_false_rejection_rate", "max_review_rate", "max_block_rate"):
            value = getattr(self, field)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} must be between 0 and 1")
        if self.max_review_rate + self.max_block_rate > 1:
            raise ValueError("Review and block capacities together cannot exceed 100%")


def _probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be a nonempty finite vector in [0,1]")
    return p


def policy_actions(probabilities, review_threshold, block_threshold):
    p = _probabilities(probabilities)
    if not (0 <= review_threshold <= block_threshold <= 1.000001):
        raise ValueError("Require 0 <= review_threshold <= block_threshold <= 1")
    return np.where(p >= block_threshold, "BLOCK", np.where(p >= review_threshold, "REVIEW", "APPROVE"))


def expected_cost_by_action(probabilities, costs):
    """Expected *per-transaction* loss conditional on the calibrated probability.

    Reviewer effectiveness is an assumption, not observed ground truth.
    """
    costs.validate()
    p = _probabilities(probabilities)
    approve = p * costs.fraud_approved_loss
    block = (1-p) * costs.legitimate_blocked_loss
    review = (costs.manual_review_cost
              + p * (1-costs.review_fraud_detection_rate) * costs.fraud_approved_loss
              + (1-p) * costs.review_false_rejection_rate * costs.legitimate_blocked_loss)
    return {"APPROVE": approve, "REVIEW": review, "BLOCK": block}


def summarize_policy(probabilities, actions, costs, y_true=None):
    """Report forecast cost and optional outcome-based expected cost.

    Validation outcomes are used only to audit realized risks for chosen policy;
    reviewer decisions themselves are hypothetical and averaged over assumed rates.
    """
    p = _probabilities(probabilities)
    actions = np.asarray(actions)
    if actions.shape != p.shape or not np.isin(actions, ["APPROVE", "REVIEW", "BLOCK"]).all():
        raise ValueError("Invalid policy actions")
    cost_matrix = expected_cost_by_action(p, costs)
    projected = sum(cost_matrix[a][actions == a].sum() for a in cost_matrix)
    n = len(p)
    result = {"transactions": n, "approve_count": int((actions == "APPROVE").sum()),
              "review_count": int((actions == "REVIEW").sum()),
              "block_count": int((actions == "BLOCK").sum()),
              "approve_rate": float((actions == "APPROVE").mean()),
              "review_rate": float((actions == "REVIEW").mean()),
              "block_rate": float((actions == "BLOCK").mean()),
              "projected_cost_total": float(projected),
              "projected_cost_per_transaction": float(projected/n)}
    if y_true is not None:
        y = np.asarray(y_true)
        if y.shape != p.shape or not np.isin(y, [0, 1]).all():
            raise ValueError("y_true must be aligned binary outcomes")
        approve_fraud = (actions == "APPROVE") & (y == 1)
        block_legitimate = (actions == "BLOCK") & (y == 0)
        review_fraud = (actions == "REVIEW") & (y == 1)
        review_legitimate = (actions == "REVIEW") & (y == 0)
        cost_derived_from_labels = (
            approve_fraud.sum() * costs.fraud_approved_loss
            + block_legitimate.sum() * costs.legitimate_blocked_loss
            + (actions == "REVIEW").sum() * costs.manual_review_cost
            + review_fraud.sum() * (1-costs.review_fraud_detection_rate) * costs.fraud_approved_loss
            + review_legitimate.sum() * costs.review_false_rejection_rate * costs.legitimate_blocked_loss
        )
        result.update({
            "observed_fraud_total": int(y.sum()),
            "fraud_approved_count": int(approve_fraud.sum()),
            "fraud_blocked_count": int(((actions == "BLOCK") & (y == 1)).sum()),
            "fraud_review_count": int(review_fraud.sum()),
            "legitimate_blocked_count": int(block_legitimate.sum()),
            "estimated_fraud_prevented": float(((actions == "BLOCK") & (y == 1)).sum()
                                            + costs.review_fraud_detection_rate * review_fraud.sum()),
            "validation_outcome_cost_total": float(cost_derived_from_labels),
            "validation_outcome_cost_per_transaction": float(cost_derived_from_labels/n),
        })
    return result


def optimize_thresholds(probabilities, costs, quantile_grid=51):
    """Find a cost-minimizing TWO-threshold policy under rate constraints.

    Optimize on predicted calibrated p ONLY (no y labels). Cutpoints are
    determined from the given score distribution (not a fitted transformer).
    """
    p = _probabilities(probabilities)
    costs.validate()
    if not isinstance(quantile_grid, int) or quantile_grid < 3:
        raise ValueError("quantile_grid must be at least 3")
    cuts = np.unique(np.r_[0, np.quantile(p, np.linspace(0, 1, quantile_grid)), 1.0 + 1e-8])
    rows = []
    best = None
    for lo in cuts:
        for hi in cuts[cuts >= lo]:
            actions = policy_actions(p, float(lo), float(hi))
            summary = summarize_policy(p, actions, costs)
            if summary["review_rate"] > costs.max_review_rate + 1e-12:
                continue
            if summary["block_rate"] > costs.max_block_rate + 1e-12:
                continue
            entry = {"review_threshold": float(lo), "block_threshold": float(hi), **summary}
            rows.append(entry)
            rank = (entry["projected_cost_total"], entry["block_rate"], entry["review_rate"])
            if best is None or rank < best[0]:
                best = (rank, entry)
    if best is None:
        raise RuntimeError("No feasible thresholds found")
    return best[1], pd.DataFrame(rows).sort_values("projected_cost_total").reset_index(drop=True)


def cost_sensitivity(probabilities, baseline_costs, fraud_losses=(10_000, 20_000, 40_000),
                     block_losses=(1_500, 5_000), review_detection_rates=(.70, .90)):
    rows = []
    for fraud_loss in fraud_losses:
        for block_loss in block_losses:
            for caught in review_detection_rates:
                c = replace(baseline_costs, fraud_approved_loss=fraud_loss,
                            legitimate_blocked_loss=block_loss,
                            review_fraud_detection_rate=caught)
                best, _ = optimize_thresholds(probabilities, c, quantile_grid=31)
                rows.append({"fraud_approved_loss": fraud_loss,
                             "legitimate_blocked_loss": block_loss,
                             "review_detection_rate": caught,
                             **best})
    return pd.DataFrame(rows)
