"""One-shot final holdout evaluation of a previously frozen fraud model/policy.

NO refitting, resampling, model selection, threshold tuning, or deployment approval.
The frozen manifest and model hash are verified before scoring the holdout.
Validation-based selection and exploratory full-dataset inspections limit the
strength of any claim that this holdout was completely unseen.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from joblib import load
import numpy as np
import pandas as pd

from src.cost_optimization import (
    CostAssumptions, policy_actions, summarize_policy,
)
from src.probability_calibration import evaluate_probabilities


def sha256_file(path: str | Path) -> str:
    """Hash file bytes without changing them."""
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen_candidate(model_path, manifest_path, feature_names, *, test_rows=600):
    """Validate the pre-test model and decision policy; never modify either."""
    model_path, manifest_path = Path(model_path), Path(manifest_path)
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Frozen model and manifest must both exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "model_sha256", "artifact_name", "feature_names", "feature_count",
        "review_threshold", "block_threshold", "calibration_strategy",
        "calibration_method", "status", "final_test_evaluation",
        "max_review_rate", "max_block_rate", "training_rows", "validation_rows",
    }
    if not required.issubset(manifest):
        raise ValueError(f"Manifest missing: {sorted(required - set(manifest))}")
    if manifest["artifact_name"] != model_path.name:
        raise ValueError("Model artifact filename disagrees with frozen manifest")
    if sha256_file(model_path) != manifest["model_sha256"]:
        raise ValueError("Model SHA-256 differs from frozen pre-holdout manifest")
    if manifest["status"] != "RESEARCH_ONLY_NOT_APPROVED_FOR_LIVE_BLOCKING":
        raise ValueError("Unexpected deployment status; refuse evaluation")
    if manifest["final_test_evaluation"] != "NOT_RUN":
        raise ValueError("Expected original pre-test frozen manifest with NOT_RUN")
    if (manifest["calibration_strategy"], manifest["calibration_method"]) != ("Unweighted", "sigmoid"):
        raise ValueError("Unexpected candidate; expected frozen Unweighted + sigmoid")
    if int(manifest["feature_count"]) != len(manifest["feature_names"]):
        raise ValueError("Invalid manifest feature count")
    if list(feature_names) != manifest["feature_names"]:
        raise ValueError("Input columns/order do not match the frozen model feature contract")
    if {"fraud", "transaction_id"} & set(feature_names):
        raise ValueError("Input columns contain target or transaction identifier")
    lo, hi = float(manifest["review_threshold"]), float(manifest["block_threshold"])
    if not (0 <= lo < hi <= 1):
        raise ValueError("Frozen thresholds invalid")
    if not (0 < float(manifest["max_review_rate"]) < 1 and 0 < float(manifest["max_block_rate"]) < 1):
        raise ValueError("Invalid capacity limits")
    if (int(manifest["training_rows"]), int(manifest["validation_rows"])) != (1800, 600):
        raise ValueError("Unexpected split sizes for this specific experiment")
    if test_rows != 600:
        raise ValueError("Expected 600 rows in the reserved holdout")
    return manifest


def prepare_costs_from_frozen_contract(manifest, costs: CostAssumptions) -> None:
    """Reject inadvertent changes to frozen operational capacity constraints."""
    costs.validate()
    if not np.isclose(costs.max_review_rate, manifest["max_review_rate"], atol=1e-12):
        raise ValueError("Review cap changed after policy freeze")
    if not np.isclose(costs.max_block_rate, manifest["max_block_rate"], atol=1e-12):
        raise ValueError("Block cap changed after policy freeze")
    # Monetary costs were not stored in the Section K manifest, so exact freeze
    # of those values cannot be proven. Check the agreed coursework assumptions.
    expected = (20_000.0, 1_500.0, 150.0, 0.90, 0.02)
    current = (costs.fraud_approved_loss, costs.legitimate_blocked_loss,
               costs.manual_review_cost, costs.review_fraud_detection_rate,
               costs.review_false_rejection_rate)
    if not np.allclose(current, expected, rtol=0, atol=1e-12):
        raise ValueError("Illustrative costs differ from the previously documented scenario")


def evaluate_frozen_holdout(*, model_path, manifest_path, X_test: pd.DataFrame,
                            y_test, transaction_ids, costs: CostAssumptions,
                            output_dir, report_dir):
    """Evaluate one frozen candidate exactly once and save an immutable audit marker.

    Test outcomes are used strictly for post-hoc metrics, never candidate/policy
    selection. A second attempt aborts while preserving the previous audit.
    """
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    marker = output_dir / "10_final_holdout_one_shot_lock.json"
    if marker.exists():
        raise FileExistsError(
            f"Final holdout was already opened: {marker}. Do not rescore, reselect or retune."
        )
    if not isinstance(X_test, pd.DataFrame) or len(X_test) != 600:
        raise ValueError("Expected a 600-row DataFrame for reserved final test")
    manifest = verify_frozen_candidate(model_path, manifest_path, X_test.columns, test_rows=len(X_test))
    prepare_costs_from_frozen_contract(manifest, costs)
    ids = np.asarray(transaction_ids, dtype=object)
    y = np.asarray(y_test)
    if ids.ndim != 1 or len(ids) != len(X_test) or pd.isna(ids).any() or len(pd.unique(ids)) != len(ids):
        raise ValueError("Expected 600 unique, aligned nonmissing transaction IDs")
    if y.ndim != 1 or len(y) != len(X_test) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected 600 aligned binary fraud labels")
    if X_test.index.has_duplicates:
        raise ValueError("Test features must preserve a unique row index")

    model = load(model_path)  # Load ONLY this locally created, trusted project artifact.
    probs = np.asarray(model.predict_proba(X_test), dtype=float)
    if probs.shape != (len(X_test), 2):
        raise ValueError("Unexpected model probability matrix shape")
    p = probs[:, 1]
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Nonfinite or invalid predicted probabilities")
    actions = policy_actions(
        p, float(manifest["review_threshold"]), float(manifest["block_threshold"]),
    )
    probabilities_metrics = evaluate_probabilities(y, p)
    policy = summarize_policy(p, actions, costs, y_true=y)
    baseline = summarize_policy(p, np.full(len(y), "APPROVE"), costs, y_true=y)
    policy.update({
        "capacity_review_exceeded": bool(policy["review_rate"] > costs.max_review_rate + 1e-12),
        "capacity_block_exceeded": bool(policy["block_rate"] > costs.max_block_rate + 1e-12),
        "baseline_approve_all_scenario_cost_total": baseline["validation_outcome_cost_total"],
        "difference_vs_approve_all_scenario_cost_total": (
            baseline["validation_outcome_cost_total"] - policy["validation_outcome_cost_total"]
        ),
    })
    metrics = {
        "n_test": len(y), "fraud_cases": int(y.sum()),
        "fraud_prevalence": float(y.mean()), **probabilities_metrics,
        "frozen_review_threshold": float(manifest["review_threshold"]),
        "frozen_block_threshold": float(manifest["block_threshold"]),
        "model_sha256": manifest["model_sha256"],
    }
    scored = pd.DataFrame({
        "transaction_id": ids, "fraud_probability": p,
        "proposed_action": actions, "observed_fraud_for_offline_audit": y.astype(int),
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    # Establish the one-shot marker before saving any result. On an interrupted
    # run, preserve it and reconcile outputs manually (never blindly rerun).
    with marker.open("x", encoding="utf-8") as f:
        json.dump({
            "status": "FINAL_HOLDOUT_OPENED_ONCE_RESEARCH_ONLY",
            "opened_utc": now, "model_sha256": manifest["model_sha256"],
            "frozen_manifest_sha256": sha256_file(manifest_path),
            "review_threshold": manifest["review_threshold"],
            "block_threshold": manifest["block_threshold"],
            "test_rows": len(y), "dataset_scope": "fixed_stratified_random_test",
            "warning": "Do not revise/reselect model or thresholds using these scores. Not production-approved.",
        }, f, indent=2)
    metric_path = output_dir / "10_final_holdout_metrics.csv"
    policy_path = output_dir / "10_final_holdout_policy_audit.csv"
    scored_path = output_dir / "10_final_holdout_scored_transactions.csv"
    report_path = report_dir / "11_final_holdout_evaluation.md"
    pd.DataFrame([metrics]).to_csv(metric_path, index=False)
    pd.DataFrame([policy]).to_csv(policy_path, index=False)
    scored.to_csv(scored_path, index=False)
    report = f"""# Final Holdout — One-Time Evaluation (Research Prototype)

## Evaluation contract
- Evaluated the PRE-FROZEN **{manifest['calibration_strategy']} / {manifest['calibration_method']}** model.
- Frozen model SHA256: `{manifest['model_sha256']}`.
- The manifest was created before final test evaluation and remained unchanged.
- Policy thresholds NOT refit/re-optimized: **review >= {manifest['review_threshold']:.6f}**, **block >= {manifest['block_threshold']:.6f}**.
- Final test rows: **{len(y)}**, fraud labels: **{int(y.sum())}** ({y.mean():.2%}).
- Feature order and model hash checked against manifest; no target/ID entered model inputs.

## Predictive metrics — final stratified holdout
- Average precision (PR-AUC summary): **{metrics['average_precision']:.4f}**.
- ROC-AUC: **{metrics['roc_auc']:.4f}**.
- Brier score: **{metrics['brier']:.4f}**.
- Log loss: **{metrics['log_loss']:.4f}**.
- ECE (10 bins): **{metrics['ece_10']:.4f}**.

## Frozen policy on final test
- Approve: {policy['approve_count']}, Review: {policy['review_count']}, Proposed block: {policy['block_count']}.
- Review rate: {policy['review_rate']:.2%} (cap: {costs.max_review_rate:.2%}); breach: **{policy['capacity_review_exceeded']}**.
- Block rate: {policy['block_rate']:.2%} (cap: {costs.max_block_rate:.2%}); breach: **{policy['capacity_block_exceeded']}**.
- Fraud approved: {policy['fraud_approved_count']}; fraud reviewed: {policy['fraud_review_count']}; fraud proposed-blocked: {policy['fraud_blocked_count']}.
- Legitimate transactions proposed-blocked: {policy['legitimate_blocked_count']}.
- Outcome-based *hypothetical* total cost: LKR {policy['validation_outcome_cost_total']:,.2f}.
- Approve-all *hypothetical* total cost: LKR {baseline['validation_outcome_cost_total']:,.2f}.
- Difference: LKR {policy['difference_vs_approve_all_scenario_cost_total']:,.2f} (NOT verified realized savings).

## Warnings and limitations
- This was one-time evaluation on the reserved random-stratified 600-row test slice.
- Earlier whole-dataset exploratory inspections mean the test was not fully unseen in the strictest sense.
- Model and thresholds were repeatedly compared on validation, so selection uncertainty remains.
- Missing full transaction timestamps/customer IDs prevents chronological and customer-group validation.
- Seven prediction-time features remain unverified, including historical and risk-score provenance.
- Financial costs, reviewer outcomes and assumed effectiveness have not been independently verified.
- Test labels have **not** been used to select a new model, change thresholds or calibrate scores.
- Any future model/policy modification requires NEW independent data; do not reuse this holdout for selection.
- Section K critical governance gates remain unresolved; **NO-GO for live automated blocking**.
- Saving the score audit is for synthetic research only. Real deployments need appropriate data minimization/access controls.
"""
    report_path.write_text(report, encoding="utf-8")
    return {
        "metrics": pd.DataFrame([metrics]),
        "policy": pd.DataFrame([policy]),
        "scored_transactions": scored,
        "file_paths": [metric_path, policy_path, scored_path, report_path, marker],
    }
