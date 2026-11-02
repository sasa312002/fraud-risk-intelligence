"""Offline model-governance evidence for the Fraud Risk Intelligence coursework.

This module intentionally does not score the reserved test set, deploy a model,
or automatically change decision thresholds following stress-test failures.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

REQUIRED_STRESS_COLUMNS = {
    "scenario", "review_rate", "block_rate", "capacity_review_exceeded",
    "capacity_block_exceeded", "action_flip_rate",
}


def assess_governance_gates(stress: pd.DataFrame, review_cap: float,
                            block_cap: float) -> pd.DataFrame:
    """Return evidence-backed GO/NO-GO gates, including explicitly unverified controls."""
    if not REQUIRED_STRESS_COLUMNS.issubset(stress.columns):
        raise ValueError(f"Missing stress columns: {sorted(REQUIRED_STRESS_COLUMNS - set(stress.columns))}")
    if len(stress) == 0 or (stress["scenario"] == "baseline").sum() != 1:
        raise ValueError("Stress results must contain exactly one baseline row")
    if not (0 < review_cap < 1 and 0 < block_cap < 1):
        raise ValueError("Capacity limits must be between zero and one")
    if stress["scenario"].duplicated().any():
        raise ValueError("Duplicate scenario names")
    if stress[["review_rate", "block_rate", "action_flip_rate"]].isna().any().any():
        raise ValueError("Stress rates must not be missing")
    if not stress[["review_rate", "block_rate", "action_flip_rate"]].apply(lambda s: s.between(0, 1)).all().all():
        raise ValueError("Stress rates must be between zero and one")
    baseline = stress.loc[stress.scenario == "baseline"].iloc[0]
    others = stress.loc[stress.scenario != "baseline"]
    review_breaches = others.loc[others.review_rate > review_cap + 1e-10, 'scenario'].tolist()
    block_breaches = others.loc[others.block_rate > block_cap + 1e-10, 'scenario'].tolist()
    rows = [
        ("baseline_review_capacity", bool(baseline.review_rate <= review_cap + 1e-10),
         "Operational", f"{baseline.review_rate:.2%} review vs {review_cap:.2%} cap"),
        ("baseline_block_capacity", bool(baseline.block_rate <= block_cap + 1e-10),
         "Operational", f"{baseline.block_rate:.2%} block vs {block_cap:.2%} cap"),
        ("stress_review_capacity", not review_breaches, "Critical",
         f"Breaching scenarios: {', '.join(review_breaches) if review_breaches else 'none'}"),
        ("stress_block_capacity", not block_breaches, "Critical",
         f"Breaching scenarios: {', '.join(block_breaches) if block_breaches else 'none'}"),
        ("prediction_time_feature_provenance", False, "Critical",
         "Pre-authorization availability for 7 VERIFY fields not independently established"),
        ("out_of_time_or_customer_group_validation", False, "Critical",
         "Full timestamps and customer identifiers unavailable"),
        ("real_review_outcomes_and_costs_verified", False, "Critical",
         "Reviewer effectiveness and business loss amounts are assumptions"),
        ("fairness_and_appeal_process_reviewed", False, "Critical",
         "No protected-class fairness audit or appeal-process evaluation completed"),
        ("independent_final_holdout_evaluated", False, "Critical",
         "Reserved 600-row final test set has not been scored"),
        ("live_monitoring_and_human_signoff", False, "Critical",
         "Monitoring ownership, escalation and deployment approval not validated"),
    ]
    return pd.DataFrame(rows, columns=["gate", "passed", "severity", "evidence"])


def build_monitoring_contract(review_cap: float, block_cap: float) -> pd.DataFrame:
    """Suggested alert policy, not a running monitor or actual SLA."""
    if not (0 < review_cap < 1 and 0 < block_cap < 1):
        raise ValueError("Capacity limits must be between zero and one")
    return pd.DataFrame([
        ("review_rate", "Per batch", f">= {0.9*review_cap:.1%}", f"> {review_cap:.1%}",
         "Route overflow to an explicit, human-approved queue; never auto-expand capacity"),
        ("block_rate", "Per batch", f">= {0.9*block_cap:.1%}", f"> {block_cap:.1%}",
         "Pause auto-block rollout, escalate to fraud operations and review policy"),
        ("authentication_missing_rate", "Daily", "Historical baseline + 5 percentage points", "+ 10 percentage points",
         "Inspect upstream auth logging and data contracts"),
        ("unseen_category_rate", "Daily", "> 1%", "> 5%", "Check encoder schema and source changes"),
        ("numeric_psi", "Weekly", ">= 0.10", ">= 0.25", "Investigate segment drift; confirm reference window is chronological"),
        ("brier_score", "After delayed labels", "+ 10% vs holdout", "+ 25% vs holdout",
         "Trigger calibration investigation on mature labels, not immediate refit"),
        ("average_precision", "After delayed labels", "- 10% vs holdout", "- 20% vs holdout",
         "Investigate label delays, sampling and class prevalence"),
        ("manual_review_backlog", "Per batch", "> 0 pending cases", "> reviewer capacity",
         "Escalate analyst coverage; avoid assumed SLA claims"),
    ], columns=["metric", "cadence", "warning_trigger", "critical_trigger", "action"])


def build_risk_register(stress: pd.DataFrame) -> pd.DataFrame:
    gates = assess_governance_gates(stress, .20, .05)
    breaches = stress.loc[stress.capacity_review_exceeded | stress.capacity_block_exceeded, 'scenario'].tolist()
    return pd.DataFrame([
        ("R01", "Capacity overflow under shift", "High", "Observed in offline stress replay: " + ", ".join(breaches),
         "Human approval for overflow routing; monitor both action rates", "Fraud Operations"),
        ("R02", "Large false-block count", "High", "17 legitimate proposed blocks on validation under current illustrative policy",
         "Human override, appeals, two-person controls for high-impact actions", "Risk Owner"),
        ("R03", "Fraud approved", "High", "34 of 73 fraud labels approved in retrospective audit",
         "Fraud-loss surveillance; human review/threshold reevaluation on fresh data", "Fraud Operations"),
        ("R04", "Feature leakage / pre-authorization availability", "High", "Multiple fields remain VERIFY",
         "Signed point-in-time feature contract and pipeline audit", "Data Engineering"),
        ("R05", "Temporal and customer generalization unknown", "High", "No customer IDs or full timestamps",
         "Secure chronological and entity-level external validation", "Model Risk"),
        ("R06", "Repeated selection on validation", "High", "Model, calibration and cost policy chosen on same validation batch",
         "Freeze whole pipeline; one-time final holdout, then fresh data for further revisions", "Model Risk"),
        ("R07", "Illustrative reviewer/cost assumptions", "High", "90% reviewer detection and LKR cost matrix unverified",
         "Replace with measured review and cost observations", "Finance / Operations"),
        ("R08", "Fairness, privacy and contestability", "High", "Sensitive attributes/consent and appeals not assessed",
         "Minimize retained identifiers, bias assessment and appeal protocol", "Compliance"),
        ("R09", "Label delay, data drift and unseen categories", "Medium", "No live monitoring established",
         "Monitor missingness, PSI, score mix and post-label outcomes", "MLOps"),
        ("R10", "Small holdouts and uncertain ranking gains", "Medium", "Validation fraud count only 73",
         "Bootstrap intervals; independent external evaluation", "Data Science"),
    ], columns=["risk_id", "risk", "severity", "evidence", "mitigation", "proposed_owner"])


def freeze_candidate_manifest(model_path: str | Path, output_path: str | Path,
                              *, model_choice: tuple[str, str], review_threshold: float,
                              block_threshold: float, feature_names: list[str],
                              training_rows: int, validation_rows: int,
                              review_cap: float, block_cap: float) -> dict:
    """Record an immutable-ish checksum and proposed policy *before* independent test.

    Saving a manifest does not make it production-approved. Model file bytes are not
    modified. No test features/labels are referenced or scored.
    """
    model_path, output_path = Path(model_path), Path(output_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing provisional model artifact: {model_path}")
    if not (0 <= review_threshold < block_threshold <= 1):
        raise ValueError("Thresholds must satisfy 0 <= review < block <= 1")
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("Feature names must be nonempty and unique")
    if {'fraud', 'transaction_id'} & set(feature_names):
        raise ValueError("Target and identifier cannot be model features")
    manifest = {
        "artifact_name": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "calibration_strategy": model_choice[0],
        "calibration_method": model_choice[1],
        "review_threshold": round(float(review_threshold), 10),
        "block_threshold": round(float(block_threshold), 10),
        "max_review_rate": float(review_cap),
        "max_block_rate": float(block_cap),
        "training_rows": int(training_rows),
        "validation_rows": int(validation_rows),
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "status": "RESEARCH_ONLY_NOT_APPROVED_FOR_LIVE_BLOCKING",
        "final_test_evaluation": "NOT_RUN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def model_card_markdown(gates: pd.DataFrame, monitoring: pd.DataFrame,
                        risks: pd.DataFrame, manifest: dict,
                        stress: pd.DataFrame) -> str:
    """Produce an explicit limitations-first offline model card."""
    failed = gates.loc[~gates.passed, 'gate'].tolist()
    breached = stress.loc[stress.capacity_review_exceeded | stress.capacity_block_exceeded, 'scenario'].tolist()
    gates_md = gates.to_markdown(index=False) if False else _markdown(gates)
    return (f"# Model Card & Governance Review — Fraud Risk Intelligence\n\n"
            f"**Deployment decision: NO-GO (research prototype).**\n\n"
            f"## Candidate policy freeze\n"
            f"- Model: {manifest['calibration_strategy']} + {manifest['calibration_method']} logistic calibration.\n"
            f"- SHA256: `{manifest['model_sha256']}`.\n"
            f"- Review at >= {manifest['review_threshold']:.6f}; propose block at >= {manifest['block_threshold']:.6f}.\n"
            f"- 1,800 training and 600 validation rows; independent test status: NOT_RUN.\n"
            f"- Policy actions are proposals only; manual review and contest/appeal required before blocking.\n\n"
            f"## Governance gates\n{gates_md}\n\n"
            f"Failed/unverified gates: **{', '.join(failed)}**.\n\n"
            f"## Monitoring contract (proposal, not deployed)\n{_markdown(monitoring)}\n\n"
            f"## Risk register\n{_markdown(risks)}\n\n"
            f"## Stress evidence\n"
            f"- Capacity-breach scenarios: {', '.join(breached) if breached else 'none'}.\n"
            f"- Fixed thresholds are not robust to velocity +2 and certain missingness shifts.\n"
            f"- Any stress cost decrease is hypothetical, not verified savings.\n\n"
            f"## Intended use and boundaries\n"
            f"Offline educational fraud *risk ranking* and human-review policy simulation only; not live automated blocking.\n"
            f"Validation results are not independent of calibration/policy selection.\n"
            f"All cost amounts, manual-review efficacy, throughput and action budgets are hypothetical.\n"
            f"No chronological validation, entity-group split, protected-attribute fairness audit, confirmed feature provenance, or real user appeal mechanism.\n\n"
            f"## Required next step\n"
            f"Freeze candidate and thresholds (this manifest); design a **one-time** final test evaluation with no additional tuning.\n"
            f"If the final test is used to modify model or thresholds, a *new* untouched evaluation set is needed.\n")


def _markdown(frame: pd.DataFrame) -> str:
    # Pure-Python markdown table; avoids mandatory 'tabulate' dependency.
    def escape(v: object) -> str:
        return str(v).replace('|', r'\|').replace('\n', ' ')
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    line = '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'
    rows = ['| ' + ' | '.join(escape(v) for v in row) + ' |'
            for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, line, *rows])
