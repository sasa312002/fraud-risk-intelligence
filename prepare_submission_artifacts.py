"""Prepare required DS02 submission artifacts WITHOUT fitting or scoring the test set.

Run once from your project's root:
    python prepare_submission_artifacts.py

Inputs must exist in their original locations; this script never changes or
re-fits the frozen model, calibration, policy, manifest, or final test lock.

Outputs:
    data/processed/02_processed_dataset.csv
    models/03_fraud_model_pipeline.joblib
    models/03_pipeline_provenance.json
    models/04_calibrated_model.joblib

03 is ONE FITTED CONSTITUENT pipeline extracted from the three-member
calibrated ensemble.  It is not the final inference model; use 04 for scoring.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_MODEL = PROJECT_ROOT / "models" / "section_f_provisional_calibrated.joblib"
MANIFEST = PROJECT_ROOT / "models" / "section_k_candidate_manifest.json"
FINAL_TEST_LOCK = PROJECT_ROOT / "outputs" / "10_final_holdout_one_shot_lock.json"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "transaction_fraud_risk_advanced_dataset.csv"
THRESHOLD_CSV = PROJECT_ROOT / "outputs" / "05_threshold_cost_analysis.csv"

EXPECTED_PRETEST_SHA256 = (
    "87217ee986fbed6763df7111294c051c94fb7934ea9ce3e185064c6841bce517"
)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def must_exist(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")


def write_new_only(path: Path, writing_function) -> None:
    if path.exists():
        raise FileExistsError(
            f"Output already exists; not overwriting: {path}\n"
            "Review it and rename/move the old version if you need to regenerate."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    writing_function(path)


def main() -> None:
    for required in (SOURCE_MODEL, MANIFEST, FINAL_TEST_LOCK, RAW_DATA, THRESHOLD_CSV):
        must_exist(required)

    pretest_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_sha = checksum(SOURCE_MODEL)
    assert source_sha == EXPECTED_PRETEST_SHA256, (
        "Frozen model hash differs from the previously verified pre-test SHA256. "
        "STOP: do not create new artifacts from this file."
    )
    assert source_sha == pretest_manifest["model_sha256"], (
        "Frozen model does not match the Section K governance manifest."
    )
    assert pretest_manifest.get("calibration_strategy") == "Unweighted"
    assert pretest_manifest.get("calibration_method") == "sigmoid"
    assert pretest_manifest.get("status") == "RESEARCH_ONLY_NOT_APPROVED_FOR_LIVE_BLOCKING"

    model = joblib.load(SOURCE_MODEL)
    if not isinstance(model, CalibratedClassifierCV):
        raise TypeError(
            f"Expected a fitted CalibratedClassifierCV; got {type(model).__name__}"
        )
    if getattr(model, "method", None) != "sigmoid":
        raise ValueError("Frozen calibration method is not sigmoid.")
    check_is_fitted(model)

    calibrated_members = model.calibrated_classifiers_
    if not calibrated_members:
        raise ValueError("No fitted calibrated classifier members found.")
    first_member = calibrated_members[0]
    # sklearn currently exposes `estimator`; support older compatible names.
    fitted_pipeline = getattr(first_member, "estimator", None)
    if fitted_pipeline is None:
        fitted_pipeline = getattr(first_member, "base_estimator", None)
    if not isinstance(fitted_pipeline, Pipeline):
        raise TypeError(
            "Expected a fitted sklearn Pipeline inside calibrated model; "
            f"got {type(fitted_pipeline).__name__}. Do not use an arbitrary model substitute."
        )
    if not {"preprocess", "classifier"}.issubset(fitted_pipeline.named_steps):
        raise ValueError("Fitted constituent pipeline is missing preprocessing/classifier steps.")
    check_is_fitted(fitted_pipeline)

    raw = pd.read_csv(RAW_DATA)
    expected_columns = list(pretest_manifest.get("feature_names", []))
    if len(raw) != 3000 or len(raw.columns) != 25:
        raise ValueError(f"Unexpected raw dataset shape {raw.shape}; expected (3000, 25).")
    if {"fraud", "transaction_id"} - set(raw.columns):
        raise ValueError("Missing transaction_id or fraud target in raw dataset.")
    if raw["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction identifiers found; do not auto-drop rows.")
    if not raw["fraud"].isin([0, 1]).all():
        raise ValueError("Invalid fraud target; do not modify rows silently.")
    if (set(raw.columns) - {"fraud", "transaction_id"}) != set(expected_columns):
        raise ValueError("Raw feature schema is inconsistent with the pretest manifest.")
    if len(expected_columns) != int(pretest_manifest.get("feature_count", -1)):
        raise ValueError("Feature count in pre-test manifest is inconsistent.")

    processed = PROJECT_ROOT / "data" / "processed" / "02_processed_dataset.csv"
    # Validated, lossless row-level snapshot. No full-dataset imputation,
    # target-guided feature construction, or fitted preprocessing here.
    write_new_only(processed, lambda p: raw.to_csv(p, index=False, float_format="%.17g"))

    pipeline_out = PROJECT_ROOT / "models" / "03_fraud_model_pipeline.joblib"
    write_new_only(pipeline_out, lambda p: joblib.dump(fitted_pipeline, p))
    pipeline_sha = checksum(pipeline_out)

    provenance_out = PROJECT_ROOT / "models" / "03_pipeline_provenance.json"
    provenance = {
        "artifact_name": pipeline_out.name,
        "sha256": pipeline_sha,
        "source_calibrated_model": SOURCE_MODEL.name,
        "source_frozen_model_sha256": source_sha,
        "extracted_member_index_zero_based": 0,
        "total_calibrated_members": len(calibrated_members),
        "description": (
            "Fitted preprocessing + logistic classifier from ONE calibration "
            "cross-validation member. Use 04_calibrated_model.joblib "
            "for final fraud probability predictions. No retraining performed."
        ),
        "deployment_status": "RESEARCH_ONLY_NOT_APPROVED_FOR_LIVE_BLOCKING",
    }
    write_new_only(
        provenance_out,
        lambda p: p.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8"),
    )

    calibrated_out = PROJECT_ROOT / "models" / "04_calibrated_model.joblib"
    write_new_only(calibrated_out, lambda p: shutil.copy2(SOURCE_MODEL, p))
    if checksum(calibrated_out) != source_sha:
        raise AssertionError("Calibrated model copy does not match frozen SHA256")
    check_is_fitted(joblib.load(calibrated_out))

    print("SUBMISSION ARTIFACTS READY (no model training/test scoring)")
    print("  ", processed.relative_to(PROJECT_ROOT))
    print("  ", pipeline_out.relative_to(PROJECT_ROOT))
    print("  ", provenance_out.relative_to(PROJECT_ROOT))
    print("  ", calibrated_out.relative_to(PROJECT_ROOT))
    print("  Existing cost file:", THRESHOLD_CSV.relative_to(PROJECT_ROOT))
    print("Frozen calibrated model SHA256 verified:", source_sha)
    print("Constituent fitted pipelines:", len(calibrated_members))
    print("Processed snapshot:", raw.shape, "missing cells:", int(raw.isna().sum().sum()))
    print("IMPORTANT: Only 04_calibrated_model.joblib is the frozen final probability scorer.")
    print("IMPORTANT: No production deployment approval is implied.")
    print("IMPORTANT: Do not rerun the final holdout cell.")


if __name__ == "__main__":
    main()
