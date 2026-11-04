# Fraud Risk Intelligence

**An Explainable and Cost-Aware Fraud Detection Framework**  
**Project type:** Data Science / Machine Learning research prototype  
**Current disposition:** **RESEARCH ONLY — NOT APPROVED FOR LIVE AUTOMATED BLOCKING**

## 1. Project overview

This project develops an end-to-end framework to estimate transaction-level fraud risk and study how estimated probabilities could support three operational actions: **Approve**, **Manual Review**, and **Propose Block**. The workflow covers data forensics, exploratory analysis, feature engineering, model comparison, imbalanced learning, probability calibration, cost-sensitive threshold selection, operational simulation, explainability, robustness testing, governance, and one-shot final holdout evaluation.

**Important:** Results are based on a limited research dataset and *illustrative business cost assumptions*. Neither the model nor its block recommendations have production authorization. A high fraud score is not proof of fraudulent intent.

## 2. Problem and objectives

A fraud detection system must identify suspicious transactions without imposing excessive false-positive friction on legitimate customers. Accordingly, this project aims to:

- Compare transparent and nonlinear fraud classifiers using class-imbalance-aware metrics.
- Produce calibrated probabilities and select operational thresholds using hypothetical business costs.
- Quantify manual-review demand, proposed blocks, missed fraud, and customer impact.
- Explain model behavior using SHAP, evaluate robustness under input changes, and document deployment risks.
- Preserve a reserved final holdout set for a single assessment of the frozen model and policy.

## 3. Dataset and validation design

| Item | Value |
|---|---:|
| Transactions | 3,000 |
| Original columns | 25 |
| Legitimate transactions | 2,636 |
| Fraudulent transactions | 364 |
| Overall fraud prevalence | 12.13% |
| Missing cells in raw data | 582 |
| Duplicate transaction IDs | 0 |
| Train / validation / final test | 1,800 / 600 / 600 |
| Fraud cases in train / validation / test | 218 / 73 / 73 |

The data was divided with stratified random sampling. The project **does not** establish out-of-time or customer-disjoint generalization because sufficient timestamps and customer identifiers were unavailable. Some candidate features also require verification that they are available at transaction authorization time.

**Data locations**

- Raw source used by the artifact preparation utility: `data/raw/transaction_fraud_risk_advanced_dataset.csv`.
- Required processed artifact: `data/processed/02_processed_dataset.csv`.

The processed CSV is a **validated, lossless row-level snapshot**: 3,000 rows, 25 original columns, and 582 remaining missing cells. It is **not** a globally imputed or one-hot-encoded table. Learned preprocessing occurs inside fitted machine-learning pipelines, which avoids fitting imputation or encoders on the entire dataset before splitting.

## 4. Methods

The experimental workflow includes:

1. **Data audit:** schema checks, missingness, duplicates, extreme values, leakage-risk review.
2. **Exploratory analysis:** fraud rates by transaction, customer, device, authentication, and behavioral characteristics; interaction analyses.
3. **Feature engineering:** 16 experimental derived predictors, compared with the original 23-input configuration.
4. **Model development:** Logistic Regression, Random Forest, and XGBoost; reproducible cross-validation and limited hyperparameter search.
5. **Imbalanced learning:** unweighted classification, class weighting, and controlled random oversampling.
6. **Calibration:** raw probabilities, sigmoid calibration, and isotonic calibration.
7. **Decision optimization:** a three-action policy with hypothetical fraud, blocking, review, and investigation-effectiveness costs.
8. **Operations:** simulated review queues, capacity limits, and scenario cost analysis.
9. **Explainability:** SHAP global and local explanations of the **underlying logistic model's log-odds** (not additive contributions to calibrated probability).
10. **Robustness and governance:** input perturbation scenarios, monitoring proposals, and deployment-readiness gates.
11. **Final evaluation:** one-shot assessment using the frozen scoring artifact and preselected policy on 600 held-out transactions.

## 5. Final scoring model and artifacts

**Frozen candidate:** unweighted Logistic Regression with sigmoid probability calibration, stored as a fitted `CalibratedClassifierCV` ensemble containing **three calibrated constituent members**.

| Artifact | Purpose |
|---|---|
| `models/03_fraud_model_pipeline.joblib` | **One fitted preprocessing + Logistic Regression constituent** extracted from the calibrated ensemble, supplied for technical inspection. It is **not** the final probability scorer. |
| `models/03_pipeline_provenance.json` | Extraction provenance and checksums for the constituent pipeline. |
| `models/04_calibrated_model.joblib` | **Frozen complete calibrated ensemble**; the only approved artifact in this project for reproducing the final *research* fraud probability scores. |
| `models/section_k_candidate_manifest.json` | Frozen candidate metadata, feature contract, checksum, and research-only status. |
| `outputs/10_final_holdout_one_shot_lock.json` | Record of the completed reserved final holdout assessment. |

**Verified SHA-256 of the frozen calibrated artifact:**

```text
87217ee986fbed6763df7111294c051c94fb7934ea9ce3e185064c6841bce517
```

`prepare_submission_artifacts.py` creates the required named dataset and model artifacts from already validated data and fitted models. It **does not retrain the model or score the final test set**. The `03_*` pipeline cannot be substituted for `04_calibrated_model.joblib` when reproducing the final probabilities.

## 6. Frozen cost-sensitive decision policy

These thresholds were selected during validation and **frozen before final testing**:

| Calibrated probability `p` | Decision |
|---|---|
| `p < 0.1520464035` | **APPROVE** |
| `0.1520464035 <= p < 0.2620026706` | **MANUAL REVIEW** |
| `p >= 0.2620026706` | **PROPOSE BLOCK** — not authorized for automatic live execution |

### Illustrative business assumptions

| Assumption | Value |
|---|---:|
| Fraud approved | LKR 20,000 |
| Legitimate transaction blocked | LKR 1,500 |
| Manual review | LKR 150 |
| Assumed reviewer fraud detection rate | 90% |
| Assumed reviewer false-rejection rate | 2% |
| Maximum review volume | 20% |
| Maximum proposed block volume | 5% |

These values were chosen for simulation and sensitivity analysis, **not measured from a financial institution**.

## 7. Final holdout results

The frozen model and thresholds were evaluated against a reserved dataset of **600 transactions**, including **73 fraudulent cases**.

| Predictive measure | Final holdout |
|---|---:|
| Average Precision (PR-AUC summary) | 0.1932 |
| ROC-AUC | 0.6640 |
| Brier score | 0.1039 |
| Log loss | 0.3550 |
| Expected Calibration Error (10 bins) | 0.0215 |
| Mean predicted fraud probability | 11.87% |
| Observed fraud prevalence | 12.17% |

**Frozen policy routing on final holdout:**

| Action | Count | Share |
|---|---:|---:|
| Approve | 474 | 79% |
| Manual review | 114 | 19% |
| Propose block | 12 | 2% |

Of **73 fraudulent transactions**, **46 were approved**, **23 were referred for manual review**, and **4 were proposed for blocking**. Of the **12 proposed blocks**, **8 were legitimate transactions**. These customer-impact and missed-fraud findings are central reasons for the **NO-GO** decision.

**Illustrative retrospective scenario costs:** approve-all **LKR 1,460,000** versus triage-policy **LKR 997,830**, a simulated difference of **LKR 462,170 (31.66%)**. This is **not realized savings** and depends on hypothetical costs and assumed review outcomes. The probability-based projected triage cost was approximately **LKR 998,406**.

Validation Average Precision was approximately **0.2408** versus **0.1932** on final holdout; the decrease warrants further independent testing and does not justify retuning on the test results.

## 8. Robustness, explainability, and governance

**Explainability:** top global SHAP drivers for the underlying logistic model included merchant category, transactions in the previous hour, new-device status, transaction channel, and merchant risk score. SHAP explanations reflect model associations, not causal proof.

**Robustness:** in the `velocity_plus2` perturbation scenario, **20.50%** of decisions changed; the review rate reached **33.83%** and the proposed block rate **7.33%**, exceeding the assumed 20% and 5% operational limits. The perturbation is a diagnostic sensitivity test, not a measured future fraud scenario.

**Governance decision:** **RESEARCH_ONLY_NOT_APPROVED_FOR_LIVE_BLOCKING**. Outstanding requirements include verified point-in-time input availability, out-of-time/customer-disjoint validation, business cost evidence, fairness and appeals assessment, capacity-aware operational safeguards, live monitoring, and documented human authorization.

Proposed monitoring includes review and proposed-block rates, input missingness, unseen categories, numerical drift/PSI, probability calibration, PR-AUC when labels mature, and reviewer backlog. Monitoring has **not** been deployed.

## 9. Project structure

The following represents key files and folders (additional reports, plots, tests, and outputs may be present):

```text
fraud-risk-intelligence/
├── data/
│   ├── raw/
│   │   └── transaction_fraud_risk_advanced_dataset.csv
│   └── processed/
│       └── 02_processed_dataset.csv
├── notebooks/
│   └── 01_fraud_risk_analysis.ipynb
├── src/
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── cost_optimization.py
│   ├── probability_calibration.py
│   ├── imbalance_learning.py
│   ├── explainability.py
│   ├── fraud_operations.py
│   ├── robustness_testing.py
│   ├── model_governance.py
│   └── final_holdout_evaluation.py
├── models/
│   ├── 03_fraud_model_pipeline.joblib
│   ├── 03_pipeline_provenance.json
│   ├── 04_calibrated_model.joblib
│   └── section_k_candidate_manifest.json
├── outputs/
│   ├── 05_threshold_cost_analysis.csv
│   ├── 10_final_holdout_metrics.csv
│   ├── 10_final_holdout_policy_audit.csv
│   ├── 10_final_holdout_scored_transactions.csv
│   └── 10_final_holdout_one_shot_lock.json
├── reports/
├── tests/
├── requirements.txt
├── prepare_submission_artifacts.py
└── 08_README.md
```

The final report PDFs (`06_model_card.pdf`, `07_business_fraud_strategy.pdf`) are required submission documents; confirm their location in the final submission package.

## 10. Installation and environment (Windows PowerShell)

From the repository root, use an existing virtual environment or create one with **Python 3.11**:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For notebook execution, open `notebooks/01_fraud_risk_analysis.ipynb` in VS Code and select the project's `.venv` kernel. If Jupyter support is not installed through the requirements, install `ipykernel` in the same environment. A machine-specific directory path is **not** required: open a terminal from the project root.

> Dependency versions may differ between installations. Use the repository's dependency specification (and `requirements-lock.txt`, if maintained) for reproduction; verify that local Python/scikit-learn/joblib versions can deserialize the provided artifacts. Only load `.joblib` files from trusted sources.

## 11. Safe verification commands

Run these commands from the repository root with `.venv` activated:

```powershell
# Automated tests; does not require rerunning the final holdout notebook cell
python -m pytest tests/ -v

# Check required files
Test-Path .\data\processed\02_processed_dataset.csv
Test-Path .\models\03_fraud_model_pipeline.joblib
Test-Path .\models\04_calibrated_model.joblib
Test-Path .\outputs\05_threshold_cost_analysis.csv

# Verify the calibrated artifact's frozen SHA-256
(Get-FileHash .\models\04_calibrated_model.joblib -Algorithm SHA256).Hash.ToLower()

# Check model deserialization (trusted artifact only); does not score test data
python -c "import joblib; m=joblib.load('models/04_calibrated_model.joblib'); print(type(m).__name__)"
```

The SHA-256 output should equal the checksum in Section 5. For first-time artifact creation only, run `python prepare_submission_artifacts.py` **if** its outputs do not already exist; the script intentionally refuses to overwrite them. After a successful artifact preparation run, **do not rerun it unnecessarily**.

### Reproducibility and holdout integrity

**Do not rerun the reserved final holdout evaluation cell** (`Section L`, previously identified in the project as the final holdout cell) or use the final test dataset for tuning, calibration, model selection, feature selection, resampling, or threshold changes. The existing outputs and one-shot lock document the completed assessment.

Training, model comparison, and validation experiments can be reviewed in the notebook. Re-running training sections can produce alternative fitted artifacts and is **not** the procedure for reproducing the locked final result. To regenerate research results independently, create a fresh versioned experiment and reserve a **new** untouched test set.

## 12. Required submission deliverables

| No. | Required item | Expected location |
|---|---|---|
| 01 | `01_fraud_risk_analysis.ipynb` | `notebooks/` |
| 02 | `02_processed_dataset.csv` | `data/processed/` |
| 03 | `03_fraud_model_pipeline.joblib` | `models/` |
| 04 | `04_calibrated_model.joblib` | `models/` |
| 05 | `05_threshold_cost_analysis.csv` | `outputs/` |
| 06 | `06_model_card.pdf` | Add to final submission |
| 07 | `07_business_fraud_strategy.pdf` | Add to final submission |
| 08 | `08_README.md` | Repository root |

Supporting evidence includes validation comparisons, calibration results, SHAP explanations, stress tests, governance documents, final holdout metrics, and the one-shot lock under `outputs/` and `reports/`.

Before submitting, verify that `.gitignore` does not unintentionally exclude required deliverables. If models or datasets are intentionally excluded from Git due to size or data governance, provide them through the approved submission method; a green local `Test-Path` does **not** mean the file exists on GitHub. Do not commit real sensitive transaction or customer data without permission.

## 13. Limitations and responsible use

- Small dataset and only 73 final-test fraud cases.
- Stratified random split without future-time or customer-disjoint validation.
- Prediction-time availability of several candidate fields not independently established.
- Reduced final holdout ranking performance relative to validation.
- High missed-fraud volume and potentially harmful false blocking proposals.
- Thresholds sensitive to assumed financial costs and operational capacity.
- No deployed drift monitoring, independently verified reviewer outcomes, fairness audit, or approved appeal process.

**Conclusion:** the project demonstrates a technical fraud decision-support workflow, **not a deployable automated fraud-blocking system**. Any operational use requires fresh validation, verified costs, human oversight, and formal governance sign-off.
