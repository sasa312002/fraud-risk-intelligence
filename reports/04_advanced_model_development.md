# Section D — Baseline and Advanced Fraud Models

## Experiment
- Training: 1800 rows (60%), 5-fold stratified CV (random state 42).
- Validation: 600 rows (20%), not used to fit the models or tune parameters.
- Final test: 600 rows (20%), not used for training, tuning or reporting predictions.
- Primary CV objective: **average precision** (a PR-AUC summary).
- Two nonlinear estimators (Random Forest and XGBoost) each used RandomizedSearchCV with 4 candidate settings and 5 folds.
- All imputation, encoding and scaling are fitted inside CV folds; engineered features are generated inside the pipeline.
- Logistic Regression has an original-feature baseline and engineered-feature baseline.
- `class_weight=balanced` for Logistic Regression, `balanced_subsample` for Random Forest, unweighted XGBoost in this phase. Formal imbalance strategy comparisons are deferred to Section G.

## Training cross-validation results
| model | cv_ap_mean | cv_ap_std | cv_roc_auc_mean | cv_roc_auc_std | selection_note |
| --- | --- | --- | --- | --- | --- |
| LogReg / Engineered 39 | 0.2457 | 0.0387 | 0.6634 | 0.0275 | untuned baseline |
| LogReg / Original 23 | 0.2362 | 0.0339 | 0.6716 | 0.0243 | untuned baseline |
| XGBoost / Engineered 39 | 0.2263 | 0.0478 | nan | nan | CV best-of-4; optimistic tuning score |
| Random Forest / Engineered 39 | 0.2118 | 0.0435 | nan | nan | CV best-of-4; optimistic tuning score |

## Untouched-during-tuning validation results
| model | validation_ap | validation_roc_auc | precision_at_05 | recall_at_05 | f1_at_05 | uncalibrated_brier |
| --- | --- | --- | --- | --- | --- | --- |
| LogReg / Original 23 | 0.2424 | 0.6895 | 0.1974 | 0.6301 | 0.3007 | 0.2243 |
| LogReg / Engineered 39 | 0.2203 | 0.6821 | 0.2036 | 0.6164 | 0.3061 | 0.2257 |
| Random Forest / Engineered 39 | 0.2249 | 0.6474 | 0.3191 | 0.2055 | 0.25 | 0.1685 |
| XGBoost / Engineered 39 | 0.213 | 0.6837 | 0.0 | 0.0 | 0.0 | 0.1031 |

## Nonlinear tuning settings
| model | best_cv_ap | best_params_json | num_candidates |
| --- | --- | --- | --- |
| Random Forest / Engineered 39 | 0.2118 | {"classifier__max_depth": 6, "classifier__max_features": "sqrt", "classifier__min_samples_leaf": 2, "classifier__n_estimators": 100} | 4 |
| XGBoost / Engineered 39 | 0.2263 | {"classifier__colsample_bytree": 1.0, "classifier__learning_rate": 0.08, "classifier__max_depth": 2, "classifier__min_child_weight": 1, "classifier__n_estimators": 100, "classifier__reg_lambda": 5.0, "classifier__subsample": 1.0} | 4 |

## Provisional model choice
Selected by training CV AP: **LogReg / Engineered 39**. This is **not** the final production model.

## Interpretation & caveats
- Model-selection CV scores for tuned models are optimistic because hyperparameters were chosen using the same folds. Independent validation results are included as diagnostics.
- All probabilities are **uncalibrated**; 0.50 threshold is provisional and is **not** a fraud-operations decision policy.
- No false-positive/negative business cost matrix has been applied yet; highest PR-AUC may not minimize operational cost.
- The existing full-dataset exploratory data audit limits strong claims of a wholly analyst-unseen test set.
- No complete timestamps or customer IDs are provided, so temporal or group leakage cannot be ruled out by chronological/group split.
- Historical/merchant/IP scores must be confirmed available at authorization time.
- No resampling, calibration or thresholds have been fitted on the final test partition.

## Next
Sections G and F: imbalance strategies and probability calibration. Then cost-sensitive threshold selection and operational Approve / Review / Block policies. Evaluate the finalized policy once on the reserved final test set.
