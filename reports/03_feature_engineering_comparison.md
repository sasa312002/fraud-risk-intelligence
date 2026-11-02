# Fraud Risk Intelligence — Section C Feature Comparison

## Objective
Compare the original 23 input features with the original plus 16 engineered features (39 total), using the same Logistic Regression settings and leakage-safe evaluation.

## Evaluation protocol
- Training set: 1800 transactions; stratified 5-fold cross-validation.
- Validation set: 600 transactions; used only after CV.
- Final test set: **not examined** for this experiment.
- Numeric imputation and scaling, categorical imputation and encoding are fitted inside each CV pipeline.
- Engineered features are computed inside the engineered pipeline.
- Both classifiers use `class_weight=balanced` and a provisional decision threshold of 0.50.
- `PR-AUC (AP)` denotes average precision, not trapezoidal integration.

## Cross-validation results
| model | PR-AUC (AP) mean | PR-AUC (AP) std | ROC-AUC mean | ROC-AUC std | Precision@0.5 mean | Precision@0.5 std | Recall@0.5 mean | Recall@0.5 std | F1@0.5 mean | F1@0.5 std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original 23 features | 0.2362 | 0.0339 | 0.6716 | 0.0243 | 0.2063 | 0.0176 | 0.6099 | 0.0566 | 0.3081 | 0.0259 |
| Engineered 39 features | 0.2457 | 0.0387 | 0.6634 | 0.0275 | 0.2004 | 0.0167 | 0.5778 | 0.0438 | 0.2974 | 0.0232 |

## Validation results
| model | PR-AUC (AP) | ROC-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 | Brier (uncalibrated) |
| --- | --- | --- | --- | --- | --- | --- |
| Original 23 features | 0.2424 | 0.6895 | 0.1974 | 0.6301 | 0.3007 | 0.2243 |
| Engineered 39 features | 0.2203 | 0.6821 | 0.2036 | 0.6164 | 0.3061 | 0.2257 |

## Main finding
Engineered minus original CV average precision = **+0.0095**.  
Engineered minus original validation average precision = **-0.0221**.

## Important limitations
- The feature comparisons are exploratory and do not establish that the 39-feature model is superior in deployment.
- Several transaction-history / risk fields require confirmation they were available before the authorization decision.
- No transaction timestamp or customer identifier is available for a genuine time-ordered or customer-grouped holdout.
- Previously viewed full-dataset descriptive plots limit claims that the final test set is entirely unseen to the analyst.
- The weighted classifier's output probabilities are **not calibrated**.
- The 0.50 threshold and class weights are not cost-optimized.
- True customer-level behavioral deviation from personal spend history cannot be calculated from this dataset.

## Next stage
Compare/tune Logistic Regression, Random Forest and XGBoost using training-only cross-validation; then perform calibration and business-cost-based decision thresholding without using the final test set.
