# Fraud Risk Intelligence - Data Quality Report

## 1. Dataset Overview

- Total transactions: 3000
- Total columns: 25
- Fraud transactions: 364
- Legitimate transactions: 2636
- Overall fraud rate: 12.13%
- Total missing cells: 582
- Transactions with missing values: 542
- Duplicate records: 0
- Duplicate transaction IDs: 0

## 2. Missing Data Assessment

| feature | missing_count | missing_fraud_rate | present_fraud_rate | fraud_rate_difference |
| --- | --- | --- | --- | --- |
| authentication_level | 289 | 19.03 | 11.4 | 7.63 |
| login_to_transaction_minutes | 60 | 11.67 | 12.14 | -0.48 |
| ip_risk_score | 45 | 11.11 | 12.15 | -1.04 |
| transaction_amount | 53 | 9.43 | 12.18 | -2.75 |
| distance_from_usual_location_km | 75 | 5.33 | 12.31 | -6.97 |
| merchant_risk_score | 60 | 3.33 | 12.31 | -8.98 |

### Findings

Authentication-level missingness is associated with
a higher observed fraud rate.

Missingness may contain predictive information.
However, these relationships are observational
and require validation.

## 3. Invalid Value Assessment

| validation_check | invalid_count | status |
| --- | --- | --- |
| Invalid hour | 0 | PASS |
| Invalid day of week | 0 | PASS |
| Invalid transaction amount | 0 | PASS |
| Negative distance | 0 | PASS |
| Negative account age | 0 | PASS |
| Negative device age | 0 | PASS |
| Negative email age | 0 | PASS |
| Invalid merchant risk score | 0 | PASS |
| Invalid IP risk score | 0 | PASS |
| Invalid fraud label | 0 | PASS |
| Invalid new device indicator | 0 | PASS |
| Invalid billing mismatch indicator | 0 | PASS |
| Invalid transaction velocity | 0 | PASS |
| Negative login duration | 0 | PASS |

### Findings

All implemented validation checks passed.

This does not guarantee that the dataset is free
from every possible data-quality issue.

## 4. Outlier Analysis

| feature | Q1 | Q3 | IQR | lower_bound | upper_bound | outlier_count | outlier_percentage | outlier_fraud_rate | maximum_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transaction_amount | 49.23 | 155.26 | 106.03 | -109.81 | 314.3 | 206 | 6.99 | 24.27 | 5538.15 |
| distance_from_usual_location_km | 16.81 | 48.45 | 31.64 | -30.65 | 95.91 | 87 | 2.97 | 22.99 | 1173.27 |
| transactions_last_1h | 1.0 | 3.0 | 2.0 | -2.0 | 6.0 | 21 | 0.7 | 33.33 | 10.0 |
| transactions_last_24h | 6.0 | 9.0 | 3.0 | 1.5 | 13.5 | 88 | 2.93 | 11.36 | 113.0 |
| login_to_transaction_minutes | 6.92 | 21.72 | 14.8 | -15.28 | 43.91 | 167 | 5.68 | 10.78 | 120.0 |
| merchant_risk_score | 0.18 | 0.41 | 0.22 | -0.15 | 0.74 | 18 | 0.61 | 27.78 | 0.82 |
| ip_risk_score | 0.12 | 0.32 | 0.2 | -0.18 | 0.61 | 36 | 1.22 | 22.22 | 0.74 |

### Overall Outlier Comparison

| has_iqr_outlier | transaction_count | fraud_count | fraud_rate |
| --- | --- | --- | --- |
| False | 2435 | 260 | 10.68 |
| True | 565 | 104 | 18.41 |

### Findings

Outlier observations may contain useful fraud signals.

No observations have been removed.

The IQR method identifies statistical extremes,
not necessarily erroneous transactions.

## 5. Categorical Feature Audit

| feature | unique_categories | missing_count |
| --- | --- | --- |
| card_presence | 2 | 0 |
| channel | 4 | 0 |
| country | 7 | 0 |
| device_type | 5 | 0 |
| merchant_category | 10 | 0 |
| authentication_level | 2 | 289 |
| customer_tier | 4 | 0 |

## 6. Data Leakage Assessment

| feature | risk | decision | reason |
| --- | --- | --- | --- |
| transaction_id | Identifier / possible ordering proxy | EXCLUDE | Not a defensible fraud predictor |
| fraud | Target leakage | TARGET ONLY | Must never be included in model inputs |
| prior_chargebacks | Historical information leakage | VERIFY | Must include only chargebacks known before transaction |
| merchant_risk_score | Possible post-event information | VERIFY | Score must be available at prediction time |
| ip_risk_score | Possible post-event information | VERIFY | Score must be computed before authorization |
| authentication_level | Prediction-time availability | VERIFY | Must be known before the fraud decision |
| distance_from_usual_location_km | Historical profile leakage | VERIFY | Customer profile must use past observations only |
| transactions_last_1h | Future transaction leakage | VERIFY | Rolling count must not include future transactions |
| transactions_last_24h | Future transaction leakage | VERIFY | Rolling count must not include future transactions |

### Leakage Decisions

- Exclude transaction_id from predictive features.
- Use fraud only as the target variable.
- Verify point-in-time availability of historical
  and risk-related features before production use.

## 7. Data Treatment Decisions

### Missing Numerical Values

Planned treatment:
Median imputation fitted on training data only.

Missing indicators will be considered where useful.

### Missing Categorical Values

Planned treatment:
Explicit missing category where appropriate.

Authentication-level missingness will be
investigated as a predictive feature.

### Outliers

Planned treatment:
Retain valid extreme observations.

Investigate robust transformations and
model sensitivity to extreme values.

### Categorical Variables

Planned treatment:
Apply appropriate categorical encoding
within the preprocessing pipeline.

### Target Variable

Fraud is the classification target.

It must not appear in the model feature matrix.

### Transaction Identifier

Transaction ID will be excluded from training.

It may be retained separately for transaction
tracking and explanation outputs.

## 8. Validation Strategy

The available dataset does not contain
complete transaction timestamps or customer IDs.

Therefore, a genuine chronological or
customer-grouped split cannot be performed
using the available fields.

A stratified train/validation/test split is
planned, with its limitations documented.

All data-dependent preprocessing will be fitted
inside training or cross-validation pipelines.

The final test set will remain untouched until
the selected model and decision policy are frozen.

## 9. Conclusion

The dataset contains valid transaction records
under the implemented checks, but also includes
missing values and extreme observations.

These observations will be retained during
initial modeling.

Further feature validation, leakage review,
and robustness testing are required before
any production recommendation.
