# Fraud Risk Intelligence
# Advanced Exploratory Data Analysis Report

## 1. Executive Summary

This report presents the advanced exploratory
data analysis performed on the fraud transaction
training dataset.

The primary objective is to investigate fraud
patterns, nonlinear relationships, and feature
interactions before model development.

The analysis uses the training subset to reduce
the risk of using final-test information for
feature engineering decisions.

## 2. Training Dataset Overview

- Total training transactions: 1800
- Fraudulent transactions: 218
- Legitimate transactions: 1582
- Overall fraud rate: 12.11%

The dataset exhibits class imbalance.

Evaluation should prioritize PR-AUC, recall,
precision, calibration, and business cost,
rather than accuracy alone.

## 3. Fraud Rate by Transaction Channel

| channel | total_transactions | fraud_transactions | fraud_rate |
| --- | --- | --- | --- |
| E-commerce | 709 | 100 | 14.1 |
| Mobile App | 347 | 45 | 12.97 |
| ATM | 163 | 16 | 9.82 |
| POS | 581 | 57 | 9.81 |

### Findings

Transaction channels exhibit different
observed fraud rates.

Channel information may provide useful
predictive information.

The relationship is observational and
does not establish causality.

## 4. Fraud Rate by Merchant Category

| merchant_category | total_transactions | fraud_transactions | fraud_rate |
| --- | --- | --- | --- |
| Jewelry | 63 | 16 | 25.4 |
| Fashion | 234 | 40 | 17.09 |
| Gaming | 147 | 24 | 16.33 |
| Restaurants | 245 | 31 | 12.65 |
| Healthcare | 132 | 16 | 12.12 |
| Travel | 134 | 16 | 11.94 |
| Electronics | 195 | 22 | 11.28 |
| Grocery | 292 | 25 | 8.56 |
| Digital Services | 188 | 16 | 8.51 |
| Fuel | 170 | 12 | 7.06 |

### Findings

Merchant categories exhibit variation
in observed fraud rates.

Sample sizes should be considered
when interpreting category-level differences.

## 5. Fraud Rate by Customer Tier

| customer_tier | total_transactions | fraud_transactions | fraud_rate |
| --- | --- | --- | --- |
| Platinum | 173 | 26 | 15.03 |
| Basic | 663 | 90 | 13.57 |
| Gold | 406 | 44 | 10.84 |
| Silver | 558 | 58 | 10.39 |

### Findings

Customer tiers exhibit differences
in fraud rates.

Customer tier alone should not be used
to justify adverse transaction decisions.

## 6. Fraud Rate by Device Type

| device_type | total_transactions | fraud_transactions | fraud_rate |
| --- | --- | --- | --- |
| iOS | 522 | 66 | 12.64 |
| Android | 700 | 86 | 12.29 |
| macOS | 197 | 24 | 12.18 |
| Windows | 217 | 24 | 11.06 |
| Other | 164 | 18 | 10.98 |

### Findings

Fraud rates across device types are
relatively similar.

Device type may still contribute through
interactions with other transaction features.

## 7. Transaction Amount Analysis

| amount_bin | total_transactions | fraud_transactions | fraud_rate | amount_range |
| --- | --- | --- | --- | --- |
| (4.119, 43.89] | 354 | 41 | 11.58 | (4.119, 43.89] |
| (43.89, 72.514] | 353 | 33 | 9.35 | (43.89, 72.514] |
| (72.514, 109.364] | 354 | 44 | 12.43 | (72.514, 109.364] |
| (109.364, 182.384] | 353 | 38 | 10.76 | (109.364, 182.384] |
| (182.384, 5502.776] | 354 | 60 | 16.95 | (182.384, 5502.776] |

### Findings

The highest transaction amount quantile
exhibits elevated fraud risk.

The relationship is not strictly monotonic,
suggesting potential nonlinear effects.

## 8. Transaction Velocity Analysis

| velocity_group | total_transactions | fraud_transactions | fraud_rate |
| --- | --- | --- | --- |
| 0 | 197 | 18 | 9.14 |
| 1 | 432 | 35 | 8.1 |
| 2 | 508 | 61 | 12.01 |
| 3-4 | 541 | 83 | 15.34 |
| 5-6 | 111 | 16 | 14.41 |
| 7+ | 11 | 5 | 45.45 |

### Findings

Higher transaction velocity groups
show elevated observed fraud rates.

The highest velocity group has a
small sample size.

Therefore, its estimated fraud rate
should be interpreted cautiously.

## 9. Distance Analysis

| distance_bin | total_transactions | fraud_transactions | fraud_rate | distance_range |
| --- | --- | --- | --- | --- |
| (0.489, 14.192] | 352 | 40 | 11.36 | (0.489, 14.192] |
| (14.192, 24.278] | 351 | 39 | 11.11 | (14.192, 24.278] |
| (24.278, 35.642] | 351 | 51 | 14.53 | (24.278, 35.642] |
| (35.642, 54.728] | 351 | 45 | 12.82 | (35.642, 54.728] |
| (54.728, 1173.269] | 352 | 39 | 11.08 | (54.728, 1173.269] |

### Findings

Distance does not show a consistently
increasing fraud rate across quantiles.

Distance may be more informative when
combined with device-related indicators.

## 10. Account Age Analysis

| account_age_bin | total_transactions | fraud_transactions | fraud_rate | account_age_range |
| --- | --- | --- | --- | --- |
| (0.999, 336.0] | 361 | 50 | 13.85 | (0.999, 336.0] |
| (336.0, 688.6] | 359 | 45 | 12.53 | (336.0, 688.6] |
| (688.6, 1045.6] | 360 | 50 | 13.89 | (688.6, 1045.6] |
| (1045.6, 1409.0] | 360 | 40 | 11.11 | (1045.6, 1409.0] |
| (1409.0, 1795.0] | 360 | 33 | 9.17 | (1409.0, 1795.0] |

### Findings

Older account groups generally exhibit
lower observed fraud rates.

The relationship is not strictly monotonic.

## 11. Merchant Risk Score Analysis

| risk_bin | total_transactions | fraud_transactions | fraud_rate | risk_range |
| --- | --- | --- | --- | --- |
| (0.008, 0.159] | 356 | 40 | 11.24 | (0.008, 0.159] |
| (0.159, 0.246] | 355 | 34 | 9.58 | (0.159, 0.246] |
| (0.246, 0.324] | 351 | 42 | 11.97 | (0.246, 0.324] |
| (0.324, 0.429] | 351 | 47 | 13.39 | (0.324, 0.429] |
| (0.429, 0.818] | 354 | 55 | 15.54 | (0.429, 0.818] |

### Findings

Higher merchant risk score quantiles
generally show elevated fraud rates.

## 12. IP Risk Score Analysis

| risk_bin | total_transactions | fraud_transactions | fraud_rate | risk_range |
| --- | --- | --- | --- | --- |
| (0.003, 0.1] | 358 | 40 | 11.17 | (0.003, 0.1] |
| (0.1, 0.174] | 355 | 37 | 10.42 | (0.1, 0.174] |
| (0.174, 0.24] | 359 | 37 | 10.31 | (0.174, 0.24] |
| (0.24, 0.343] | 352 | 43 | 12.22 | (0.24, 0.343] |
| (0.343, 0.738] | 355 | 58 | 16.34 | (0.343, 0.738] |

### Findings

The highest IP risk score quantile
exhibits elevated fraud risk.

## 13. New Device and Distance Interaction

| new_device | distance_group | total_transactions | fraud_transactions | fraud_rate |
| --- | --- | --- | --- | --- |
| 0 | (0.489, 16.51] | 365 | 30 | 8.22 |
| 0 | (16.51, 29.96] | 376 | 49 | 13.03 |
| 0 | (29.96, 49.16] | 371 | 46 | 12.4 |
| 0 | (49.16, 1173.269] | 357 | 30 | 8.4 |
| 1 | (0.489, 16.51] | 75 | 16 | 21.33 |
| 1 | (16.51, 29.96] | 63 | 15 | 23.81 |
| 1 | (29.96, 49.16] | 68 | 9 | 13.24 |
| 1 | (49.16, 1173.269] | 82 | 19 | 23.17 |

### Findings

New-device transactions exhibit higher
fraud rates across the analyzed
distance groups.

The magnitude of the difference varies
across groups.

These results suggest potential
interaction effects requiring
model-based verification.

## 14. Transaction Amount and Merchant Risk Interaction

| amount_group | merchant_risk_group | total_transactions | fraud_transactions | fraud_rate |
| --- | --- | --- | --- | --- |
| Low | Low | 195 | 17 | 8.72 |
| Low | Medium | 182 | 15 | 8.24 |
| Low | High | 202 | 32 | 15.84 |
| Medium | Low | 201 | 22 | 10.95 |
| Medium | Medium | 190 | 15 | 7.89 |
| Medium | High | 187 | 25 | 13.37 |
| High | Low | 184 | 26 | 14.13 |
| High | Medium | 208 | 30 | 14.42 |
| High | High | 186 | 34 | 18.28 |

### Findings

The combination of high transaction amount
and high merchant risk exhibits elevated
observed fraud risk.

Both features should be evaluated individually
and through interaction features.

## 15. Authentication Level and Channel Interaction

| authentication_level | channel | total_transactions | fraud_transactions | fraud_rate |
| --- | --- | --- | --- | --- |
| Missing | ATM | 18 | 2 | 11.11 |
| Missing | E-commerce | 59 | 16 | 27.12 |
| Missing | Mobile App | 33 | 8 | 24.24 |
| Missing | POS | 51 | 7 | 13.73 |
| Standard | ATM | 65 | 7 | 10.77 |
| Standard | E-commerce | 330 | 45 | 13.64 |
| Standard | Mobile App | 170 | 20 | 11.76 |
| Standard | POS | 254 | 28 | 11.02 |
| Strong | ATM | 80 | 7 | 8.75 |
| Strong | E-commerce | 320 | 39 | 12.19 |
| Strong | Mobile App | 144 | 17 | 11.81 |
| Strong | POS | 276 | 22 | 7.97 |

### Findings

Missing authentication information in
E-commerce and Mobile App transactions
is associated with elevated observed fraud rates.

Missingness indicators and channel interactions
should be evaluated during feature engineering.

## 16. Spearman Correlation Analysis

### Top Numerical Correlations With Fraud

| feature | spearman_correlation |
| --- | --- |
| new_device | 0.1109 |
| transactions_last_1h | 0.0969 |
| transactions_last_24h | 0.073 |
| prior_chargebacks | 0.0646 |
| merchant_risk_score | 0.064 |
| ip_risk_score | 0.0561 |
| transaction_amount | 0.0548 |
| account_age_days | -0.0536 |
| login_to_transaction_minutes | -0.0515 |
| billing_address_mismatch | 0.0457 |

### Findings

Individual numerical feature correlations
with the fraud target are generally weak.

New-device and transaction-velocity indicators
show some of the largest observed
absolute correlations.

Weak marginal correlations do not imply
that nonlinear models or interactions
will be ineffective.

Correlation does not establish causation.

## 17. Feature Engineering Recommendations

The following candidate features will be
investigated in the next project phase:

1. velocity_ratio
2. hourly_velocity_intensity
3. transaction_amount_log
4. amount_merchant_risk_interaction
5. new_device_distance_interaction
6. combined_risk_score
7. authentication_level_missing
8. authentication_channel_interaction
9. account_device_age_ratio
10. failed_login_ip_risk
11. billing_device_interaction
12. high_velocity_indicator

These features are hypotheses derived
from exploratory analysis.

Their usefulness must be measured through
leakage-safe cross-validation and
comparison with baseline features.

## 18. Limitations

- The dataset contains only 3,000 transactions.
- Some subgroup analyses have small sample sizes.
- Full transaction timestamps are unavailable.
- Customer identifiers are unavailable.
- True chronological validation is not possible.
- Several historical and risk-related features
  require prediction-time availability checks.
- Earlier preliminary audits examined the full
  dataset, limiting claims of a completely
  unseen test set.
- Observed associations are not causal evidence.
- Correlation analysis does not capture all
  nonlinear relationships or interactions.

## 19. Next Steps

The next development phase is Feature Engineering.

The implementation will:

- Create at least 10 derived features.
- Preserve original data.
- Avoid target-derived leakage.
- Use prediction-time available information.
- Compare models with and without
  engineered features.
- Evaluate feature stability and robustness.

## 20. Conclusion

The exploratory analysis identified several
potential fraud-risk signals involving
transaction velocity, transaction amount,
new-device activity, risk scores,
and authentication missingness.

Multiple feature interactions were observed.

These findings motivate the development
of nonlinear classification models and
defensible engineered features.

Final predictive usefulness must be established
using appropriate training, validation,
and business-cost evaluation.

## Visualization Reference

The accompanying Jupyter notebook contains
the detailed EDA visualizations and
corresponding statistical tables.

Notebook:
../notebooks/01_fraud_risk_analysis.ipynb
