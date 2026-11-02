# Section H — Explainable AI / SHAP

## Objective and model
- Provisional model: **Unweighted Logistic Regression + sigmoid calibration**.
- Fitted base logistic fold members: **3**.
- Global importance sample: **200 TRAINING rows** with **120 TRAINING rows** as background.
- Local examples: **9 VALIDATION rows**, selected from model-proposed actions without labels.
- Final test has not been accessed for feature importance or case selection.

## Scale and fidelity (critical)
SHAP was applied to each underlying logistic regression **decision_function** after its fitted imputation, scaling and one-hot encoding. SHAP contributions are grouped to original raw features, then aggregated across the three base folds. They explain **uncalibrated LOG-ODDS margins** and **do not add up to the ensemble's CALIBRATED FRAUD PROBABILITY**. The probabilities shown next to cases come from the full calibrated ensemble. Averaging grouped fold SHAP values reconstructs the mean base log-odds, but that is NOT the same as the calibrated probability. These are associations, not causal fraud explanations.

Maximum additive base-fold check residual: 9.99e-16.

## Global explanation importance
| feature | mean_abs_shap_log_odds |
| --- | --- |
| merchant_category | 0.3182 |
| transactions_last_1h | 0.2655 |
| new_device | 0.2542 |
| channel | 0.201 |
| merchant_risk_score | 0.176 |
| account_age_days | 0.1644 |
| customer_tier | 0.1553 |
| ip_risk_score | 0.1493 |
| country | 0.1486 |
| authentication_level | 0.1302 |
| device_type | 0.0983 |
| prior_chargebacks | 0.0959 |
| login_to_transaction_minutes | 0.0821 |
| failed_logins_24h | 0.069 |
| card_presence | 0.0671 |
| transaction_amount | 0.0617 |
| billing_address_mismatch | 0.0574 |
| distance_from_usual_location_km | 0.038 |
| device_age_days | 0.0345 |
| email_age_days | 0.0297 |

## Local proposed-action reason examples
| transaction_id | action_proposed | calibrated_fraud_probability | rank_abs_effect | feature | base_log_odds_shap | base_risk_direction | observed_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TXN5000873 | BLOCK | 0.4417 | 1 | transaction_amount | 4.3635 | RAISES | 5107.9496616138995 |
| TXN5000873 | BLOCK | 0.4417 | 2 | new_device | 0.7363 | RAISES | 1 |
| TXN5000873 | BLOCK | 0.4417 | 3 | merchant_category | -0.4107 | LOWERS | Grocery |
| TXN5000873 | BLOCK | 0.4417 | 4 | merchant_risk_score | 0.3033 | RAISES | 0.506 |
| TXN5000873 | BLOCK | 0.4417 | 5 | ip_risk_score | -0.2637 | LOWERS | 0.023 |
| TXN5002538 | BLOCK | 0.4025 | 1 | transaction_amount | 4.7393 | RAISES | 5538.151571050594 |
| TXN5002538 | BLOCK | 0.4025 | 2 | channel | -0.2423 | LOWERS | POS |
| TXN5002538 | BLOCK | 0.4025 | 3 | ip_risk_score | -0.2029 | LOWERS | 0.065 |
| TXN5002538 | BLOCK | 0.4025 | 4 | customer_tier | 0.1783 | RAISES | Basic |
| TXN5002538 | BLOCK | 0.4025 | 5 | merchant_risk_score | 0.1648 | RAISES | 0.414 |
| TXN5002190 | BLOCK | 0.401 | 1 | transactions_last_1h | 0.9024 | RAISES | 6 |
| TXN5002190 | BLOCK | 0.401 | 2 | new_device | 0.7363 | RAISES | 1 |
| TXN5002190 | BLOCK | 0.401 | 3 | merchant_category | 0.3724 | RAISES | Gaming |
| TXN5002190 | BLOCK | 0.401 | 4 | failed_logins_24h | 0.2958 | RAISES | 2 |
| TXN5002190 | BLOCK | 0.401 | 5 | ip_risk_score | 0.2371 | RAISES | 0.369 |
| TXN5002643 | REVIEW | 0.262 | 1 | new_device | 0.7363 | RAISES | 1 |
| TXN5002643 | REVIEW | 0.262 | 2 | authentication_level | 0.6915 | RAISES | nan |
| TXN5002643 | REVIEW | 0.262 | 3 | customer_tier | 0.3187 | RAISES | Platinum |
| TXN5002643 | REVIEW | 0.262 | 4 | channel | 0.2077 | RAISES | E-commerce |
| TXN5002643 | REVIEW | 0.262 | 5 | merchant_risk_score | -0.0836 | LOWERS | 0.249 |
| TXN5000899 | REVIEW | 0.2586 | 1 | merchant_risk_score | 0.484 | RAISES | 0.626 |
| TXN5000899 | REVIEW | 0.2586 | 2 | prior_chargebacks | 0.4404 | RAISES | 1 |
| TXN5000899 | REVIEW | 0.2586 | 3 | customer_tier | 0.3187 | RAISES | Platinum |
| TXN5000899 | REVIEW | 0.2586 | 4 | transactions_last_1h | 0.217 | RAISES | 3 |
| TXN5000899 | REVIEW | 0.2586 | 5 | channel | 0.2077 | RAISES | E-commerce |

## Safeguards and limitations
1. No transaction IDs, fraud labels, or hindsight outcomes are model input features.
2. Local cases are selected by predicted score and proposed action only; no outcome-based cherry-picking.
3. One-hot category terms are grouped to their parent raw feature; do not confuse the feature contribution with verified evidence of fraud.
4. Importance can differ across calibration folds and background choices; group ranking should be checked for stability before deployment.
5. Predictions remain **offline simulated**; SHAP output is not a valid justification to automatically block real bank transactions.
6. Correlations, missing values, data collection changes, class imbalance, customer impact and fairness warrant further investigation.
7. This provisional candidate was selected with validation; an untouched single final-test evaluation is required after policy freeze.

## Artifacts
- `../outputs/07_shap_global_importance.csv`
- `../outputs/07_shap_local_reasons.csv`
- `../outputs/figures/07_shap_global_importance.png`
- `../outputs/figures/07_shap_local_example.png`

## Next
Section J — robustness checks, and Section K — governance and final single test evaluation.
