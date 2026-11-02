# Section J — Robustness and Stress Testing

## Scope
- Candidate: **Unweighted / sigmoid**, selected provisionally using validation metrics.
- **Frozen thresholds:** review >= 0.152046, block >= 0.262003.
- Validation rows: **600**; training and test features/labels are not passed to perturbation functions.
- Fitted classifier, calibration and thresholds are NEVER refitted/reoptimized inside this stress audit.
- Original validation labels appear only in labeled **diagnostic** metrics. This is NOT an unbiased holdout evaluation because the candidate/policy were already selected with validation.

## Stress results
| scenario | n_transactions | mean_probability | mean_abs_score_change | p95_abs_score_change | action_flip_rate | review_rate | block_rate | capacity_review_exceeded | capacity_block_exceeded | projected_cost_per_transaction | projected_cost_total | validation_ap_diagnostic | validation_roc_auc_diagnostic | validation_brier_diagnostic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 600 | 0.1251 | 0.0 | 0.0 | 0.0 | 0.2 | 0.04 | False | False | 1651.5741 | 990944.4761 | 0.2408 | 0.6928 | 0.1016 |
| numeric_missing_10pct | 600 | 0.1242 | 0.0022 | 0.0137 | 0.0133 | 0.1933 | 0.04 | False | False | 1657.6572 | 994594.3148 | 0.2319 | 0.6872 | 0.1019 |
| numeric_missing_25pct | 600 | 0.1231 | 0.0054 | 0.0233 | 0.0417 | 0.195 | 0.0333 | False | False | 1651.167 | 990700.1844 | 0.227 | 0.6741 | 0.1024 |
| authentication_missing_20pct | 600 | 0.1347 | 0.0096 | 0.0569 | 0.0617 | 0.245 | 0.0483 | True | False | 1619.4651 | 971679.0455 | 0.2222 | 0.6721 | 0.1029 |
| unseen_channel_10pct | 600 | 0.1258 | 0.0012 | 0.0065 | 0.0083 | 0.205 | 0.0383 | True | False | 1650.9582 | 990574.9047 | 0.2403 | 0.6883 | 0.1017 |
| amount_up_25pct | 600 | 0.1266 | 0.0015 | 0.0034 | 0.0083 | 0.205 | 0.0417 | True | False | 1648.5948 | 989156.8701 | 0.2445 | 0.6943 | 0.1018 |
| distance_up_50pct | 600 | 0.1265 | 0.0014 | 0.0035 | 0.0083 | 0.205 | 0.0417 | True | False | 1651.5864 | 990951.8189 | 0.2424 | 0.6927 | 0.1015 |
| velocity_plus2 | 600 | 0.1549 | 0.0298 | 0.0521 | 0.205 | 0.3383 | 0.0733 | True | True | 1556.9568 | 934174.0751 | 0.2416 | 0.6934 | 0.1023 |
| risk_scores_noise_0p05 | 600 | 0.125 | 0.0045 | 0.0135 | 0.0233 | 0.2 | 0.0383 | False | False | 1653.3292 | 991997.5312 | 0.2382 | 0.6929 | 0.1017 |

## Training-versus-validation PSI
| feature | train_validation_psi |
| --- | --- |
| ip_risk_score | 0.0293 |
| merchant_risk_score | 0.0261 |
| transactions_last_24h | 0.0245 |
| account_age_days | 0.0238 |
| distance_from_usual_location_km | 0.0216 |
| transaction_amount | 0.0098 |
| transactions_last_1h | 0.0057 |

## Operational observations
- Baseline review capacity exceeded: **False**.
- Baseline block cap exceeded: **False**.
- Review-capacity exceedance scenarios: ['authentication_missing_20pct', 'unseen_channel_10pct', 'amount_up_25pct', 'distance_up_50pct', 'velocity_plus2'].
- Block-cap exceedance scenarios: ['velocity_plus2'].
- Stress-test outcomes are illustrative: they do not estimate real prevalence or future fraud rates.
- A threshold selected under the original input distribution may violate operational capacity when the distribution changes.

## Critical limitations
1. The data lack complete event timestamps/customer IDs; no temporal out-of-time or grouped generalization test is possible.
2. The final 600-row test holdout was NOT accessed and remains reserved for one-time assessment after full policy freeze.
3. Changing transaction amount, device velocity or risk scores may change real fraud outcomes; keeping validation labels fixed is a diagnostic assumption, not causal evidence.
4. PSI compares a **randomly stratified training/validation split**; it does not prove deployment drift. Thresholds for PSI require organizational baselines and validation.
5. Synthetic noise, unknown categories, and missingness injections do not reproduce every real-world system failure or fraud attack.
6. Fraud costs, review success, and block inconvenience are illustrative, not verified operational costs.
7. Group fairness, point-in-time data availability, provenance, monitoring and rollback require governance review.

## Recommended next steps
- Section K: model governance, model card, monitoring thresholds, rollback process and clear release gates.
- Only after model and policy specifications are frozen should the final test set be evaluated exactly once.
