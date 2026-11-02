# Section G — Imbalanced Learning

## Objective
Compare three controlled imbalance strategies for Logistic Regression using
original features: unweighted, balanced class weights, and RandomOverSampler
with 1:1 positive/negative training examples.

## Data & leakage controls
- Train: 1800 rows, fraud prevalence 12.11%.
- Validation: 600 rows, fraud prevalence 12.17%.
- Final test: strictly reserved; no predictions were generated.
- 5-fold stratified training CV; every imputer, encoder, scaler, and sampler
  is **fitted inside the fold's training partition only**.
- RandomOverSampler duplicates actual samples; no synthetic one-hot vectors.
- All three pipelines receive exactly the same original input features.

## Training cross-validation
| strategy | cv_ap_mean | cv_ap_std | cv_roc_auc_mean | cv_roc_auc_std | cv_precision_05_mean | cv_precision_05_std | cv_recall_05_mean | cv_recall_05_std | cv_f1_05_mean | cv_f1_05_std | cv_brier_mean | cv_brier_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unweighted | 0.2373 | 0.0412 | 0.6639 | 0.028 | 0.3952 | 0.3465 | 0.032 | 0.0183 | 0.0566 | 0.0323 | 0.1055 | 0.007 |
| Class-weighted | 0.2362 | 0.0339 | 0.6716 | 0.0243 | 0.2063 | 0.0176 | 0.6099 | 0.0566 | 0.3081 | 0.0259 | 0.218 | 0.011 |
| Random oversampling | 0.2357 | 0.0368 | 0.6711 | 0.0297 | 0.2086 | 0.015 | 0.61 | 0.0595 | 0.3107 | 0.0235 | 0.2169 | 0.0109 |

## Validation comparison
| strategy | cv_ap_mean | cv_ap_std | cv_roc_auc_mean | cv_roc_auc_std | cv_precision_05_mean | cv_precision_05_std | cv_recall_05_mean | cv_recall_05_std | cv_f1_05_mean | cv_f1_05_std | cv_brier_mean | cv_brier_std | validation_ap | validation_roc_auc | validation_brier | precision_at_05 | recall_at_05 | f1_at_05 | false_positives_at_05 | missed_fraud_at_05 | true_positives_at_05 | predicted_high_risk_at_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unweighted | 0.2373 | 0.0412 | 0.6639 | 0.028 | 0.3952 | 0.3465 | 0.032 | 0.0183 | 0.0566 | 0.0323 | 0.1055 | 0.007 | 0.2421 | 0.6884 | 0.1022 | 0.4286 | 0.0411 | 0.075 | 4 | 70 | 3 | 7 |
| Class-weighted | 0.2362 | 0.0339 | 0.6716 | 0.0243 | 0.2063 | 0.0176 | 0.6099 | 0.0566 | 0.3081 | 0.0259 | 0.218 | 0.011 | 0.2424 | 0.6895 | 0.2243 | 0.1974 | 0.6301 | 0.3007 | 187 | 27 | 46 | 233 |
| Random oversampling | 0.2357 | 0.0368 | 0.6711 | 0.0297 | 0.2086 | 0.015 | 0.61 | 0.0595 | 0.3107 | 0.0235 | 0.2169 | 0.0109 | 0.2388 | 0.6811 | 0.2234 | 0.207 | 0.6438 | 0.3133 | 180 | 26 | 47 | 227 |

## Limitations
- 0.50 is only a placeholder threshold; optimize business thresholds later.
- Imbalance corrections may change the base rate reflected in predicted
  probabilities; calibration must be evaluated before financial decisions.
- The earlier full-data audit limits claims that test patterns were never examined.
- No real timestamps/customer IDs exist for chronological/group validation.
- A higher recall can increase false positives and review workload.
- Findings on one 600-row validation set have sampling uncertainty.

## Next steps
Section F: compare sigmoid (Platt) and isotonic calibration using disjoint
calibration folds or cross-fitting within training data, then evaluate on
validation data. Section E: optimize Approve / Review / Block thresholds using
explicit review and fraud-loss assumptions. Final test evaluation occurs once
only after policy selection is frozen.
