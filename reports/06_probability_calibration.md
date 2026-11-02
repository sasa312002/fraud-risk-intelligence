# Section F — Probability Calibration

## Purpose and protocol
- Evaluated 3 Logistic Regression class-imbalance strategies with raw, sigmoid and isotonic probability output.
- All fitting and calibration uses only 1,800 training rows (218 fraud cases); 3-fold stratified cross-fitting.
- Preprocessing and any resampling are inside estimator training folds; calibration targets are untouched held-out folds.
- Compared 9 configurations on 600-row validation set (73 fraud). The final test set was not scored.
- Brier score, log loss and expected calibration error (10 equal-width bins) assess probability quality; AP and ROC-AUC assess ranking.
- Reliability plots use eight quantile bins. ECE and isotonic estimates have binning and sample-size uncertainty.

## Validation metrics
| strategy | calibration_method | average_precision | roc_auc | brier | log_loss | ece_10 | mean_predicted_fraud | observed_fraud |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Class-weighted | isotonic | 0.2324 | 0.6868 | 0.1021 | 0.3477 | 0.0211 | 0.1278 | 0.1217 |
| Class-weighted | sigmoid | 0.2426 | 0.6907 | 0.1025 | 0.3495 | 0.0102 | 0.125 | 0.1217 |
| Class-weighted | raw | 0.2424 | 0.6895 | 0.2243 | 0.6546 | 0.3274 | 0.449 | 0.1217 |
| Random oversampling | isotonic | 0.2477 | 0.6809 | 0.1017 | 0.346 | 0.0099 | 0.1276 | 0.1217 |
| Random oversampling | sigmoid | 0.2376 | 0.6836 | 0.1022 | 0.3491 | 0.0112 | 0.1254 | 0.1217 |
| Random oversampling | raw | 0.2388 | 0.6811 | 0.2234 | 0.6518 | 0.3207 | 0.4424 | 0.1217 |
| Unweighted | sigmoid | 0.2408 | 0.6928 | 0.1016 | 0.3467 | 0.0155 | 0.1251 | 0.1217 |
| Unweighted | isotonic | 0.2438 | 0.6843 | 0.1017 | 0.3466 | 0.0194 | 0.1272 | 0.1217 |
| Unweighted | raw | 0.2421 | 0.6884 | 0.1022 | 0.347 | 0.0341 | 0.1295 | 0.1217 |

## Provisional selection
Rule: choose lowest validation Brier score among candidates within 0.02 average-precision points of the highest validation AP.
Selected: **Unweighted / sigmoid**.
AP=0.2408; Brier=0.1016; ROC-AUC=0.6928.

## Limitations and next steps
- Class weighting or resampling can distort the apparent base rate. Calibration seeks to correct this.
- Small calibration folds make isotonic regression potentially unstable.
- Validation-based model/method selection may be optimistic. Do not claim production readiness from these results.
- No automated blocking threshold is chosen in this section; Section E will compare explicit costs, review capacity and thresholds.
- Final test data remains untouched and must be used once after policy selection is frozen.
- Some feature availability and temporal leakage assumptions cannot be verified with the provided records.
