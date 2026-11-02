# Section E — Cost-Sensitive Fraud Policy (Exploratory)

## Scope and integrity
- Provisional model: **Unweighted / sigmoid** from Section F.
- Optimized thresholds over 600 VALIDATION probabilities. Final test labels and probabilities were not used.
- Threshold search used only probability scores and explicitly stated loss assumptions, not outcome labels.
- Validation labels were then used for a diagnostic, *review-outcome-averaged* financial audit.
- This policy is not a production authorization to block transactions.

## Hypothetical costs and operational constraints (LKR)
- Fraud approved: 20,000.
- Legitimate blocked: 1,500.
- Manual review per transaction: 150.
- Reviewer catches fraudulent transactions: 90% (assumed).
- Reviewer incorrectly rejects legitimate transactions: 2% (assumed).
- Maximum review workload: 20% of transactions.
- Maximum block rate: 5% of transactions.
- All figures are assumed scenario parameters, **not** values measured from the dataset.

## Per-transaction expected cost formulas
- APPROVE: `p * fraud_approved_loss`.
- BLOCK: `(1-p) * legitimate_blocked_loss`.
- REVIEW: `review_cost + p * (1-review_detection) * fraud_loss + (1-p) * review_false_rejection * legitimate_blocked_loss`.

## Chosen constrained policy
- APPROVE when `p < 0.152046`.
- REVIEW when `0.152046 <= p < 0.262003`.
- BLOCK when `p >= 0.262003`.
- Predicted total cost on validation: 990,944.48 illustrative LKR.
- Review rate: 20.0%; block rate: 4.0%.

## Comparison against baselines
| policy | approve_count | review_count | block_count | projected_cost_total | validation_outcome_cost_total |
| --- | --- | --- | --- | --- | --- |
| Approve all | 600 | 0 | 0 | 1501171.13 | 1460000.0 |
| Binary block at 0.50 | 600 | 0 | 0 | 1501171.13 | 1460000.0 |
| Optimized triage (capacity constrained) | 456 | 120 | 24 | 990944.48 | 790140.0 |

## Sensitivity analysis
| fraud_approved_loss | legitimate_blocked_loss | review_detection_rate | review_threshold | block_threshold | projected_cost_total |
| --- | --- | --- | --- | --- | --- |
| 10000 | 1500 | 0.7 | 0.154 | 0.272 | 566665.753 |
| 10000 | 1500 | 0.9 | 0.154 | 0.272 | 520518.547 |
| 10000 | 5000 | 0.7 | 0.162 | 1.0 | 593987.969 |
| 10000 | 5000 | 0.9 | 0.162 | 1.0 | 541425.266 |
| 20000 | 1500 | 0.7 | 0.154 | 0.272 | 1091986.199 |
| 20000 | 1500 | 0.9 | 0.154 | 0.272 | 999691.786 |
| 20000 | 5000 | 0.7 | 0.154 | 0.272 | 1146458.583 |
| 20000 | 5000 | 0.9 | 0.154 | 0.272 | 1054164.17 |
| 40000 | 1500 | 0.7 | 0.154 | 0.272 | 2142627.09 |
| 40000 | 1500 | 0.9 | 0.154 | 0.272 | 1958038.265 |
| 40000 | 5000 | 0.7 | 0.154 | 0.272 | 2197099.474 |
| 40000 | 5000 | 0.9 | 0.154 | 0.272 | 2012510.649 |

## Limitations and recommended action
- **Review outcomes are simulated**: the data does not contain human-review outcomes, throughput, or actual cost per mistake.
- Thresholds and assumed costs must be endorsed by operations and risk stakeholders.
- Max review/block percentages are illustrative; real queue capacities are unknown.
- Monetary total compares policies on a 600-transaction validation sample; no extrapolation to annual savings is warranted.
- Validation was already used in prior preliminary model/calibration selection; estimates are subject to selection optimism.
- Calibration estimates have noise from only 73 validation fraud cases. Sensitivity analysis is essential.
- No claim of real-world financial savings or guaranteed reviewer detection is justified.
- The same decisions and cost rules need **one final frozen-policy evaluation on the untouched test set** after all model/policy selection decisions.

## Next phase
Use selected policy as a *provisional* input to the fraud operations simulation,
then stress-test reviewer accuracy, capacity, score shift, and false-block impacts.
