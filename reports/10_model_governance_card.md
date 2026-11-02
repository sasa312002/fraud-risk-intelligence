# Model Card & Governance Review — Fraud Risk Intelligence

**Deployment decision: NO-GO (research prototype).**

## Candidate policy freeze
- Model: Unweighted + sigmoid logistic calibration.
- SHA256: `87217ee986fbed6763df7111294c051c94fb7934ea9ce3e185064c6841bce517`.
- Review at >= 0.152046; propose block at >= 0.262003.
- 1,800 training and 600 validation rows; independent test status: NOT_RUN.
- Policy actions are proposals only; manual review and contest/appeal required before blocking.

## Governance gates
| gate | passed | severity | evidence |
| --- | --- | --- | --- |
| baseline_review_capacity | True | Operational | 20.00% review vs 20.00% cap |
| baseline_block_capacity | True | Operational | 4.00% block vs 5.00% cap |
| stress_review_capacity | False | Critical | Breaching scenarios: authentication_missing_20pct, unseen_channel_10pct, amount_up_25pct, distance_up_50pct, velocity_plus2 |
| stress_block_capacity | False | Critical | Breaching scenarios: velocity_plus2 |
| prediction_time_feature_provenance | False | Critical | Pre-authorization availability for 7 VERIFY fields not independently established |
| out_of_time_or_customer_group_validation | False | Critical | Full timestamps and customer identifiers unavailable |
| real_review_outcomes_and_costs_verified | False | Critical | Reviewer effectiveness and business loss amounts are assumptions |
| fairness_and_appeal_process_reviewed | False | Critical | No protected-class fairness audit or appeal-process evaluation completed |
| independent_final_holdout_evaluated | False | Critical | Reserved 600-row final test set has not been scored |
| live_monitoring_and_human_signoff | False | Critical | Monitoring ownership, escalation and deployment approval not validated |

Failed/unverified gates: **stress_review_capacity, stress_block_capacity, prediction_time_feature_provenance, out_of_time_or_customer_group_validation, real_review_outcomes_and_costs_verified, fairness_and_appeal_process_reviewed, independent_final_holdout_evaluated, live_monitoring_and_human_signoff**.

## Monitoring contract (proposal, not deployed)
| metric | cadence | warning_trigger | critical_trigger | action |
| --- | --- | --- | --- | --- |
| review_rate | Per batch | >= 18.0% | > 20.0% | Route overflow to an explicit, human-approved queue; never auto-expand capacity |
| block_rate | Per batch | >= 4.5% | > 5.0% | Pause auto-block rollout, escalate to fraud operations and review policy |
| authentication_missing_rate | Daily | Historical baseline + 5 percentage points | + 10 percentage points | Inspect upstream auth logging and data contracts |
| unseen_category_rate | Daily | > 1% | > 5% | Check encoder schema and source changes |
| numeric_psi | Weekly | >= 0.10 | >= 0.25 | Investigate segment drift; confirm reference window is chronological |
| brier_score | After delayed labels | + 10% vs holdout | + 25% vs holdout | Trigger calibration investigation on mature labels, not immediate refit |
| average_precision | After delayed labels | - 10% vs holdout | - 20% vs holdout | Investigate label delays, sampling and class prevalence |
| manual_review_backlog | Per batch | > 0 pending cases | > reviewer capacity | Escalate analyst coverage; avoid assumed SLA claims |

## Risk register
| risk_id | risk | severity | evidence | mitigation | proposed_owner |
| --- | --- | --- | --- | --- | --- |
| R01 | Capacity overflow under shift | High | Observed in offline stress replay: authentication_missing_20pct, unseen_channel_10pct, amount_up_25pct, distance_up_50pct, velocity_plus2 | Human approval for overflow routing; monitor both action rates | Fraud Operations |
| R02 | Large false-block count | High | 17 legitimate proposed blocks on validation under current illustrative policy | Human override, appeals, two-person controls for high-impact actions | Risk Owner |
| R03 | Fraud approved | High | 34 of 73 fraud labels approved in retrospective audit | Fraud-loss surveillance; human review/threshold reevaluation on fresh data | Fraud Operations |
| R04 | Feature leakage / pre-authorization availability | High | Multiple fields remain VERIFY | Signed point-in-time feature contract and pipeline audit | Data Engineering |
| R05 | Temporal and customer generalization unknown | High | No customer IDs or full timestamps | Secure chronological and entity-level external validation | Model Risk |
| R06 | Repeated selection on validation | High | Model, calibration and cost policy chosen on same validation batch | Freeze whole pipeline; one-time final holdout, then fresh data for further revisions | Model Risk |
| R07 | Illustrative reviewer/cost assumptions | High | 90% reviewer detection and LKR cost matrix unverified | Replace with measured review and cost observations | Finance / Operations |
| R08 | Fairness, privacy and contestability | High | Sensitive attributes/consent and appeals not assessed | Minimize retained identifiers, bias assessment and appeal protocol | Compliance |
| R09 | Label delay, data drift and unseen categories | Medium | No live monitoring established | Monitor missingness, PSI, score mix and post-label outcomes | MLOps |
| R10 | Small holdouts and uncertain ranking gains | Medium | Validation fraud count only 73 | Bootstrap intervals; independent external evaluation | Data Science |

## Stress evidence
- Capacity-breach scenarios: authentication_missing_20pct, unseen_channel_10pct, amount_up_25pct, distance_up_50pct, velocity_plus2.
- Fixed thresholds are not robust to velocity +2 and certain missingness shifts.
- Any stress cost decrease is hypothetical, not verified savings.

## Intended use and boundaries
Offline educational fraud *risk ranking* and human-review policy simulation only; not live automated blocking.
Validation results are not independent of calibration/policy selection.
All cost amounts, manual-review efficacy, throughput and action budgets are hypothetical.
No chronological validation, entity-group split, protected-attribute fairness audit, confirmed feature provenance, or real user appeal mechanism.

## Required next step
Freeze candidate and thresholds (this manifest); design a **one-time** final test evaluation with no additional tuning.
If the final test is used to modify model or thresholds, a *new* untouched evaluation set is needed.
