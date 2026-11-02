# Section I — Fraud Operations Simulation

## Scope
Illustrative replay on **600 VALIDATION transactions** using the provisional, fixed Section F model and Section E decision thresholds. Test set has not been accessed. Scores, IDs and thresholds determine decisions; hindsight fraud labels are never inputs to queue allocation.

## Policy and resources
- Approve for p < 0.152046.
- Manual review for 0.152046 <= p < 0.262003.
- Proposed block for p >= 0.262003.
- Simulated reviewers: 4, each assumed capable of 30 reviews **per modeled batch**.
- Total review capacity per batch: 120; backlog: 0.
- This is not a measured hourly throughput or SLA; no event timestamps/reviewer logs exist.

## Proposed actions
- Approve: 456.
- Manual review: 120.
- Block: 24.
- Review assigned: 120.

## Cost and observed outcome diagnostics
- Projected cost using calibrated probabilities and assumed costs: LKR 990,944.48.
- Outcome-based scenario cost using validation labels **and hypothetical reviewer performance**: LKR 790,140.00.
- Fraud approved: 34.
- Fraud sent to review: 32.
- Fraud proposed for block: 7.
- Legitimate transactions proposed for block: 17.
- Expected fraud prevented assuming 90% reviewer effectiveness: 35.8; not actual operational counts.

## Bootstrap sample variability (fixed-policy validation audit)
|   bootstrap_replicates |   cost_per_transaction_2p5 |   cost_per_transaction_median |   cost_per_transaction_97p5 |
|-----------------------:|---------------------------:|------------------------------:|----------------------------:|
|                    400 |                     936.35 |                       1316.17 |                      1652.9 |

## Required controls / caveats
1. These actions are **proposals for a simulation only**; do not auto-block live transactions.
2. Actual fraud outcomes are not available at decision time and are not used to prioritize the queue.
3. Analyst capacity, review success, review rejection rate, fraud loss and customer friction are hypothetical.
4. Review backlog would remain pending if review demand exceeded modeled capacity; never silently approve overflow.
5. Previously used validation labels for model/calibration selection, so retrospective performance can be optimistic.
6. Without real timestamps, customers or review history, actual SLAs, analyst skill variation and long-run queue dynamics cannot be estimated.
7. Costs, discrimination/fairness implications, regulatory controls and the frozen policy require stakeholder review.
8. A SINGLE untouched final-test evaluation is needed after all design choices are frozen.

## Next section
Section H: Explainability / SHAP for predictions, and Section J: robustness and governance.
