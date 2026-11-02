# Final Holdout — One-Time Evaluation (Research Prototype)

## Evaluation contract
- Evaluated the PRE-FROZEN **Unweighted / sigmoid** model.
- Frozen model SHA256: `87217ee986fbed6763df7111294c051c94fb7934ea9ce3e185064c6841bce517`.
- The manifest was created before final test evaluation and remained unchanged.
- Policy thresholds NOT refit/re-optimized: **review >= 0.152046**, **block >= 0.262003**.
- Final test rows: **600**, fraud labels: **73** (12.17%).
- Feature order and model hash checked against manifest; no target/ID entered model inputs.

## Predictive metrics — final stratified holdout
- Average precision (PR-AUC summary): **0.1932**.
- ROC-AUC: **0.6640**.
- Brier score: **0.1039**.
- Log loss: **0.3550**.
- ECE (10 bins): **0.0215**.

## Frozen policy on final test
- Approve: 474, Review: 114, Proposed block: 12.
- Review rate: 19.00% (cap: 20.00%); breach: **False**.
- Block rate: 2.00% (cap: 5.00%); breach: **False**.
- Fraud approved: 46; fraud reviewed: 23; fraud proposed-blocked: 4.
- Legitimate transactions proposed-blocked: 8.
- Outcome-based *hypothetical* total cost: LKR 997,830.00.
- Approve-all *hypothetical* total cost: LKR 1,460,000.00.
- Difference: LKR 462,170.00 (NOT verified realized savings).

## Warnings and limitations
- This was one-time evaluation on the reserved random-stratified 600-row test slice.
- Earlier whole-dataset exploratory inspections mean the test was not fully unseen in the strictest sense.
- Model and thresholds were repeatedly compared on validation, so selection uncertainty remains.
- Missing full transaction timestamps/customer IDs prevents chronological and customer-group validation.
- Seven prediction-time features remain unverified, including historical and risk-score provenance.
- Financial costs, reviewer outcomes and assumed effectiveness have not been independently verified.
- Test labels have **not** been used to select a new model, change thresholds or calibrate scores.
- Any future model/policy modification requires NEW independent data; do not reuse this holdout for selection.
- Section K critical governance gates remain unresolved; **NO-GO for live automated blocking**.
- Saving the score audit is for synthetic research only. Real deployments need appropriate data minimization/access controls.
