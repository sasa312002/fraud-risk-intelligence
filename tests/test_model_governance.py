import hashlib
import json
import pandas as pd
import pytest
from src.model_governance import (
    assess_governance_gates, build_monitoring_contract, build_risk_register,
    freeze_candidate_manifest, model_card_markdown,
)


@pytest.fixture
def stress():
    return pd.DataFrame([
        {"scenario": "baseline", "review_rate": .2, "block_rate": .04,
         "capacity_review_exceeded": False, "capacity_block_exceeded": False,
         "action_flip_rate": 0},
        {"scenario": "velocity_plus2", "review_rate": .34, "block_rate": .073,
         "capacity_review_exceeded": True, "capacity_block_exceeded": True,
         "action_flip_rate": .205},
    ])


def test_governance_catches_capacity_failure_and_never_approves_unverified(stress):
    gates = assess_governance_gates(stress, .20, .05)
    values = gates.set_index('gate').passed.to_dict()
    assert values['baseline_review_capacity']
    assert values['baseline_block_capacity']
    assert not values['stress_review_capacity']
    assert not values['stress_block_capacity']
    assert not values['independent_final_holdout_evaluated']
    assert not gates.passed.all()


def test_governance_requires_valid_stress_evidence(stress):
    with pytest.raises(ValueError, match='Missing stress columns'):
        assess_governance_gates(stress.drop(columns='review_rate'), .2, .05)
    with pytest.raises(ValueError, match='exactly one baseline'):
        assess_governance_gates(stress[stress.scenario != 'baseline'], .2, .05)
    with pytest.raises(ValueError, match='Capacity'):
        assess_governance_gates(stress, 1., .05)


def test_monitoring_risk_register_and_model_card(stress, tmp_path):
    monitoring = build_monitoring_contract(.2, .05)
    risks = build_risk_register(stress)
    assert 'review_rate' in monitoring.metric.values
    assert 'block_rate' in monitoring.metric.values
    assert 'R01' in risks.risk_id.values
    artifact = tmp_path/'model.joblib'; artifact.write_bytes(b'fake-research-artifact')
    manifest = freeze_candidate_manifest(artifact, tmp_path/'manifest.json',
             model_choice=('Unweighted', 'sigmoid'), review_threshold=.152,
             block_threshold=.262, feature_names=['amount', 'velocity'],
             training_rows=1800, validation_rows=600, review_cap=.2, block_cap=.05)
    report = model_card_markdown(assess_governance_gates(stress, .2, .05),
                                 monitoring, risks, manifest, stress)
    assert 'NO-GO' in report and 'velocity_plus2' in report
    assert 'NOT_RUN' in report


def test_freeze_sha256_and_reject_leakage_fields(tmp_path):
    f = tmp_path/'m.joblib'; f.write_bytes(b'abc123')
    kw = dict(model_choice=('Unweighted','sigmoid'), review_threshold=.15,
              block_threshold=.26, training_rows=1800, validation_rows=600,
              review_cap=.20, block_cap=.05)
    manifest = freeze_candidate_manifest(f, tmp_path/'m.json', feature_names=['hour'], **kw)
    assert manifest['model_sha256'] == hashlib.sha256(b'abc123').hexdigest()
    assert json.loads((tmp_path/'m.json').read_text())['final_test_evaluation'] == 'NOT_RUN'
    with pytest.raises(ValueError, match='Target'):
        freeze_candidate_manifest(f, tmp_path/'bad.json', feature_names=['fraud'], **kw)
    with pytest.raises(ValueError, match='Thresholds'):
        freeze_candidate_manifest(f, tmp_path/'bad.json', feature_names=['hour'], **(kw | dict(review_threshold=.4)))
