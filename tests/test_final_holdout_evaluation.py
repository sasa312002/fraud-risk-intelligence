"""Tests for pre-frozen, one-shot final holdout research evaluation."""
from pathlib import Path
import hashlib
import json

from joblib import dump
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.cost_optimization import CostAssumptions
from src.final_holdout_evaluation import evaluate_frozen_holdout, verify_frozen_candidate


@pytest.fixture
def frozen_experiment(tmp_path):
    rng = np.random.default_rng(42)
    train = pd.DataFrame({'a': rng.normal(size=250), 'b': rng.normal(size=250)})
    y_train = (train['a'] + train['b'] > 0.1).astype(int)
    fitted = LogisticRegression().fit(train, y_train)
    model_path = tmp_path / 'frozen.joblib'
    dump(fitted, model_path)
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        'model_sha256': sha, 'artifact_name': model_path.name,
        'feature_names': ['a', 'b'], 'feature_count': 2,
        'review_threshold': 0.15, 'block_threshold': 0.26,
        'calibration_strategy': 'Unweighted', 'calibration_method': 'sigmoid',
        'status': 'RESEARCH_ONLY_NOT_APPROVED_FOR_LIVE_BLOCKING',
        'final_test_evaluation': 'NOT_RUN', 'max_review_rate': 0.2,
        'max_block_rate': 0.05, 'training_rows': 1800, 'validation_rows': 600,
    }
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    test = pd.DataFrame({'a': rng.normal(size=600), 'b': rng.normal(size=600)})
    y = (test['a'] + test['b'] > 0.2).astype(int).to_numpy()
    ids = [f'TEST{i:04d}' for i in range(600)]
    return tmp_path, model_path, manifest_path, test, y, ids


def run(frozen_experiment, **changes):
    root, model, manifest, X, y, ids = frozen_experiment
    kwargs = dict(model_path=model, manifest_path=manifest, X_test=X, y_test=y,
                  transaction_ids=ids, costs=CostAssumptions(),
                  output_dir=root/'outputs', report_dir=root/'reports')
    kwargs.update(changes)
    return evaluate_frozen_holdout(**kwargs)


def test_one_shot_metrics_and_no_refitting(frozen_experiment):
    result = run(frozen_experiment)
    assert result['metrics'].iloc[0]['n_test'] == 600
    assert result['metrics'].iloc[0]['average_precision'] > 0
    assert result['policy'].iloc[0]['approve_count'] + result['policy'].iloc[0]['review_count'] + result['policy'].iloc[0]['block_count'] == 600
    assert all(path.exists() for path in result['file_paths'])
    assert (frozen_experiment[0]/'manifest.json').exists()
    with pytest.raises(FileExistsError, match='already opened'):
        run(frozen_experiment)


def test_hash_mismatch_rejected_before_scoring(frozen_experiment):
    root, model, manifest, X, y, ids = frozen_experiment
    contents = json.loads(manifest.read_text())
    contents['model_sha256'] = '0'*64
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match='SHA-256'):
        run(frozen_experiment)
    assert not (root/'outputs'/'10_final_holdout_one_shot_lock.json').exists()


def test_feature_order_and_target_leakage_guard(frozen_experiment):
    root, model, manifest, X, y, ids = frozen_experiment
    with pytest.raises(ValueError, match='contract'):
        run(frozen_experiment, X_test=X[['b','a']])
    with pytest.raises(ValueError, match='contract'):
        run(frozen_experiment, X_test=X.assign(fraud=y))


def test_changed_cost_contract_rejected(frozen_experiment):
    with pytest.raises(ValueError, match='costs differ'):
        run(frozen_experiment, costs=CostAssumptions(fraud_approved_loss=999))
