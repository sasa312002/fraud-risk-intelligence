"""Section H: faithful SHAP explanations of base logistic margins.

The trained production *candidate* is a CalibratedClassifierCV ensemble of
preprocessed logistic pipelines. SHAP is computed for every fitted base-fold
log-odds function and aggregated to original raw inputs. This decomposition
DOES NOT exactly explain the nonlinear sigmoid calibration or ensemble
probability: display calibrated scores separately.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


@dataclass
class MarginExplanations:
    feature_names: list
    shap_values: np.ndarray            # (rows, raw_features) average signed log-odds effects across base folds
    global_importance: pd.DataFrame    # mean |grouped effect| averaged across rows and folds
    mean_base_margin: np.ndarray       # (rows,) average base model intercept + background effects
    mean_margin: np.ndarray            # (rows,) average base model log odds
    base_values: np.ndarray            # (rows,) average reference model margin
    fold_count: int
    max_residual: float


def _raw_feature(encoded_name: str, raw_names: list[str]) -> str:
    """Map fitted ColumnTransformer output back to its input feature."""
    if encoded_name.startswith('numerical__'):
        stripped = encoded_name[len('numerical__'):]
        if stripped in raw_names:
            return stripped
    if encoded_name.startswith('categorical__'):
        stripped = encoded_name[len('categorical__'):]
        for raw in sorted(raw_names, key=len, reverse=True):
            if stripped == raw or stripped.startswith(raw + '_'):
                return raw
    raise ValueError(f'Could not attribute encoded feature to raw input: {encoded_name}')


def _as_dense(matrix):
    if hasattr(matrix, 'toarray'):
        return matrix.toarray()
    return np.asarray(matrix)


def explain_logistic_margins(fitted_model, X_background: pd.DataFrame,
                             X_explain: pd.DataFrame) -> MarginExplanations:
    """Explain every base fitted logistic model on the additive log-odds scale.

    No labels or ID are accepted. X_background must contain TRAINING-only rows.
    X_explain may be validation cases selected without consulting labels.
    Not an additive explanation of final CALIBRATED probabilities.
    """
    if not isinstance(X_background, pd.DataFrame) or not isinstance(X_explain, pd.DataFrame):
        raise TypeError('Both inputs must be pandas DataFrames with raw model inputs')
    if X_background.empty or X_explain.empty:
        raise ValueError('Provide nonempty background and explanation samples')
    raw_names = list(X_background.columns)
    if len(raw_names) != len(set(raw_names)):
        raise ValueError('Duplicate input columns')
    if set(X_explain.columns) != set(raw_names):
        raise ValueError('Inputs must use the identical raw feature set')
    if {'fraud', 'transaction_id'} & set(raw_names):
        raise ValueError('Target/ID must not enter explanation pipeline')
    X_explain = X_explain.loc[:, raw_names]

    if isinstance(fitted_model, CalibratedClassifierCV):
        fitted_pipes = [fold.estimator for fold in fitted_model.calibrated_classifiers_]
    else:
        fitted_pipes = [fitted_model]
    if not fitted_pipes:
        raise ValueError('Fitted model has no base estimators')

    grouped_per_fold = []
    margins = []
    bases = []
    errors = []
    for pipe in fitted_pipes:
        if not hasattr(pipe, 'named_steps'):
            raise TypeError('Expected a fitted pipeline with preprocess and classifier steps')
        prep = pipe.named_steps['preprocess']
        model = pipe.named_steps['classifier']
        if not isinstance(model, LogisticRegression):
            raise TypeError('This explainer supports logistic base models only')
        bg = _as_dense(prep.transform(X_background))
        ex = _as_dense(prep.transform(X_explain))
        names = list(prep.get_feature_names_out())
        if bg.shape[1] != len(names) or ex.shape[1] != len(names):
            raise ValueError('Transformed feature schema mismatch')
        sh = shap.LinearExplainer(model, bg)(ex)
        values = np.asarray(sh.values, dtype=float)
        base = np.broadcast_to(np.asarray(sh.base_values, dtype=float), (len(ex),)).copy()
        margin = np.asarray(model.decision_function(ex), dtype=float)
        if values.shape != ex.shape:
            raise ValueError('Unexpected SHAP output dimensions')
        error = float(np.max(np.abs(base + values.sum(axis=1) - margin)))
        if error > 1e-5:
            raise RuntimeError(f'Base logistic SHAP additive check failed: {error:.8f}')
        groups = np.zeros((len(ex), len(raw_names)))
        original_idx = {n: j for j, n in enumerate(raw_names)}
        for i, name in enumerate(names):
            groups[:, original_idx[_raw_feature(name, raw_names)]] += values[:, i]
        grouped_per_fold.append(groups)
        margins.append(margin)
        bases.append(base)
        errors.append(error)

    all_values = np.stack(grouped_per_fold, axis=0)
    mean_values = np.mean(all_values, axis=0)
    global_imp = np.mean(np.abs(all_values), axis=(0, 1))
    global_df = pd.DataFrame({'feature': raw_names, 'mean_abs_shap_log_odds': global_imp})
    global_df = global_df.sort_values('mean_abs_shap_log_odds', ascending=False).reset_index(drop=True)
    return MarginExplanations(
        feature_names=raw_names,
        shap_values=mean_values,
        global_importance=global_df,
        mean_base_margin=np.mean(np.stack(bases, axis=0), axis=0),
        mean_margin=np.mean(np.stack(margins, axis=0), axis=0),
        base_values=np.mean(np.stack(bases, axis=0), axis=0),
        fold_count=len(fitted_pipes),
        max_residual=max(errors),
    )


def local_reason_table(explanations: MarginExplanations, transaction_ids,
                       scores, actions, top_k=4) -> pd.DataFrame:
    """Top signed base-logit contributions. Not SHAP of calibrated p or causal proof."""
    ids = list(transaction_ids)
    p = np.asarray(scores, dtype=float)
    acts = list(actions)
    if len(ids) != len(p) or len(ids) != len(acts) or len(ids) != len(explanations.shap_values):
        raise ValueError('Transaction metadata does not match explanation rows')
    if top_k < 1:
        raise ValueError('top_k must be positive')
    records = []
    for i, (tx, score, action) in enumerate(zip(ids, p, acts)):
        impacts = explanations.shap_values[i]
        sorted_idx = np.argsort(-np.abs(impacts), kind='stable')[:top_k]
        for rank, j in enumerate(sorted_idx, start=1):
            records.append({
                'transaction_id': str(tx),
                'action_proposed': str(action),
                'calibrated_fraud_probability': float(score),
                'rank_abs_effect': rank,
                'feature': explanations.feature_names[int(j)],
                'base_log_odds_shap': float(impacts[j]),
                'base_risk_direction': ('RAISES' if impacts[j] > 0 else 'LOWERS' if impacts[j] < 0 else 'NEUTRAL'),
            })
    return pd.DataFrame(records)
