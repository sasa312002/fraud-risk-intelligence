"""Leakage-safe fraud imbalance strategies for Section G.

All fitted preprocessing and oversampling live in the estimator pipeline.
RandomOverSampler duplicates training rows only; it does NOT synthesize
fractional one-hot category values as plain SMOTE could.
"""
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.linear_model import LogisticRegression
from src.model_training import build_fraud_pipeline

VALID_STRATEGIES = ("Unweighted", "Class-weighted", "Random oversampling")


def build_imbalance_pipeline(X_schema, strategy, seed=42):
    """Construct one of three logistic pipelines using only schema (never labels)."""
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}; choose {VALID_STRATEGIES}")
    logistic = LogisticRegression(
        class_weight="balanced" if strategy == "Class-weighted" else None,
        solver="liblinear", max_iter=2000, random_state=seed,
    )
    base = build_fraud_pipeline(X_schema, logistic, engineered=False)
    if strategy == "Random oversampling":
        return ImbalancedPipeline([
            *base.steps[:-1],
            ("sampler", RandomOverSampler(sampling_strategy=1.0, random_state=seed)),
            base.steps[-1],
        ])
    return base
