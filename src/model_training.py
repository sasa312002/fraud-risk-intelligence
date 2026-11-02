"""Reproducible preprocessing and model builders for fraud risk analysis.

Raw inputs must exclude the transaction identifier and fraud target.
Any fitted imputation, scaling and categorical encoding belongs inside
training/CV pipelines, never before data splitting.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from src.feature_engineering import FraudFeatureEngineer


def build_fraud_pipeline(X_schema, estimator, engineered=False):
    """Make estimator pipeline; inspect schema only, do not fit on X_schema."""
    if {"fraud", "transaction_id"} & set(X_schema.columns):
        raise ValueError("Remove target and transaction identifier from X_schema")
    schema = (FraudFeatureEngineer().transform(X_schema)
              if engineered else X_schema)
    numerical = schema.select_dtypes(include=np.number).columns.tolist()
    categorical = schema.select_dtypes(exclude=np.number).columns.tolist()
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numerical", num_pipe, numerical),
        ("categorical", cat_pipe, categorical),
    ])
    steps = []
    if engineered:
        steps.append(("feature_engineering", FraudFeatureEngineer()))
    steps.extend([("preprocess", preprocess), ("classifier", estimator)])
    return Pipeline(steps)


def make_logistic(seed=42):
    return LogisticRegression(
        class_weight="balanced", solver="liblinear",
        max_iter=2000, random_state=seed,
    )


def make_random_forest(seed=42):
    return RandomForestClassifier(
        n_estimators=160, class_weight="balanced_subsample",
        random_state=seed, n_jobs=1,
    )


def make_xgboost(seed=42):
    # Weighting is deliberately deferred to the imbalance comparisons in Section G.
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=seed,
        n_estimators=160, n_jobs=1,
    )


RANDOM_FOREST_PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": [60, 100, 150],
    "classifier__max_depth": [None, 6, 12],
    "classifier__min_samples_leaf": [2, 5, 10],
    "classifier__max_features": ["sqrt", 0.3],
}

XGBOOST_PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": [100, 180, 260],
    "classifier__max_depth": [2, 3, 5],
    "classifier__learning_rate": [0.03, 0.08, 0.15],
    "classifier__min_child_weight": [1, 3, 6],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0],
    "classifier__reg_lambda": [1.0, 5.0],
}
