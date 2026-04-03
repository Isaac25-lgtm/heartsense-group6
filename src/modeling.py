"""
Model Definitions and Training
===============================
Model factories, hyperparameter grids, and training helpers.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np

from src.config import RANDOM_STATE, N_FOLDS
from src.preprocessing import build_preprocessor


def get_model_definitions(pos_weight=1.0):
    """
    Return a dict of model name -> (estimator, param_grid) pairs.

    Parameters
    ----------
    pos_weight : float
        Positive class weight ratio for XGBoost (n_negative / n_positive).
    """
    models = {
        "lr": {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__penalty": ["l1", "l2", "elasticnet"],
                "model__l1_ratio": [0.5],  # only used with elasticnet
            },
            "scale_numeric": True,
        },
        "rf": {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_grid": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [5, 10, 15, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
            },
            "scale_numeric": False,
        },
        "xgb": {
            "name": "XGBoost",
            "estimator": XGBClassifier(
                scale_pos_weight=pos_weight,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "param_grid": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
                "model__reg_alpha": [0, 0.1, 1],
                "model__reg_lambda": [1, 5, 10],
            },
            "scale_numeric": False,
        },
    }
    return models


def build_model_pipeline(model_key, pos_weight=1.0, feature_set="full"):
    """
    Build a complete Pipeline (preprocessor + model) for the given model key.

    Returns
    -------
    pipeline : sklearn Pipeline
    param_grid : dict
    model_name : str
    """
    models = get_model_definitions(pos_weight)
    model_def = models[model_key]

    preprocessor = build_preprocessor(
        scale_numeric=model_def["scale_numeric"],
        feature_set=feature_set,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_def["estimator"]),
    ])

    return pipeline, model_def["param_grid"], model_def["name"]
