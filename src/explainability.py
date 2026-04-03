"""
Explainability Utilities
========================
SHAP computation, feature importance extraction, and error analysis helpers.
"""

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X, feature_names=None, model_type="tree"):
    """
    Compute SHAP values for the given model and data.

    Parameters
    ----------
    model : fitted model (not pipeline -- extract the model step first)
    X : array-like, preprocessed features
    feature_names : list of str
    model_type : str, 'tree' or 'linear'

    Returns
    -------
    shap_values : shap.Explanation
    explainer : shap.Explainer
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.Explainer(model, X)

    shap_values = explainer(X)

    if feature_names is not None:
        shap_values.feature_names = feature_names

    return shap_values, explainer


def get_feature_importance_from_shap(shap_values):
    """
    Compute mean absolute SHAP values per feature.

    Returns
    -------
    importance_df : DataFrame with columns [feature, mean_abs_shap], sorted descending
    """
    vals = np.abs(shap_values.values)
    if vals.ndim == 3:
        # Multi-output: take positive class
        vals = vals[:, :, 1]

    mean_abs = vals.mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance_df


def extract_lr_coefficients(pipeline, feature_names):
    """
    Extract Logistic Regression coefficients from a fitted pipeline.

    Returns
    -------
    coef_df : DataFrame with columns [feature, coefficient, abs_coefficient, direction]
    """
    lr_model = pipeline.named_steps["model"]
    coefs = lr_model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
        "direction": ["Increases risk" if c > 0 else "Decreases risk" for c in coefs],
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    return coef_df


def profile_misclassifications(X, y_true, y_pred, feature_names=None):
    """
    Profile false negatives and false positives.

    Returns
    -------
    fn_profile : DataFrame summary of false negatives
    fp_profile : DataFrame summary of false positives
    """
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = X.copy()

    df = df.copy()
    df["y_true"] = y_true.values if hasattr(y_true, "values") else y_true
    df["y_pred"] = y_pred

    fn_mask = (df["y_true"] == 1) & (df["y_pred"] == 0)
    fp_mask = (df["y_true"] == 0) & (df["y_pred"] == 1)

    fn_profile = df[fn_mask].describe()
    fp_profile = df[fp_mask].describe()

    return fn_profile, fp_profile, df[fn_mask], df[fp_mask]
