"""
Shared Utilities
================
Helper functions used across notebooks.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(X_numeric):
    """
    Compute Variance Inflation Factor for numeric features.

    Parameters
    ----------
    X_numeric : DataFrame with numeric columns only (no NaN)

    Returns
    -------
    vif_df : DataFrame with columns [feature, VIF]
    """
    X = X_numeric.dropna()
    vif_data = []
    for i, col in enumerate(X.columns):
        vif_val = variance_inflation_factor(X.values, i)
        vif_data.append({"feature": col, "VIF": round(vif_val, 2)})
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)


def detect_outliers_iqr(series, factor=1.5):
    """
    Detect outliers using the IQR method.

    Returns
    -------
    lower_bound, upper_bound, n_outliers, outlier_mask
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (series < lower) | (series > upper)
    return lower, upper, mask.sum(), mask
