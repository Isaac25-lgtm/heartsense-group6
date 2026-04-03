"""
Feature Engineering
===================
Clinically grounded derived features, applied before pipeline encoding.
"""

import numpy as np
import pandas as pd
from src.config import BP_NORMAL_UPPER, BP_ELEVATED_UPPER, HIGH_CHOLESTEROL_THRESHOLD


def add_bp_category(df):
    """
    Categorise RestingBP per WHO thresholds into ordinal levels.
    Normal (0) < Elevated (1) < Hypertensive (2).
    NaN values in RestingBP produce NaN in BP_Category.
    """
    df = df.copy()
    conditions = [
        df["RestingBP"] < BP_NORMAL_UPPER,
        (df["RestingBP"] >= BP_NORMAL_UPPER) & (df["RestingBP"] <= BP_ELEVATED_UPPER),
        df["RestingBP"] > BP_ELEVATED_UPPER,
    ]
    choices = [0, 1, 2]
    df["BP_Category"] = np.select(conditions, choices, default=np.nan)
    # Preserve NaN where RestingBP is NaN
    df.loc[df["RestingBP"].isna(), "BP_Category"] = np.nan
    return df


def add_cholesterol_flag(df):
    """
    Binary indicator for high cholesterol (> 240 mg/dl).
    NaN values in Cholesterol produce NaN in HighCholesterol.
    """
    df = df.copy()
    df["HighCholesterol"] = (df["Cholesterol"] > HIGH_CHOLESTEROL_THRESHOLD).astype(float)
    df.loc[df["Cholesterol"].isna(), "HighCholesterol"] = np.nan
    return df


def apply_feature_engineering(df):
    """Apply all feature engineering steps."""
    df = add_bp_category(df)
    df = add_cholesterol_flag(df)
    return df
