"""
Data Loading
============
Load raw dataset and apply zero-as-missing replacement.
"""

import pandas as pd
import numpy as np
from src.config import DATASET_PATH, ZERO_AS_MISSING, TARGET, ALL_FEATURES


def load_raw_data():
    """Load the raw heart disease dataset from CSV."""
    df = pd.read_csv(DATASET_PATH)
    return df


def replace_zero_with_nan(df):
    """
    Replace clinically implausible zero values with NaN.
    Cholesterol=0 and RestingBP=0 are treated as missing data.
    """
    df = df.copy()
    for col in ZERO_AS_MISSING:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df


def load_data_for_modeling():
    """Load data with zeros replaced by NaN, ready for pipeline imputation."""
    df = load_raw_data()
    df = replace_zero_with_nan(df)
    return df
