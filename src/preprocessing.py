"""
Preprocessing Pipelines
=======================
Builds sklearn Pipeline and ColumnTransformer objects.
Shared by notebooks and the Streamlit app to ensure identical preprocessing.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.config import (
    NUMERIC_FEATURES, NOMINAL_FEATURES, ORDINAL_FEATURES,
    PASSTHROUGH_FEATURES, ST_SLOPE_ORDER,
    NUMERIC_ROUTINE, NOMINAL_ROUTINE, ORDINAL_ROUTINE,
)


def build_preprocessor(scale_numeric=True, feature_set="full"):
    """
    Build a ColumnTransformer for preprocessing.

    Parameters
    ----------
    scale_numeric : bool
        If True, apply StandardScaler after imputation (for Logistic Regression).
        If False, skip scaling (for tree-based models).
    feature_set : str
        'full' uses all 11 features. 'routine' uses only 7 non-stress-test features.

    Returns
    -------
    preprocessor : ColumnTransformer
    """
    if feature_set == "routine":
        num_cols = NUMERIC_ROUTINE
        nom_cols = NOMINAL_ROUTINE
        ord_cols = ORDINAL_ROUTINE
    else:
        num_cols = NUMERIC_FEATURES
        nom_cols = NOMINAL_FEATURES
        ord_cols = ORDINAL_FEATURES

    # Numeric pipeline: impute then optionally scale
    if scale_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    # Nominal categorical: impute then one-hot encode
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [
        ("numeric", numeric_pipeline, num_cols),
        ("nominal", nominal_pipeline, nom_cols),
    ]

    # Ordinal encoding for ST_Slope (only in full feature set)
    if ord_cols:
        ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=[ST_SLOPE_ORDER], handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("ordinal", ordinal_pipeline, ord_cols))

    # FastingBS: passthrough (already binary 0/1)
    transformers.append(("passthrough", "passthrough", PASSTHROUGH_FEATURES))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_feature_names(preprocessor, feature_set="full"):
    """
    Get output feature names after fitting the preprocessor.
    Must be called after preprocessor.fit().
    """
    return list(preprocessor.get_feature_names_out())
