"""
Artifact Save/Load
===================
Centralized functions for saving and loading all project artifacts.
Used by notebooks (save) and Streamlit app (load).
"""

import json
import joblib
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, OUTPUTS_DIR, TABLES_DIR, FIGURES_DIR, SHAP_DIR, APP_ASSETS_DIR, DATASET_PATH
from src.preprocessing import build_preprocessor

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # pragma: no cover
    InconsistentVersionWarning = UserWarning


# ----------------------------------------------
# SAVE FUNCTIONS (used by notebooks)
# ----------------------------------------------

def save_model(pipeline, filename):
    """Save a fitted pipeline to models/ directory."""
    path = MODELS_DIR / filename
    joblib.dump(pipeline, path)
    print(f"Saved model: {path}")


def save_json(data, directory, filename):
    """Save a dict/list to JSON."""
    path = Path(directory) / filename

    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)
    print(f"Saved JSON: {path}")


def save_dataframe(df, directory, filename):
    """Save a DataFrame to CSV."""
    path = Path(directory) / filename
    df.to_csv(path, index=False)
    print(f"Saved CSV: {path}")


def save_pickle(obj, directory, filename):
    """Save an object with joblib."""
    path = Path(directory) / filename
    joblib.dump(obj, path)
    print(f"Saved pickle: {path}")


def _has_broken_imputer_state(pipeline):
    """
    Detect sklearn version-mismatch breakage in deserialized SimpleImputer objects.

    Pipelines saved under sklearn 1.7.x and loaded under 1.8.x may lose fitted
    SimpleImputer state (for example, missing `statistics_` / `_fill_dtype`), which
    breaks transform/predict_proba at runtime.
    """
    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor")
    if preprocessor is None:
        return False

    for _, transformer, _ in preprocessor.transformers:
        if not hasattr(transformer, "steps"):
            continue
        for _, step in transformer.steps:
            if step.__class__.__name__ == "SimpleImputer" and not hasattr(step, "statistics_"):
                return True
    return False


def _repair_loaded_pipeline(pipeline, filename):
    """
    Rebuild and fit only the preprocessor using the saved train split, then attach
    the already-fitted model step.

    This keeps the estimator weights/trees intact while restoring a compatible
    preprocessor under the current sklearn version.
    """
    repair_config = {
        "best_model.joblib": {"scale_numeric": False, "feature_set": "full"},
        "rf_pipeline.joblib": {"scale_numeric": False, "feature_set": "full"},
        "xgb_pipeline.joblib": {"scale_numeric": False, "feature_set": "full"},
        "lr_pipeline.joblib": {"scale_numeric": True, "feature_set": "full"},
        "reduced_model.joblib": {"scale_numeric": False, "feature_set": "routine"},
    }

    config = repair_config.get(filename)
    if config is None:
        return pipeline

    split_path = MODELS_DIR / "train_test_split.joblib"
    split_data = joblib.load(split_path)
    X_train = split_data["X_train"]

    preprocessor = build_preprocessor(
        scale_numeric=config["scale_numeric"],
        feature_set=config["feature_set"],
    )
    preprocessor.fit(X_train)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", pipeline.named_steps["model"]),
    ])


# ----------------------------------------------
# LOAD FUNCTIONS (used by Streamlit app)
# ----------------------------------------------

def load_model(filename):
    """
    Load a saved pipeline from models/ directory.

    Always rebuilds the preprocessor from scratch using the saved training
    split, then reattaches the already-fitted estimator. This guarantees
    compatibility regardless of the sklearn version used to save the artifact.
    """
    path = MODELS_DIR / filename
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            pipeline = joblib.load(path)
        # Always rebuild — sklearn version-mismatch can silently corrupt
        # SimpleImputer fitted state even when hasattr checks pass.
        return _repair_loaded_pipeline(pipeline, filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found: {path}. Run the training notebook first.")


def load_best_model():
    """Load the final selected full-feature model."""
    return load_model("best_model.joblib")


def load_reduced_model():
    """Load the reduced-feature (routine-care) model."""
    return load_model("reduced_model.joblib")


def load_json(directory, filename):
    """Load a JSON file and return dict/list."""
    path = Path(directory) / filename
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON not found: {path}. Run the relevant notebook first.")


def load_threshold(model_type="full"):
    """Load the tuned threshold. model_type: 'full' or 'reduced'."""
    filename = "best_threshold.json" if model_type == "full" else "reduced_threshold.json"
    data = load_json(MODELS_DIR, filename)
    return data["threshold"]


def load_metrics():
    """Load the evaluation metrics table."""
    path = TABLES_DIR / "test_metrics.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metrics not found: {path}. Run evaluation notebook first.")


def load_cv_results():
    """Load cross-validation results."""
    path = TABLES_DIR / "cv_results.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CV results not found: {path}. Run training notebook first.")


def load_shap_values():
    """Load pre-computed SHAP values."""
    path = SHAP_DIR / "shap_values.pkl"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"SHAP values not found: {path}. Run explainability notebook first.")


def load_shap_explainer():
    """Load the SHAP explainer."""
    path = SHAP_DIR / "shap_explainer.pkl"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"SHAP explainer not found: {path}. Run explainability notebook first.")


def load_dataset():
    """Load the raw dataset."""
    return pd.read_csv(DATASET_PATH)


def load_eda_assets():
    """Load pre-computed EDA summary data."""
    return load_json(APP_ASSETS_DIR, "eda_summary.json")


def load_feature_columns(model_type="full"):
    """Load feature column names for the given model type."""
    data = load_json(MODELS_DIR, "feature_columns.json")
    if model_type == "routine":
        return data.get("reduced_pipeline_columns", data.get("full_pipeline_columns"))
    return data.get("full_pipeline_columns", data.get("full"))


def load_model_metadata():
    """Load model selection metadata."""
    return load_json(MODELS_DIR, "model_metadata.json")
