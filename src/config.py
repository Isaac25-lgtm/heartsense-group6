"""
HeartSense Configuration
========================
Central configuration for paths, random seeds, constants, and feature definitions.
All notebooks and the Streamlit app import from here to ensure consistency.
"""

from pathlib import Path

# ----------------------------------------------
# PATHS
# ----------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
SHAP_DIR = OUTPUTS_DIR / "shap"
APP_ASSETS_DIR = OUTPUTS_DIR / "app_assets"

DATASET_PATH = DATA_RAW / "heart.csv"

# ----------------------------------------------
# REPRODUCIBILITY
# ----------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# ----------------------------------------------
# TARGET VARIABLE
# ----------------------------------------------
TARGET = "HeartDisease"

# ----------------------------------------------
# FEATURE DEFINITIONS
# ----------------------------------------------

# All 11 input features
ALL_FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak",
    "ST_Slope"
]

# Features requiring exercise stress testing (not available at HC III/IV)
STRESS_TEST_FEATURES = ["MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]

# Routine-care features (available without stress testing)
ROUTINE_CARE_FEATURES = [f for f in ALL_FEATURES if f not in STRESS_TEST_FEATURES]

# Numeric features
NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
NUMERIC_ROUTINE = [f for f in NUMERIC_FEATURES if f not in STRESS_TEST_FEATURES]

# Categorical features -- nominal (one-hot encoded)
NOMINAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina"]
NOMINAL_ROUTINE = [f for f in NOMINAL_FEATURES if f not in STRESS_TEST_FEATURES]

# Categorical features -- ordinal
ORDINAL_FEATURES = ["ST_Slope"]
ORDINAL_ROUTINE = [f for f in ORDINAL_FEATURES if f not in STRESS_TEST_FEATURES]

# Binary passthrough
PASSTHROUGH_FEATURES = ["FastingBS"]

# ST_Slope ordinal ordering (clinically justified: Down is worst prognosis)
ST_SLOPE_ORDER = ["Down", "Flat", "Up"]

# ----------------------------------------------
# FEATURE ENGINEERING THRESHOLDS
# ----------------------------------------------

# WHO blood pressure categories
BP_NORMAL_UPPER = 120
BP_ELEVATED_UPPER = 129
# >= 130 is Hypertensive

# High cholesterol threshold (standard clinical cutoff)
HIGH_CHOLESTEROL_THRESHOLD = 240

# ----------------------------------------------
# ZERO-VALUE HANDLING
# ----------------------------------------------
# These features have clinically implausible zero values that represent missing data
ZERO_AS_MISSING = ["Cholesterol", "RestingBP"]

# ----------------------------------------------
# MODEL NAMES
# ----------------------------------------------
MODEL_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost"
}

# ----------------------------------------------
# RISK CATEGORY DEFAULTS (will be refined after threshold tuning)
# ----------------------------------------------
RISK_THRESHOLDS = {
    "low_upper": 0.30,
    "moderate_upper": 0.60,
    # >= moderate_upper is High risk
}

# ----------------------------------------------
# STREAMLIT
# ----------------------------------------------
APP_NAME = "HeartSense"
APP_TAGLINE = "Heart Disease Risk Prediction for Early Clinical Decision Support"
APP_VERSION = "1.0.0"
