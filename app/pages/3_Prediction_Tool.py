import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

from app.components.sidebar import render_sidebar
from src.artifacts import (
    load_best_model,
    load_model_metadata,
    load_reduced_model,
    load_threshold,
)
from src.data_loader import replace_zero_with_nan

st.set_page_config(page_title="Prediction Tool | HeartSense", page_icon="\u2764\ufe0f", layout="wide")
render_sidebar()

st.title("Clinical Prediction Tool")
st.caption("Workflow Step 8: Deployment Prototype")

st.markdown(
    "Enter a patient's clinical measurements below and the model will estimate their "
    "heart disease risk. You can switch between the full model (which uses all 11 clinical "
    "features including exercise test results) and the routine-care model (which uses only "
    "the 7 features available without specialised equipment)."
)
st.markdown(
    """
**What this page is trying to show:** This is the deployment prototype of the project. It turns the trained model
into a usable screening tool that can accept live patient measurements and return an interpretable risk estimate.
"""
)


@st.cache_resource
def load_models():
    best = load_best_model()
    reduced = load_reduced_model()
    best_thresh = load_threshold("full")
    reduced_thresh = load_threshold("reduced")
    meta = load_model_metadata()
    return best, reduced, best_thresh, reduced_thresh, meta


best_model, reduced_model, best_threshold, reduced_threshold, metadata = load_models()

st.markdown("---")
mode = st.radio(
    "Select Prediction Mode:",
    ["Full Model (11 clinical inputs)", "Routine-Care Model (7 inputs)"],
    horizontal=True,
    help="The routine-care model uses only features available without exercise stress testing.",
)

is_routine = "Routine" in mode

if is_routine:
    st.info(
        "**Routine-Care Mode:** Uses only indicators available during standard patient "
        "encounters at lower-level health facilities. Stress-test features "
        "(MaxHR, ExerciseAngina, Oldpeak, ST_Slope) are not required."
    )
else:
    st.info(
        "**Full Model Mode:** Uses all 11 predictors, including exercise stress test features. This is the strongest "
        "version of the model because it has access to the richest clinical information."
    )

st.markdown("---")
st.subheader("Patient Clinical Data")
st.markdown(
    """
**How to read this section:** The inputs are grouped into demographics, cardiovascular indicators, cardiac
assessment, and exercise test results so that the form follows the same clinical structure used in the dataset.
"""
)

if st.button("Load Sample Patient"):
    st.session_state["sample_loaded"] = True

sample = st.session_state.get("sample_loaded", False)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographics**")
    age = st.slider("Age (years)", 20, 90, 55 if sample else 50)
    st.caption("(Older age is one of the strongest risk factors for heart disease)")
    sex = st.selectbox("Sex", ["M", "F"], index=0)
    st.caption("(Males tend to have higher heart disease rates in this dataset)")

    st.markdown("**Cardiovascular Indicators**")
    resting_bp = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 140 if sample else 120)
    st.caption("(Blood pressure measured at rest. Values above 130 mmHg indicate hypertension)")
    cholesterol = st.slider("Serum Cholesterol (mg/dl)", 0, 600, 250 if sample else 200)
    st.caption("(A waxy substance in the blood. High levels increase heart disease risk. Enter 0 if not measured)")
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?",
        [0, 1],
        index=1 if sample else 0,
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    st.caption("(Elevated fasting blood sugar can indicate diabetes, a heart disease risk factor)")

with col2:
    st.markdown("**Cardiac Assessment**")
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"],
        index=0,
        format_func=lambda x: {
            "ASY": "ASY (Asymptomatic)",
            "ATA": "ATA (Atypical Angina)",
            "NAP": "NAP (Non-Anginal Pain)",
            "TA": "TA (Typical Angina)",
        }[x],
    )
    st.caption("(Type of chest pain reported. Asymptomatic is the most common type in heart disease patients)")
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        index=0,
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "ST-T Abnormality",
            "LVH": "Left Ventricular Hypertrophy",
        }[x],
    )
    st.caption("(The heart's electrical activity at rest. Abnormalities may signal existing heart problems)")

    if not is_routine:
        st.markdown("**Exercise Test Results**")
        max_hr = st.slider("Max Heart Rate Achieved", 60, 202, 150 if sample else 150)
        st.caption("(Highest heart rate during a stress test. Lower values are associated with higher risk)")
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina?",
            ["N", "Y"],
            index=1 if sample else 0,
            format_func=lambda x: "Yes" if x == "Y" else "No",
        )
        st.caption("(Chest pain triggered by exercise, a sign the heart may not be getting enough blood)")
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.5 if sample else 0.0, step=0.1)
        st.caption("(Drop in the ECG ST segment during exercise. Higher values suggest reduced blood flow)")
        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"],
            index=1 if sample else 0,
            format_func=lambda x: {
                "Up": "Upsloping (normal)",
                "Flat": "Flat (intermediate)",
                "Down": "Downsloping (worst)",
            }[x],
        )
        st.caption("(Direction of the ST segment during exercise. Flat or downsloping is more concerning)")

st.markdown("---")

if st.button("Predict Risk", type="primary", use_container_width=True):
    input_data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
    }

    if not is_routine:
        input_data.update(
            {
                "MaxHR": max_hr,
                "ExerciseAngina": exercise_angina,
                "Oldpeak": oldpeak,
                "ST_Slope": st_slope,
            }
        )

    input_df = pd.DataFrame([input_data])
    input_df = replace_zero_with_nan(input_df)

    if is_routine:
        model = reduced_model
        threshold = reduced_threshold
        input_df = input_df[
            ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG"]
        ]
        model_label = "Routine-Care"
    else:
        model = best_model
        threshold = best_threshold
        input_df = input_df[
            [
                "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
                "Oldpeak", "ST_Slope",
            ]
        ]
        model_label = "Full"

    prob = float(model.predict_proba(input_df)[0, 1])
    prediction = int(prob >= threshold)

    if prob < 0.30:
        risk_cat = "LOW"
        status_renderer = st.success
    elif prob < 0.60:
        risk_cat = "MODERATE"
        status_renderer = st.warning
    else:
        risk_cat = "HIGH"
        status_renderer = st.error

    st.markdown("---")
    st.subheader("Prediction Result")

    result_col, prob_col, risk_col = st.columns(3)
    with result_col:
        if prediction == 1:
            st.error("**Heart Disease: PRESENT**")
        else:
            st.success("**Heart Disease: ABSENT**")

    with prob_col:
        st.metric("Probability Score", f"{prob:.1%}")

    with risk_col:
        status_renderer(f"**Risk Category: {risk_cat}**")

    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(
        f"The progress bar visualizes the predicted probability. The decision rule is simple: because the selected "
        f"{model_label.lower()} threshold is **{threshold:.2f}**, any probability at or above that cutoff is flagged "
        "as disease."
    )
    st.markdown(
        f"""
**What this result is trying to show:** The app combines the trained model, the selected threshold, and the
patient's entered values to produce a screening decision.

**What this result means:** This patient received a predicted probability of **{prob:.1%}**. Because that value is
{"above" if prediction == 1 else "below"} the selected threshold of **{threshold:.2f}**, the tool classified the
patient as **{"Heart Disease: PRESENT" if prediction == 1 else "Heart Disease: ABSENT"}**.
"""
    )

    # Clinical interpretation (safe, proof-of-concept language)
    if risk_cat == "HIGH":
        st.markdown(
            "This patient shows several risk factors that the model associates with heart disease. "
            "Further clinical assessment may be warranted."
        )
    elif risk_cat == "MODERATE":
        st.markdown(
            "This patient shows some risk indicators. "
            "Clinical judgement should guide whether further assessment is needed."
        )
    else:
        st.markdown(
            "This patient does not show strong risk indicators in this model. "
            "Routine monitoring and clinical judgement remain important."
        )

    # Priority features based on global importance
    st.markdown("**Priority features to review in this mode:**")
    try:
        shap_imp = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "tables", "shap_importance.csv")
        )
        available_features = set(input_df.columns)
        clean_features = (
            shap_imp["feature"]
            .str.replace("numeric__", "", regex=False)
            .str.replace("nominal__", "", regex=False)
            .str.replace("ordinal__", "", regex=False)
            .str.replace("passthrough__", "", regex=False)
        )
        top_features = []
        for clean_name in clean_features:
            if clean_name in available_features and clean_name not in top_features:
                top_features.append(clean_name)
            if len(top_features) == 5:
                break

        for feat in top_features:
            st.markdown(f"- **{feat}** (high global predictive contribution)")

        st.caption(
            "These are the most important features in this prediction mode based on the "
            "model's global explainability results. They are not a live patient-specific "
            "SHAP explanation."
        )
    except Exception:
        st.caption("Feature importance data not available.")

    st.markdown(
        """
**How to interpret this section safely:** The probability score and risk category are meant to support screening and
referral decisions, not to replace clinical judgement. The listed priority features are global model drivers, not a
live patient-specific explanation.
"""
    )
    st.caption(
        f"Model: {model_label} ({metadata['selected_model_name']})  |  "
        f"Threshold: {threshold:.2f}  |  "
        f"Based on analysis of 918 clinical records from 5 cardiology institutions"
    )

    st.warning(
        "This tool is a screening aid for research and demonstration purposes only. "
        "It does not replace clinical diagnosis. Refer to a cardiologist for definitive assessment."
    )
