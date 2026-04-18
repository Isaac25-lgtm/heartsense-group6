import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import streamlit as st
from app.components.sidebar import render_sidebar

st.set_page_config(
    page_title="HeartSense",
    page_icon="\u2764\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

# --- Header ---
st.title("HeartSense")
st.subheader("Heart Disease Risk Prediction for Early Clinical Decision Support")

st.markdown(
    """
    **Uganda Christian University**
    | Faculty of Engineering Design & Technology
    | Department of Computing & Technology

    **Course:** CSC8204: Artificial Intelligence and Machine Learning
    | **Programme:** MSc Data Science and Analytics
    | **Group 6** | Easter 2026

    | Student | Registration No. |
    |---------|-----------------|
    | Nixon Kamugisha | B00321 |
    | Musoke Emmanuel | B31367 |
    | Mwesigwa Simon Peter Godwin | B31335 |
    | Omoding Isaac | B31331 |
    """
)

st.markdown("---")

# --- Opening ---
st.markdown(
    "We set out to answer a simple question: can routine clinical data help identify "
    "patients at higher risk of heart disease early enough to support better screening "
    "and referral decisions in Uganda?"
)

# --- Key stats ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset Records", "918")
col2.metric("Clinical Features", "11")
col3.metric("Source Institutions", "5")
col4.metric("ML Models Compared", "3")
st.caption(
    "These summary cards orient the audience before the technical pages begin. They show the scale of the dataset, "
    "how many predictors were available, where the records came from, and how many candidate algorithms were compared."
)

st.markdown("---")

# --- Problem ---
st.markdown(
    """
    ### The Problem

    Cardiovascular diseases cause an estimated **17.9 million deaths annually** worldwide.
    In Uganda, non-communicable diseases accounted for approximately **36% of all deaths** in 2019,
    with cardiovascular conditions representing a significant and growing proportion.

    At the primary healthcare level, clinicians at **Health Centre III and IV** facilities assess
    heart disease risk through manual evaluation of routine indicators. This process is
    inconsistent, time-consuming, and unable to detect complex nonlinear interactions among
    multiple risk factors.

    ### Our Approach

    This project develops and evaluates machine learning models for predicting heart disease risk
    using routine clinical indicators, and implements the best-performing model as a prototype
    clinical decision support tool.

    | Stage | Description |
    |-------|-------------|
    | **Data** | Combined Heart Disease Dataset (5 cardiology institutions, 918 records) |
    | **Models** | Logistic Regression, Random Forest, XGBoost |
    | **Evaluation** | Recall-prioritised screening metrics with locked evaluation sequence |
    | **Prototype** | Dual-mode prediction tool (full model + routine-care model) |
    | **Explainability** | SHAP values for global and per-patient explanations |

    ### Two Prediction Modes

    - **Full Model** (11 clinical inputs). This is a proof-of-concept using all available features.
    - **Routine-Care Model** (7 inputs). It uses only features available without exercise
      stress testing, suitable for lower-level health facilities in Uganda.

    Navigate using the sidebar to explore data insights, model performance,
    the clinical prediction tool, and model explainability.
    """
)

st.markdown(
    """
**How to read the app:** The pages are arranged in the same order as the project workflow. We begin with the
dataset and early patterns, then justify the final model, then demonstrate live prediction, and finally explain the
model's behaviour, errors, and limitations.
"""
)
