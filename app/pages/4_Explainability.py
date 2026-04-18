import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd

from src.config import FIGURES_DIR, TABLES_DIR, APP_ASSETS_DIR
from src.artifacts import load_model_metadata, load_json
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Explainability | HeartSense", page_icon="\u2764\ufe0f", layout="wide")
render_sidebar()

st.title("Model Explainability and Limitations")
st.caption("Workflow Step 9: Documentation, Reflection, and Limitations")

metadata = load_model_metadata()
shap_df = pd.read_csv(TABLES_DIR / "shap_importance.csv") if (TABLES_DIR / "shap_importance.csv").exists() else None

st.markdown("---")

# --- Section A: Global Feature Importance ---
st.subheader("Global Feature Importance (SHAP)")
st.markdown(
    "SHAP (SHapley Additive exPlanations) measures how much each feature pushes "
    "a prediction toward disease or toward normal. Features at the top matter most."
)
if shap_df is not None:
    top_features = (
        shap_df["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("nominal__", "", regex=False)
        .str.replace("ordinal__", "", regex=False)
        .str.replace("passthrough__", "", regex=False)
        .head(5)
        .tolist()
    )
    st.markdown(
        f"""
**What this section is trying to show:** It explains which variables the selected model relied on most across the
test set.

**What our results show:** The strongest global contributors were **{', '.join(top_features)}**. That is important
because these were also among the strongest signals seen earlier in the EDA stage, so the explainability results are
consistent with the broader data story.
"""
    )

tab1, tab2 = st.tabs(["SHAP Summary (Beeswarm)", "SHAP Bar (Mean |SHAP|)"])

with tab1:
    st.markdown(
        "Each dot is one patient from the test set. The horizontal position shows whether "
        "that feature pushed the prediction toward disease (right) or away from it (left). "
        "Red means the feature value was high for that patient, blue means it was low."
    )
    fig_path = FIGURES_DIR / "shap_summary.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")
    st.markdown(
        """
**How to interpret it:** Features near the top matter most overall. A wide horizontal spread means the feature can
strongly influence predictions. Red points clustering on the right suggest higher values tend to push predictions
toward disease, while blue points on the right suggest lower values are associated with higher risk.
"""
    )

with tab2:
    st.markdown(
        "This bar chart ranks features by their average impact on predictions across "
        "all patients in the test set."
    )
    fig_path = FIGURES_DIR / "shap_bar.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")
    st.markdown(
        """
**How to interpret it:** This plot simplifies the SHAP story into a ranking. The longer the bar, the greater the
average predictive contribution of that feature across the test set.
"""
    )

# Feature importance comparison across models
fig_path = FIGURES_DIR / "feature_importance_comparison.png"
if fig_path.exists():
    st.markdown("**Feature Importance Across All Three Models**")
    st.markdown(
        "Different algorithms weigh features differently. This comparison shows the top "
        "features according to SHAP values, Random Forest Gini importance, and XGBoost "
        "gain importance. Features appearing near the top across all three are the most "
        "reliably important."
    )
    st.image(str(fig_path), width="stretch")
    st.caption(
        "These rankings show predictive contribution within the models. "
        "They do not imply that these features cause heart disease."
    )
    st.markdown(
        """
**What this comparison is trying to show:** Different models measure feature importance differently. When the same
features stay near the top across SHAP, Random Forest importance, and XGBoost gain, that strengthens confidence that
those predictors genuinely mattered to model behaviour.
"""
    )

st.markdown("---")

# --- Section B: Individual Patient Explanations ---
st.subheader("Individual Patient Explanations")
st.markdown(
    "These are three specific patients from the test set. For each one, the waterfall "
    "chart shows which features pushed the prediction up (toward disease) or down "
    "(toward normal), and by how much."
)

try:
    sample_patients = load_json(APP_ASSETS_DIR, "sample_patients.json")

    selected_patient = st.selectbox(
        "Select a pre-computed sample patient:",
        list(sample_patients.keys())
    )

    patient_data = sample_patients[selected_patient]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Probability:** {patient_data['probability']:.4f}")
        st.markdown(f"**True Label:** {'Disease' if patient_data['true_label'] == 1 else 'Normal'}")
        st.markdown(f"**Predicted:** {'Disease' if patient_data['predicted'] == 1 else 'Normal'}")

    with col2:
        st.markdown("**Patient Features:**")
        features_df = pd.DataFrame([patient_data["features"]])
        st.dataframe(features_df, hide_index=True)

    safe_label = selected_patient.replace(" ", "_").replace("(", "").replace(")", "")
    waterfall_path = FIGURES_DIR / f"shap_waterfall_{safe_label}.png"
    if waterfall_path.exists():
        st.image(str(waterfall_path), width="stretch")
    else:
        st.caption("Waterfall plot not available for this patient.")

    case_type = selected_patient.lower()
    if "true positive" in case_type:
        interpretation = (
            "This example shows a correctly flagged disease case. The waterfall plot helps explain why the model was "
            "confident and which features pushed the prediction strongly upward."
        )
    elif "false negative" in case_type:
        interpretation = (
            "This example shows a missed disease case. It is useful because it reveals where the model can still fail "
            "and reminds us why clinical judgement remains essential."
        )
    else:
        interpretation = (
            "This example sits closer to the decision boundary. It helps show how competing feature effects can push a "
            "prediction only slightly toward one class or the other."
        )
    st.markdown(f"**How to interpret this case:** {interpretation}")

except Exception as e:
    st.caption(f"Sample patient data not available: {e}")

st.markdown("---")

# --- Section C: Logistic Regression Coefficients ---
st.subheader("Logistic Regression Coefficients")
st.markdown(
    "Logistic Regression is the most transparent of our three models. Each feature gets a "
    "coefficient: positive means it increases predicted risk, negative means it decreases "
    "risk. Larger bars indicate stronger effects. These are calculated after feature scaling."
)

lr_coef_path = TABLES_DIR / "lr_coefficients.csv"
if lr_coef_path.exists():
    lr_df = pd.read_csv(lr_coef_path)
    st.dataframe(lr_df.style.format({
        "coefficient": "{:.4f}", "abs_coefficient": "{:.4f}"
    }), width="stretch", hide_index=True)

    fig_path = FIGURES_DIR / "lr_coefficients.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")

    st.caption(
        "Coefficients reflect association within the model after scaling, "
        "not causal clinical relationships."
    )
    if not lr_df.empty:
        strongest = lr_df.iloc[0]
        st.markdown(
            f"""
**What this section is trying to show:** Logistic Regression provides the most direct coefficient-based explanation
of feature influence.

**What our results show:** The strongest coefficient by absolute size is **{strongest['feature']}**
({strongest['coefficient']:.4f}), which the linear model associates with **{strongest['direction'].lower()}**.
This helps us compare a transparent baseline model with the more complex tree-based models.
"""
        )

st.markdown("---")

# --- Section D: Error Analysis ---
st.subheader("Error Analysis")
st.markdown(
    "No model is perfect. Here we look at the patients our model got wrong. "
    "Understanding these errors helps us know where clinical judgement remains essential."
)

try:
    error_summary = load_json(APP_ASSETS_DIR, "error_summary.json")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Set Size", error_summary["n_test"])
    col2.metric("False Negatives (Missed)", error_summary["n_fn"])
    col3.metric("False Positives (Over-referred)", error_summary["n_fp"])

    fig_path = FIGURES_DIR / "error_analysis.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")

    st.markdown(
        """
        False negatives are the most concerning errors in a screening context because
        they represent disease patients the model missed entirely. False positives
        increase referral burden on hospitals but do not directly harm patients.
        The recall-focused threshold aims to minimise missed cases at the cost of
        accepting more false alarms.
        """
    )
    st.markdown(
        f"""
**What this section is trying to show:** It focuses on where the model still fails, not just where it succeeds.

**What our results show:** On the 184-patient test set, the model produced **{error_summary['n_fn']} false
negatives** and **{error_summary['n_fp']} false positives** at the selected threshold of
**{error_summary['threshold']:.2f}**. That means the model was tuned to miss fewer disease cases, even if that
required accepting some extra false alarms.
"""
    )

except Exception as e:
    st.caption(f"Error analysis data not available: {e}")

st.markdown("---")

# --- Section E: Limitations & Future Work ---
st.subheader("Limitations and Future Work")

st.markdown(
    """
    **Dataset Limitations**

    The dataset originates from five international cardiology institutions and may not
    represent Ugandan or East African patient populations in terms of demographics,
    genetics, or lifestyle-related risk factors. 172 records (18.7%) had missing cholesterol
    values (coded as zero), which were imputed using median values from the training data.

    **Feature Limitations**

    Four features (MaxHR, ExerciseAngina, Oldpeak, ST_Slope) require exercise stress testing,
    which is not routinely available at primary health facilities in Uganda.
    The routine-care model demonstrates what is achievable without these features.

    **Model Limitations**

    The models were trained on 918 records, which limits the generalisability of tuned
    hyperparameters. No external validation dataset was available to assess
    out-of-distribution performance. Feature engineering (BP_Category, HighCholesterol) was
    implemented and explored during data preparation, but the final deployed pipelines
    use the original clinical variables only.

    **Deployment Considerations**

    Local validation using Ugandan patient data is required before any real-world adoption.
    Integration with existing clinical workflows (e.g. DHIS2) would need further development.
    This prototype is framed as a screening and referral support tool, not a diagnostic instrument.

    **Future Improvements**

    Collecting and validating on locally sourced Ugandan clinical data would be the most
    important next step. Other directions include exploring deep learning approaches with
    larger datasets, incorporating additional routine-care features available in Ugandan
    health records, and developing a mobile-friendly deployment for offline use at
    Health Centre III/IV.
    """
)
st.markdown(
    """
**How to read this section:** These limitations are not weaknesses to hide. They are what keep the project honest.
They explain why the app should be presented as a proof of concept screening-support system rather than as a finished
clinical product.
"""
)
