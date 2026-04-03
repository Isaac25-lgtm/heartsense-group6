import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from src.artifacts import load_json, load_model_metadata
from src.config import TABLES_DIR, APP_ASSETS_DIR, FIGURES_DIR
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Model Performance | HeartSense", page_icon="\u2764\ufe0f", layout="wide")
render_sidebar()

st.title("Model Performance")
st.caption("Workflow Steps 5, 6, and 7: Model Selection, Overfitting Checks, and Evaluation")

metadata = load_model_metadata()
best_model_name = metadata["selected_model_name"]
best_threshold = metadata["selected_threshold"]

st.success(
    f"**Selected Model:** {best_model_name}  |  "
    f"**Threshold:** {best_threshold}  |  "
    f"**Selection:** Locked before test set evaluation"
)

st.markdown("---")

# --- Section A: Cross-Validation Comparison ---
st.subheader("Cross-Validation Comparison (Training Data)")
st.markdown(
    "**Workflow Step 5: Model Selection and Experimentation.** "
    "We trained three algorithms and tested each using 5-fold cross-validation on the training data. "
    "The table shows the average performance across all five rounds."
)

cv_df = pd.read_csv(TABLES_DIR / "cv_results.csv")
display_cols = ["model", "recall_mean", "recall_std", "precision_mean", "f1_mean",
                "roc_auc_mean", "average_precision_mean"]
cv_display = cv_df[display_cols].copy()
cv_display.columns = ["Model", "Recall", "Recall Std", "Precision", "F1", "ROC-AUC", "PR-AUC"]
st.dataframe(cv_display.style.format({
    "Recall": "{:.4f}", "Recall Std": "{:.4f}", "Precision": "{:.4f}",
    "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"
}).highlight_max(subset=["Recall", "F1", "ROC-AUC", "PR-AUC"], color="#C8E6C9"),
    width="stretch", hide_index=True)

fig_path = FIGURES_DIR / "cv_comparison.png"
if fig_path.exists():
    st.image(str(fig_path), width="stretch")

st.markdown("---")

# --- Section B: Test Set Results ---
st.subheader("Test Set Results (Hold-Out, Evaluated Once)")
st.markdown(
    "**Workflow Step 7: Evaluation and Result Interpretation.** "
    "After selecting the best model, we evaluated all three on a completely separate test set "
    "that was never used during training or tuning. These are the final results."
)

test_df = pd.read_csv(TABLES_DIR / "test_metrics.csv")
display_test = test_df[["model", "threshold", "recall", "precision", "f1",
                         "roc_auc", "pr_auc", "brier_score", "specificity"]].copy()
display_test.columns = ["Model", "Threshold", "Recall", "Precision", "F1",
                         "ROC-AUC", "PR-AUC", "Brier", "Specificity"]
st.dataframe(display_test.style.format({
    "Threshold": "{:.2f}", "Recall": "{:.4f}", "Precision": "{:.4f}", "F1": "{:.4f}",
    "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "Brier": "{:.4f}", "Specificity": "{:.4f}"
}).highlight_max(subset=["Recall", "F1", "ROC-AUC", "PR-AUC"], color="#C8E6C9")
 .highlight_min(subset=["Brier"], color="#C8E6C9"),
    width="stretch", hide_index=True)

st.markdown("---")

# --- Section C: Visual Evaluation ---
st.subheader("Visual Evaluation")
st.markdown("**Workflow Step 6: Overfitting and Generalization Checks.**")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Confusion Matrices", "ROC Curves", "PR Curves", "Calibration", "Learning Curves"
])

with tab1:
    st.markdown(
        "A confusion matrix breaks down predictions into four outcomes: correctly identified "
        "disease (true positive), correctly identified healthy (true negative), healthy patients "
        "flagged as at-risk (false positive), and disease patients missed (false negative)."
    )
    fig_path = FIGURES_DIR / "confusion_matrices.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")

with tab2:
    st.markdown(
        "The ROC curve shows the tradeoff between catching disease cases (sensitivity) "
        "and incorrectly flagging healthy patients. A curve closer to the top-left corner "
        "is better. AUC of 1.0 is perfect, 0.5 is random guessing."
    )
    fig_path = FIGURES_DIR / "roc_pr_curves.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")

with tab3:
    st.markdown(
        "Precision asks: of the patients flagged, how many actually had disease? "
        "Recall asks: of all disease patients, how many did we catch? "
        "This curve shows the tradeoff between the two."
    )
    curves_data = load_json(APP_ASSETS_DIR, "test_curves.json")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"lr": "#1976D2", "rf": "#4CAF50", "xgb": "#FF9800"}
    names = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
    for key in ["lr", "rf", "xgb"]:
        ax.plot(curves_data[key]["pr_recall"], curves_data[key]["pr_precision"],
                label=f"{names[key]} (AP={curves_data[key]['pr_auc']:.3f})",
                color=colors[key], linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend()
    st.pyplot(fig)
    plt.close()

with tab4:
    st.markdown(
        "When the model says a patient has a 75% chance of heart disease, roughly 75 out of "
        "100 such patients should actually have it. A line close to the diagonal means the "
        "model's confidence scores match reality."
    )
    fig_path = FIGURES_DIR / "calibration_curves.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")

with tab5:
    st.markdown(
        "Learning curves show how performance changes with more training data. "
        "A large gap between training and validation scores indicates overfitting. "
        "Both scores being low indicates underfitting."
    )
    fig_path = FIGURES_DIR / "learning_curves.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")

st.markdown("---")

# --- Section D: Threshold Analysis ---
st.subheader("Threshold Analysis")
st.markdown(
    "Every prediction comes with a probability score. We need to choose a cutoff: "
    "above this number, we flag the patient. A lower cutoff catches more disease cases "
    "but also creates more false alarms. We tuned this cutoff on training data only, "
    f"targeting at least 90% recall. The selected threshold is **{best_threshold}**."
)

fig_path = FIGURES_DIR / "threshold_analysis.png"
if fig_path.exists():
    st.image(str(fig_path), width="stretch")

st.markdown("---")

# --- Section E: Full vs Reduced Feature Experiment ---
st.subheader("Full Model vs Routine-Care Model")
st.markdown(
    "In many Ugandan health facilities, exercise stress testing equipment is not available. "
    "We tested what happens when we remove those four features and train with only the "
    "seven measurements a health worker can collect during a routine visit."
)

comp_df = pd.read_csv(TABLES_DIR / "full_vs_reduced_comparison.csv")
st.dataframe(comp_df.style.format({
    col: "{:.4f}" for col in comp_df.columns if col != "Metric"
}), width="stretch", hide_index=True)

st.markdown(
    """
    The routine-care model maintains strong recall even without stress-test features.
    Precision and specificity decrease, meaning more false-positive referrals.
    This is an acceptable tradeoff for a screening tool: over-referring is safer than missing cases.
    Both models require local validation with Ugandan patient data before real-world use.
    """
)

st.markdown("---")

# --- Model Selection Rationale ---
st.subheader("Model Selection Rationale")

st.info(f"""
{metadata['selection_rationale']}

This decision was made using cross-validation metrics, threshold analysis, and calibration
results from the training set only. The test set was evaluated once, after all decisions
were locked.
""")
