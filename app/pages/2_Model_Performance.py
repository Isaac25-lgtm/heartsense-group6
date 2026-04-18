import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
selection_evidence = pd.DataFrame(metadata.get("selection_evidence", []))
curves_data = load_json(APP_ASSETS_DIR, "test_curves.json")

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
rf_cv = cv_df.loc[cv_df["model"] == "Random Forest"].iloc[0]
display_cols = ["model", "recall_mean", "recall_std", "precision_mean", "f1_mean",
                "roc_auc_mean", "average_precision_mean"]
cv_display = cv_df[display_cols].copy()
cv_display.columns = ["Model", "Recall", "Recall Std", "Precision", "F1", "ROC-AUC", "PR-AUC"]
st.dataframe(cv_display.style.format({
    "Recall": "{:.4f}", "Recall Std": "{:.4f}", "Precision": "{:.4f}",
    "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"
}).highlight_max(subset=["Recall", "F1", "ROC-AUC", "PR-AUC"], color="#C8E6C9"),
    width="stretch", hide_index=True)
st.caption("Highlighted cells mark the strongest value in each comparison column.")
st.markdown(
    f"""
**What this table is trying to show:** It compares the three candidate models on repeated training folds so that
the final model choice is based on stable evidence rather than one lucky split.

**What our results show:** Random Forest was the strongest overall candidate on training-based comparison. It had
the highest recall (**{rf_cv['recall_mean']:.4f}**), highest F1-score (**{rf_cv['f1_mean']:.4f}**),
highest ROC-AUC (**{rf_cv['roc_auc_mean']:.4f}**), and highest PR-AUC (**{rf_cv['average_precision_mean']:.4f}**).
That is why it moved forward as the leading model before the hold-out test set was used.
"""
)

fig_path = FIGURES_DIR / "cv_comparison.png"
if fig_path.exists():
    st.markdown(
        """
**What this graph is trying to show:** The chart turns the cross-validation table into a visual comparison so it is
easier to see which model stays strongest across the main evaluation metrics.
"""
    )
    st.image(str(fig_path), width="stretch")
    st.markdown(
        """
**How to interpret it:** Bars or points that are consistently higher indicate a model with a stronger overall
balance. In our case, Random Forest stays strongest across the key metrics, which supports the decision to select it.
"""
    )

st.markdown("---")

# --- Section B: Test Set Results ---
st.subheader("Test Set Results (Hold-Out, Evaluated Once)")
st.markdown(
    "**Workflow Step 7: Evaluation and Result Interpretation.** "
    "After selecting the best model, we evaluated all three on a completely separate test set "
    "that was never used during training or tuning. These are the final results."
)

test_df = pd.read_csv(TABLES_DIR / "test_metrics.csv")
rf_test = test_df.loc[test_df["model"] == "Random Forest"].iloc[0]
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
st.caption("For Brier score, lower is better because it means the predicted probabilities are closer to reality.")
st.markdown(
    f"""
**What this table is trying to show:** This is the final, most trustworthy check of model performance because the
test set was kept separate until all modelling decisions were locked.

**What our results show:** Random Forest remained strong on unseen data. It achieved recall
**{rf_test['recall']:.4f}**, precision **{rf_test['precision']:.4f}**, F1-score **{rf_test['f1']:.4f}**,
ROC-AUC **{rf_test['roc_auc']:.4f}**, PR-AUC **{rf_test['pr_auc']:.4f}**, specificity
**{rf_test['specificity']:.4f}**, and the best Brier score **{rf_test['brier_score']:.4f}**. That means it not
only caught most disease cases, but also produced the most reliable probability estimates.
"""
)

st.markdown("---")

# --- Section C: Visual Evaluation ---
st.subheader("Visual Evaluation")
st.markdown(
    "**Workflow Step 6: Overfitting and Generalization Checks.** "
    "These graphs answer different questions about model quality: what kinds of errors the models make, how well "
    "they separate disease from normal cases, how trustworthy their probability scores are, and whether they show "
    "signs of overfitting."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Confusion Matrices", "ROC Curves", "PR Curves", "Calibration", "Learning Curves"
])

with tab1:
    st.markdown(
        """
**What this graph is trying to show:** A confusion matrix breaks predictions into four outcomes:
correctly identified disease (**true positive**), correctly identified healthy (**true negative**),
healthy patients flagged as at-risk (**false positive**), and disease patients missed (**false negative**).

In a screening project, false negatives matter most because they are the disease cases the model fails to catch.
False positives are also important, but they mainly increase referral burden.
"""
    )
    fig_path = FIGURES_DIR / "confusion_matrices.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")
    st.markdown(
        f"""
**What our graph shows:** For the selected Random Forest model on the test set, the confusion matrix corresponds to
**{int(rf_test['tp'])} true positives**, **{int(rf_test['tn'])} true negatives**, **{int(rf_test['fp'])} false
positives**, and **{int(rf_test['fn'])} false negatives**. In simple terms, the model caught most disease cases
while keeping false alarms fairly low.
"""
    )

with tab2:
    st.markdown(
        """
**What this graph is trying to show:** The ROC curve compares sensitivity against the false positive rate across all
possible thresholds. A curve closer to the top-left corner is better because it means the model catches more disease
cases while keeping false alarms lower.

**How to read it:** AUC of **1.0** would mean perfect separation, while **0.5** would mean the model is no better
than random guessing.
"""
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"lr": "#1976D2", "rf": "#4CAF50", "xgb": "#FF9800"}
    names = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
    for key in ["lr", "rf", "xgb"]:
        ax.plot(
            curves_data[key]["roc_fpr"],
            curves_data[key]["roc_tpr"],
            label=f"{names[key]} (AUC={curves_data[key]['roc_auc']:.3f})",
            color=colors[key],
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend()
    st.pyplot(fig)
    plt.close()
    st.markdown(
        f"""
**What our graph shows:** Random Forest has the highest ROC-AUC (**{curves_data['rf']['roc_auc']:.4f}**), which
means it was the strongest overall at ranking disease cases above normal cases across thresholds. Its curve stays
closest to the top-left region, which is why it was the best discriminator of the three models.
"""
    )

with tab3:
    st.markdown(
        """
**What this graph is trying to show:** Precision asks, *of the patients flagged as disease, how many truly had
disease?* Recall asks, *of all the true disease patients, how many did we catch?* The PR curve shows the tradeoff
between those two goals across thresholds.

This graph is especially important in our project because this is a screening-support problem, so recall was the lead
metric and false negatives were more concerning than extra referrals.
"""
    )
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
    st.markdown(
        f"""
**What our graph shows:** Random Forest has the highest PR-AUC / Average Precision
(**{curves_data['rf']['pr_auc']:.4f}**), which means it achieved the strongest balance between catching disease
cases and keeping false alarms under control. That is why it was especially attractive for a screening context.
"""
    )

with tab4:
    st.markdown(
        """
**What this graph is trying to show:** Calibration asks whether the model's probability estimates are trustworthy.
If the model says a patient has a 75% chance of disease, then roughly 75 out of 100 similar patients should actually
have disease.

**How to read it:** A calibration line close to the diagonal is better because it means the model's confidence scores
match reality more closely.
"""
    )
    fig_path = FIGURES_DIR / "calibration_curves.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")
    briers = metadata.get("brier_scores", {})
    if briers:
        st.markdown(
            f"""
**What our graph shows:** Random Forest had the lowest Brier score (**{briers['rf']:.4f}**) compared with
Logistic Regression (**{briers['lr']:.4f}**) and XGBoost (**{briers['xgb']:.4f}**). That means its probability
estimates were the most reliable, which matters because the app reports both a class label and a risk score.
"""
        )

with tab5:
    st.markdown(
        """
**What this graph is trying to show:** Learning curves show how performance changes as the model is trained on more
data. They compare the training score with the validation score.

**How to read it:** A large persistent gap between training and validation suggests overfitting. If both curves stay
low, that suggests underfitting. If they move closer together at a good level, that supports generalization.
"""
    )
    fig_path = FIGURES_DIR / "learning_curves.png"
    if fig_path.exists():
        st.image(str(fig_path), width="stretch")
    st.markdown(
        """
**What our graph shows:** The selected model does not show the large and persistent train-validation gap we would
expect from severe overfitting. That supports the conclusion that Random Forest was learning useful general patterns
rather than simply memorizing the training set.
"""
    )

st.markdown("---")

# --- Section D: Threshold Analysis ---
st.subheader("Threshold Analysis")
st.markdown(
    "Every prediction comes with a probability score. We need to choose a cutoff: "
    "above this number, we flag the patient. A lower cutoff catches more disease cases "
    "but also creates more false alarms. We tuned this cutoff on training data only, "
    f"targeting at least 90% recall. The selected threshold is **{best_threshold}**."
)
st.markdown(
    """
**What this graph is trying to show:** It shows how recall, precision, and related performance measures change as the
classification threshold moves up or down.

**Why it matters here:** In a screening project, using the default threshold of 0.5 is not always the best choice.
We wanted a threshold that would still catch at least 90% of true disease cases on the training-based tuning step.
"""
)

fig_path = FIGURES_DIR / "threshold_analysis.png"
if fig_path.exists():
    st.image(str(fig_path), width="stretch")
if not selection_evidence.empty:
    rf_selection = selection_evidence.loc[selection_evidence["Model"] == "Random Forest"].iloc[0]
    st.markdown(
        f"""
**What our graph shows:** At the selected Random Forest threshold of **{rf_selection['Threshold']:.2f}**, the
training-based tuning results achieved recall **{rf_selection['Recall@Thresh']:.4f}**, precision
**{rf_selection['Precision@Thresh']:.4f}**, and F1-score **{rf_selection['F1@Thresh']:.4f}**. This met the recall
target while avoiding an unnecessarily low cutoff that would have produced many more false alarms.
"""
    )

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
st.caption("Delta is calculated as Routine-Care Model minus Full Model for each metric.")

recall_row = comp_df.loc[comp_df["Metric"] == "Recall"].iloc[0]
precision_row = comp_df.loc[comp_df["Metric"] == "Precision"].iloc[0]
specificity_row = comp_df.loc[comp_df["Metric"] == "Specificity"].iloc[0]
brier_row = comp_df.loc[comp_df["Metric"] == "Brier Score"].iloc[0]

st.markdown(
    f"""
**What this table is trying to show:** It tests whether the project could still support screening in lower-resource
settings where exercise stress test features are unavailable.

**What our results show:** When the four stress-test-dependent variables were removed, recall rose slightly from
**{recall_row['Full Model (11 inputs)']:.4f}** to **{recall_row['Routine-Care Model (7 inputs)']:.4f}**, but
precision fell from **{precision_row['Full Model (11 inputs)']:.4f}** to
**{precision_row['Routine-Care Model (7 inputs)']:.4f}**, specificity fell from
**{specificity_row['Full Model (11 inputs)']:.4f}** to
**{specificity_row['Routine-Care Model (7 inputs)']:.4f}**, and the Brier score worsened from
**{brier_row['Full Model (11 inputs)']:.4f}** to **{brier_row['Routine-Care Model (7 inputs)']:.4f}**. In simple
terms, the routine-care model catches slightly more cases, but at the cost of more false-positive referrals and less
reliable risk scores.
"""
)

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
if not selection_evidence.empty:
    st.markdown(
        "**Selection evidence at the tuned thresholds:** these are the training-based numbers used to compare the "
        "final candidates before the hold-out test set was touched."
    )
    st.dataframe(
        selection_evidence.style.format({
            "Threshold": "{:.2f}",
            "Recall@Thresh": "{:.4f}",
            "Precision@Thresh": "{:.4f}",
            "F1@Thresh": "{:.4f}",
            "Brier": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )

st.info(f"""
{metadata['selection_rationale']}

This decision was made using cross-validation metrics, threshold analysis, and calibration
results from the training set only. The test set was evaluated once, after all decisions
were locked.
""")
