import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.artifacts import load_dataset, load_eda_assets
from src.data_loader import replace_zero_with_nan
from src.config import TARGET, NUMERIC_FEATURES
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="Data Insights | HeartSense", page_icon="\u2764\ufe0f", layout="wide")
render_sidebar()

st.title("Data Insights")
st.caption("Workflow Steps 2 and 3: Dataset Description and Exploratory Data Analysis")

# Load data
df = load_dataset()
eda = load_eda_assets()

# --- Medical Glossary ---
with st.expander("Medical Terms Glossary (click to expand)"):
    st.markdown(
        """
        | Term | Meaning |
        |------|---------|
        | **Cholesterol (mg/dl)** | A waxy substance in the blood. High levels increase heart disease risk. |
        | **RestingBP (mmHg)** | Blood pressure measured while the patient is at rest. High values indicate hypertension. |
        | **MaxHR** | The highest heart rate reached during an exercise stress test. In this dataset, lower values were often associated with higher heart disease risk. |
        | **Oldpeak** | How much the ST segment on an ECG drops during exercise compared to rest. Higher values suggest reduced blood flow to the heart. |
        | **ST_Slope** | The direction of the ST segment on an ECG during exercise. Flat or downsloping patterns are more concerning than upsloping. |
        | **FastingBS** | Whether fasting blood sugar exceeds 120 mg/dl, which can indicate diabetes, a known risk factor for heart disease. |
        | **ExerciseAngina** | Chest pain triggered by physical activity, a classic sign that the heart muscle is under stress. |
        | **ChestPainType** | The type of chest pain reported. Asymptomatic (ASY) means no chest pain, which is paradoxically the most common type among heart disease patients in this dataset. |
        | **RestingECG** | The electrical activity of the heart measured at rest. ST-T abnormalities or LVH (left ventricular hypertrophy) may indicate existing heart problems. |
        | **ECG** | Electrocardiogram. A test that records the electrical signals of the heart. |
        """
    )

st.markdown("---")

# --- Section A: Dataset Summary ---
st.subheader("Dataset Summary")
st.markdown(
    "**Workflow Step 2: Dataset Acquisition and Description.** "
    "Our dataset combines clinical records from five international cardiology institutions "
    "into 918 unique patient records with 11 clinical features."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", eda["n_records"])
c2.metric("Clinical Features", eda["n_features"])
c3.metric("Heart Disease", f"{eda['target_counts']['disease']} ({eda['target_pct']['disease']}%)")
c4.metric("Normal", f"{eda['target_counts']['normal']} ({eda['target_pct']['normal']}%)")
st.caption(
    "These headline numbers establish the scale of the dataset and confirm that this is a binary classification "
    "problem with moderate class imbalance rather than a perfectly balanced dataset."
)

st.markdown(
    "This chart shows how many patients had heart disease versus those who did not. "
    "The class balance is fairly moderate, but still uneven enough to justify "
    "stratified splitting and recall-focused evaluation."
)

fig, ax = plt.subplots(figsize=(5, 3))
colors = ["#4CAF50", "#E53935"]
bars = ax.bar(["Normal (0)", "Disease (1)"],
              [eda["target_counts"]["normal"], eda["target_counts"]["disease"]],
              color=colors)
for bar, count in zip(bars, [eda["target_counts"]["normal"], eda["target_counts"]["disease"]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(count), ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Count")
ax.set_title("Target Class Distribution", fontweight="bold")
st.pyplot(fig)
plt.close()
st.markdown(
    f"""
**What this graph is trying to show:** It compares the number of normal patients with the number of disease
patients so we can see the class balance before modelling begins.

**What our graph shows:** The disease class is slightly larger (**{eda['target_counts']['disease']}** cases,
{eda['target_pct']['disease']}%) than the normal class (**{eda['target_counts']['normal']}** cases,
{eda['target_pct']['normal']}%). That imbalance is not severe, but it is large enough to justify stratified
splitting and recall-focused evaluation later in the project.
"""
)

st.markdown("---")

# --- Section B: Feature Explorer ---
st.subheader("Feature Explorer")
st.markdown(
    "**Workflow Step 3: Exploratory Data Analysis.** "
    "Select any feature below to see how it is distributed across patients with "
    "and without heart disease."
)

all_features = eda["numeric_features"] + eda["categorical_features"]
selected_feature = st.selectbox("Select a feature to explore:", all_features)

df_clean = replace_zero_with_nan(df)

if selected_feature in eda["numeric_features"]:
    st.markdown(
        "The histogram shows how this measurement is spread across both groups. "
        "The boxplot highlights the median and range for each group."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, color, name in [(0, "#4CAF50", "Normal"), (1, "#E53935", "Disease")]:
            subset = df_clean[df_clean[TARGET] == label][selected_feature].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=name, edgecolor="white")
        ax.set_title(f"{selected_feature} Distribution by Target", fontweight="bold")
        ax.set_xlabel(selected_feature)
        ax.legend()
        st.pyplot(fig)
        plt.close()
        normal_median = df_clean[df_clean[TARGET] == 0][selected_feature].median()
        disease_median = df_clean[df_clean[TARGET] == 1][selected_feature].median()
        direction = "higher" if disease_median > normal_median else "lower"
        st.markdown(
            f"""
**What this histogram is trying to show:** It compares the spread of **{selected_feature}** in normal and disease
patients so we can see whether the two groups cluster differently.

**What our graph shows:** The disease group tends to have a **{direction}** median value for
**{selected_feature}** than the normal group ({disease_median:.2f} vs {normal_median:.2f}). That means this feature
shows some separation between the classes and may contribute useful predictive information.
"""
        )

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_clean, x=TARGET, y=selected_feature,
                    palette={0: "#4CAF50", 1: "#E53935"}, hue=TARGET, legend=False, ax=ax)
        ax.set_xticklabels(["Normal", "Disease"])
        ax.set_title(f"{selected_feature} by Target", fontweight="bold")
        st.pyplot(fig)
        plt.close()
        st.markdown(
            """
**What this boxplot is trying to show:** It summarizes the median, spread, and extreme values for each class.

**How to interpret it:** A shift in the median line or overall box position suggests that the feature behaves
differently in disease and normal patients. Outliers are shown but not automatically treated as errors, because in a
clinical dataset extreme values may still be real and meaningful.
"""
        )

    st.markdown("**Summary Statistics by Class:**")
    st.dataframe(df_clean.groupby(TARGET)[selected_feature].describe().round(2))
    st.caption("These summary statistics provide the exact numerical version of the patterns seen in the histogram and boxplot.")

else:
    st.markdown(
        "This bar chart compares how often each category appears in patients with and "
        "without heart disease. Categories strongly associated with disease are useful for prediction."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ct = pd.crosstab(df[selected_feature], df[TARGET])
        ct.columns = ["Normal", "Disease"]
        ct.plot(kind="bar", ax=ax, color=["#4CAF50", "#E53935"], edgecolor="white", rot=0)
        ax.set_title(f"{selected_feature} by Target", fontweight="bold")
        ax.set_xlabel("")
        ax.legend(title="")
        st.pyplot(fig)
        plt.close()
        ct_pct = pd.crosstab(df[selected_feature], df[TARGET], normalize="index") * 100
        top_category = ct_pct[1].idxmax()
        top_rate = ct_pct.loc[top_category, 1]
        st.markdown(
            f"""
**What this bar chart is trying to show:** It compares how often each category appears in the normal and disease
groups.

**What our graph shows:** For **{selected_feature}**, the category with the highest disease rate is
**{top_category}**, where about **{top_rate:.1f}%** of patients fall in the disease class. That is why some
categories become strong predictors later in the modelling stage.
"""
        )

    with col2:
        st.markdown("**Disease Rate by Category:**")
        ct_pct.columns = ["Normal %", "Disease %"]
        st.dataframe(ct_pct.round(1))
        st.caption("This table converts raw counts into percentages, which makes the class association easier to interpret.")

st.markdown("---")

# --- Section C: Data Quality ---
st.subheader("Data Quality")
st.markdown(
    "**Workflow Step 4: Data Preprocessing.** "
    "Real world clinical data is never perfect. Before modelling, we identified "
    "and handled missing and implausible values."
)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    **Cholesterol Zero-Value Audit**
    - Records with Cholesterol = 0: **{eda['cholesterol_zeros']}** ({eda['cholesterol_zeros']/eda['n_records']*100:.1f}%)
    - These represent missing data from the Hungarian and Swiss subsets
    - Median cholesterol (excluding zeros): **{eda['cholesterol_median_clean']:.0f}** mg/dl
    - **Treatment:** Replaced with NaN, imputed via median within the sklearn pipeline
    """)

with col2:
    st.markdown(f"""
    **RestingBP Zero-Value Audit**
    - Records with RestingBP = 0: **{eda['restingbp_zeros']}**
    - Same treatment: NaN replacement + median imputation

    **Duplicates:** {eda['duplicates']}
    """)
st.markdown(
    """
**Why this section matters:** These checks show that EDA did not stop at drawing charts. It directly identified the
major data-quality problem in the project and motivated the preprocessing decisions used later in the pipeline.
"""
)

st.markdown("---")

# --- Section D: Correlation Heatmap ---
st.subheader("Correlation Analysis")
st.markdown(
    "This heatmap shows how strongly each pair of numeric features moves together. "
    "We check this so highly overlapping features do not distort interpretation, especially in Logistic Regression."
)

corr_cols = NUMERIC_FEATURES + [TARGET]
corr_matrix = df_clean[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title("Pearson Correlation Heatmap", fontweight="bold")
st.pyplot(fig)
plt.close()

target_corr = corr_matrix[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
top_corr_feature = target_corr.index[0]
top_corr_value = target_corr.iloc[0]
st.markdown(
    f"""
**What this heatmap is trying to show:** It checks how strongly the numeric features move together and how each one
relates to the target.

**What our graph shows:** The strongest numeric relationship with the target is **{top_corr_feature}**
({top_corr_value:.2f}). More broadly, the heatmap helps us see whether numeric predictors overlap too strongly, which
matters especially for interpretation and for models like Logistic Regression.
"""
)

st.markdown("---")

# --- Section E: Key Findings ---
st.subheader("Key Findings from EDA")

st.markdown(
    """
    The target split is roughly 55% disease and 45% normal. This moderate imbalance
    justifies using stratified data splitting and class weighting during training.

    The strongest separators between the two groups are ST_Slope (Flat pattern),
    ChestPainType (Asymptomatic), ExerciseAngina (Yes), MaxHR (lower in disease patients),
    and Oldpeak (higher in disease patients).

    Cholesterol had 172 records (18.7%) with zero values representing missing data from
    the Hungarian and Swiss source datasets. These were replaced with NaN and imputed
    using the training set median inside the sklearn pipeline.

    Four of the strongest predictors (MaxHR, ExerciseAngina, Oldpeak, ST_Slope) require
    exercise stress testing, which is not routinely available at Health Centre III/IV
    in Uganda. The reduced-feature experiment in the Model Performance section
    quantifies the impact of removing these features.
    """
)
