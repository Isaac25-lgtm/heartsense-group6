# HeartSense

Heart Disease Risk Prediction for Early Clinical Decision Support Using Machine Learning.

## About

HeartSense is a machine learning-based screening support tool that predicts heart disease risk from routine clinical indicators. It was built as a proof-of-concept prototype for primary healthcare settings in Uganda and East Africa, where access to specialist cardiac diagnostics is limited.

The system features a dual-mode prediction tool: a full model using all 11 clinical features (including exercise stress test results), and a routine-care model using only the 7 features available during a standard patient visit at Health Centre III/IV facilities.

**Live Demo:** [heartsense-group6.streamlit.app](https://heartsense-group6.streamlit.app)

## Team

**Group 6** | Uganda Christian University | Faculty of Engineering Design & Technology

| Student | Registration No. | Programme |
|---------|-----------------|-----------|
| Nixon Kamugisha | B00321 | MSc Data Science and Analytics |
| Musoke Emmanuel | B31367 | MSc Data Science and Analytics |
| Mwesigwa Simon Peter Godwin | B31335 | MSc Data Science and Analytics |
| Omoding Isaac | B31331 | MSc Computer Science |

**Course:** CSC8204 - Artificial Intelligence and Machine Learning | Easter 2026

## Dataset

The [Combined Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by fedesoriano, published under CC BY 4.0. It merges clinical records from five cardiology institutions (Cleveland, Hungarian, Swiss, Long Beach VA, Statlog) into 918 unique patient records with 11 clinical features and 1 binary target variable.

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient age in years |
| Sex | Categorical | M (Male) / F (Female) |
| ChestPainType | Categorical | ASY, ATA, NAP, or TA |
| RestingBP | Numeric | Resting blood pressure (mmHg) |
| Cholesterol | Numeric | Serum cholesterol (mg/dl) |
| FastingBS | Binary | 1 if fasting blood sugar > 120 mg/dl |
| RestingECG | Categorical | Normal / ST / LVH |
| MaxHR | Numeric | Maximum heart rate achieved |
| ExerciseAngina | Binary | Y (Yes) / N (No) |
| Oldpeak | Numeric | ST depression induced by exercise |
| ST_Slope | Ordinal | Up / Flat / Down |
| HeartDisease | Target | 1 = disease present, 0 = normal |

## Models and Results

Three algorithms were trained and compared using stratified 5-fold cross-validation with recall as the primary metric (prioritising the detection of true disease cases in a screening context).

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC | Brier Score |
|-------|--------|-----------|-----|---------|--------|-------------|
| Logistic Regression | 0.8922 | 0.8349 | 0.8626 | 0.8918 | 0.8805 | 0.1281 |
| **Random Forest** | **0.8824** | **0.9000** | **0.8911** | **0.9317** | **0.9334** | **0.1043** |
| XGBoost | 0.8529 | 0.8788 | 0.8657 | 0.9099 | 0.9193 | 0.1327 |

**Selected model:** Random Forest (threshold 0.49). It tied with XGBoost for highest recall at the tuned threshold and was chosen for stronger precision, F1-score, and calibration (lowest Brier score). Model selection was locked before the test set was evaluated.

### Full Model vs Routine-Care Model

| Metric | Full (11 inputs) | Routine-Care (7 inputs) | Delta |
|--------|-----------------|------------------------|-------|
| Recall | 0.8824 | 0.9020 | +0.0196 |
| Precision | 0.9000 | 0.7931 | -0.1069 |
| F1 | 0.8911 | 0.8440 | -0.0471 |
| ROC-AUC | 0.9317 | 0.8797 | -0.0520 |

The routine-care model maintains strong recall even without exercise stress test features, demonstrating feasibility for settings where that equipment is unavailable.

## App Pages

| Page | Workflow Step | What It Shows |
|------|-------------|---------------|
| Home | Problem Definition | Project overview, team, problem context |
| Data Insights | Dataset Description + EDA | Interactive feature explorer, data quality audit, correlations, medical glossary |
| Model Performance | Model Selection + Evaluation | CV comparison, test metrics, confusion matrices, ROC/PR/calibration curves, threshold analysis, full vs reduced comparison |
| Prediction Tool | Deployment Prototype | Dual-mode clinical prediction with risk classification, probability score, and feature context |
| Explainability | Documentation + Limitations | SHAP analysis, feature importance, LR coefficients, error analysis, limitations, future work |

## Project Structure

```
heartsense-group6/
├── app/
│   ├── Home.py                         # Streamlit entry point
│   ├── pages/
│   │   ├── 1_Data_Insights.py
│   │   ├── 2_Model_Performance.py
│   │   ├── 3_Prediction_Tool.py
│   │   └── 4_Explainability.py
│   └── components/
│       └── sidebar.py                  # Shared sidebar branding
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_training.ipynb               # Preprocessing + Model Training
│   ├── 03_evaluation.ipynb             # Threshold Tuning + Test Evaluation
│   └── 04_explainability.ipynb         # SHAP + Error Analysis
├── src/
│   ├── config.py                       # Paths, seeds, feature definitions
│   ├── data_loader.py                  # Dataset loading and cleaning
│   ├── preprocessing.py                # sklearn Pipeline construction
│   ├── feature_engineering.py          # Derived clinical features
│   ├── modeling.py                     # Model definitions and grids
│   ├── evaluation.py                   # Metrics and threshold analysis
│   ├── explainability.py               # SHAP and error profiling
│   └── artifacts.py                    # Save/load functions for all artifacts
├── models/                             # Saved pipelines and metadata
├── data/raw/heart.csv                  # Dataset
├── outputs/                            # Figures, tables, SHAP artifacts
└── requirements.txt
```

## Local Setup

1. Clone the repository:
   ```
   git clone https://github.com/Isaac25-lgtm/heartsense-group6.git
   cd heartsense-group6
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS/Linux
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the app:
   ```
   streamlit run app/Home.py
   ```

5. Open `http://localhost:8501` in your browser and use the sidebar to navigate.

## Key Technical Decisions

- **Recall over accuracy.** In a screening context, missing a true disease case (false negative) is more costly than a false alarm. Recall was the primary evaluation metric.
- **Locked evaluation sequence.** Threshold tuning and model selection were completed on training data only. The test set was evaluated once, after all decisions were locked. This prevents data leakage.
- **Pipeline-based preprocessing.** All transformations (imputation, encoding, scaling) are encapsulated in sklearn Pipeline objects fitted on training data only, ensuring identical preprocessing between training and prediction.
- **ST_Slope ordinal encoding.** Down < Flat < Up follows standard cardiology interpretation, where downsloping ST segments indicate worse prognosis.
- **Always-rebuild model loading.** The app rebuilds the preprocessor from scratch at load time using the saved training data, then reattaches the trained estimator. This guarantees compatibility across sklearn versions.

## Limitations

- The dataset originates from international cardiology centres and may not represent Ugandan patient populations.
- Four features require exercise stress testing not available at lower-level health facilities in Uganda.
- The sample size (918 records) limits generalisability of tuned hyperparameters.
- This is a proof-of-concept screening aid, not a validated clinical diagnostic tool.
- Local validation with Ugandan patient data is required before any real-world use.

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (fedesoriano, Kaggle)
