# HeartSense

Heart Disease Risk Prediction for Early Clinical Decision Support Using Machine Learning.

**CSC8204: AI & ML | Group 6 | Uganda Christian University | Easter 2026**

## Setup

1. Create and activate virtual environment:
   ```
   C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Verify dataset exists at `data/raw/heart.csv` (918 records).

## Run the App

```
.venv\Scripts\python.exe -m streamlit run app/Home.py
```

## Project Structure

```
data/raw/          - Raw dataset
notebooks/         - Jupyter notebooks (EDA, Training, Evaluation, Explainability)
src/               - Shared Python modules (config, preprocessing, modeling, etc.)
models/            - Saved model pipelines and metadata
outputs/           - Figures, tables, SHAP artifacts, app assets
app/               - Streamlit multi-page application
```

## Dataset

Combined Heart Disease Dataset (fedesoriano, Kaggle) - CC BY 4.0
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
