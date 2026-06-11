# Heart Disease Risk Prediction

**Health Analytics | Machine Learning Evaluation | Responsible AI | Reproducible Portfolio Project**

This repository turns a classic heart disease dataset into a more production-ready analytical asset: a reproducible ML pipeline, a static executive dashboard, documented data-quality checks, and model evaluation outputs that go beyond a single accuracy score.

The goal is not to build a diagnostic product. The goal is to show strong analytical engineering, clearer ML validation, and responsible interpretation in a healthcare context.

## Live dashboard

[Open the analytical dashboard](https://luandarodrigues.github.io/heart-disease-risk-prediction/)

## What this project shows

- Reproducible pipeline outputs instead of one-off notebook results
- Structured preprocessing with categorical encoding and numeric scaling
- Side-by-side evaluation of Logistic Regression and Random Forest
- Calibration, confusion-matrix, and threshold analysis
- Explicit data-quality checks for suspicious clinical values
- Clinical interpretation layer that separates model signal from real-world completeness

## Dataset

The project uses a structured heart disease dataset with:

- `918` records
- `12` columns
- `11` predictive variables
- Binary target: `HeartDisease`

Available variables include age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise angina, Oldpeak, and ST slope.

## Current model snapshot

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.897 | 0.930 | 0.888 | 0.931 | 0.909 |
| Random Forest | 0.897 | 0.932 | 0.881 | 0.941 | 0.910 |

## Why this version is stronger

The repository now behaves more like a small analytical application than a loose ML exercise.

- The pipeline is packaged under `src/`
- The script entry point is preserved for easy execution
- Tests validate the output contract used by the dashboard
- The generated artifacts include:
  - `model_metrics.csv`
  - `feature_importance_permutation.csv`
  - `data_quality_summary.csv`
  - `rate_by_ChestPainType.csv`
  - `rate_by_ExerciseAngina.csv`
  - `rate_by_ST_Slope.csv`
  - `calibration_by_model.csv`
  - `confusion_matrix_summary.csv`
  - `threshold_metrics.csv`
  - `model_run_metadata.json`

## Repository structure

```text
heart-disease-risk-prediction/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- data/
|-- docs/
|   `-- index.html
|-- scripts/
|   |-- _bootstrap.py
|   `-- heart_disease_model.py
|-- src/
|   `-- heart_disease_risk_prediction/
|       |-- __init__.py
|       |-- cli.py
|       `-- pipeline.py
`-- tests/
    `-- test_pipeline_outputs.py
```

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python scripts/heart_disease_model.py data/heart.csv data
```

You can also use the package entry point:

```bash
heart-disease-pipeline data/heart.csv data
```

### 3. Run tests

```bash
pytest
```

## Pipeline steps

1. Load the source dataset
2. Split train and test with stratification
3. Build a preprocessing pipeline for numeric and categorical variables
4. Train Logistic Regression and Random Forest models
5. Score both models on holdout data
6. Export performance, calibration, threshold, and confusion outputs
7. Export dashboard-ready descriptive summaries

## Data-quality checks

The project explicitly flags issues that matter in a healthcare dataset:

- Missing cells
- Duplicate rows
- `RestingBP = 0`
- `Cholesterol = 0`
- `MaxHR = 0`

This matters because apparent model performance can look strong even when source values contain implausible or coded placeholders.

## Interpretability layer

Permutation importance indicates the strongest signals are concentrated in:

- `ST_Slope`
- `ChestPainType`
- `Sex`
- `ExerciseAngina`
- `Cholesterol`
- `Oldpeak`

The dashboard translates those technical variables into clinical meaning so the project reads like analytical work, not only code output.

## Responsible-use position

This repository should not be interpreted as a deployable clinical model.

Key limitations:

- No external validation
- No subgroup fairness analysis
- No prospective monitoring
- No formal calibration governance in a clinical environment
- Important modern cardiovascular risk factors are missing from the dataset

That separation is intentional: strong portfolio work should show both what the model can do and where it stops being trustworthy.

## Stack

Python, Pandas, scikit-learn, pytest, HTML, CSS, JavaScript, GitHub Pages, machine learning evaluation, health analytics, and responsible AI framing.
