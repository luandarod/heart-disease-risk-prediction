from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "HeartDisease"
RATE_EXPORT_COLUMNS = ["ChestPainType", "ExerciseAngina", "ST_Slope"]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def load_dataset(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include="number").columns.tolist()
    categorical_features = x.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
            ]
        ),
    }


def evaluate_model(
    name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, float | str], Pipeline, pd.Series]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index, name="probability")
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }
    return metrics, model, probabilities


def build_data_quality_summary(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rows": len(df),
                "columns": df.shape[1],
                "target_rate": y.mean(),
                "missing_cells": int(df.isna().sum().sum()),
                "duplicates": int(df.duplicated().sum()),
                "zero_RestingBP": int((df["RestingBP"] == 0).sum()),
                "zero_Cholesterol": int((df["Cholesterol"] == 0).sum()),
                "zero_MaxHR": int((df["MaxHR"] == 0).sum()),
            }
        ]
    )


def build_rate_by_group(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)[TARGET_COLUMN]
        .agg(records="size", heart_disease_rate="mean")
        .reset_index()
        .sort_values("heart_disease_rate", ascending=False)
    )
    grouped["heart_disease_rate_pct"] = grouped["heart_disease_rate"] * 100
    return grouped


def build_feature_importance(best_model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    permutation = permutation_importance(best_model, x_test, y_test, n_repeats=20, random_state=42)
    return (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": permutation.importances_mean,
                "importance_std": permutation.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def build_confusion_summary(
    fitted_models: dict[str, Pipeline], x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for name, model in fitted_models.items():
        tn, fp, fn, tp = confusion_matrix(y_test, model.predict(x_test)).ravel()
        rows.append(
            {
                "model": name,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )
    return pd.DataFrame(rows)


def build_calibration_summary(probabilities_by_model: dict[str, pd.Series], y_test: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for name, probabilities in probabilities_by_model.items():
        observed_rate, mean_predicted = calibration_curve(y_test, probabilities, n_bins=5, strategy="quantile")
        for index, (observed, predicted) in enumerate(zip(observed_rate, mean_predicted), start=1):
            rows.append(
                {
                    "model": name,
                    "bin": index,
                    "mean_predicted_probability": predicted,
                    "observed_rate": observed,
                }
            )
    return pd.DataFrame(rows)


def build_threshold_metrics(probabilities_by_model: dict[str, pd.Series], y_test: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for name, probabilities in probabilities_by_model.items():
        for threshold in THRESHOLDS:
            predictions = (probabilities >= threshold).astype(int)
            rows.append(
                {
                    "model": name,
                    "threshold": threshold,
                    "precision": precision_score(y_test, predictions, zero_division=0),
                    "recall": recall_score(y_test, predictions, zero_division=0),
                    "f1": f1_score(y_test, predictions, zero_division=0),
                    "positive_predictions": int(predictions.sum()),
                }
            )
    return pd.DataFrame(rows)


def run_pipeline(input_path: Path, output_dir: Path = Path("data")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    preprocessor = build_preprocessor(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = build_models(preprocessor)
    metrics_rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    probabilities_by_model: dict[str, pd.Series] = {}

    for name, model in models.items():
        metrics, fitted_model, probabilities = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        metrics_rows.append(metrics)
        fitted_models[name] = fitted_model
        probabilities_by_model[name] = probabilities

    pd.DataFrame(metrics_rows).to_csv(output_dir / "model_metrics.csv", index=False)
    build_feature_importance(fitted_models["Random Forest"], x_test, y_test).to_csv(
        output_dir / "feature_importance_permutation.csv", index=False
    )
    build_data_quality_summary(df, y).to_csv(output_dir / "data_quality_summary.csv", index=False)

    for column in RATE_EXPORT_COLUMNS:
        build_rate_by_group(df, column).to_csv(output_dir / f"rate_by_{column}.csv", index=False)

    build_confusion_summary(fitted_models, x_test, y_test).to_csv(
        output_dir / "confusion_matrix_summary.csv", index=False
    )
    build_calibration_summary(probabilities_by_model, y_test).to_csv(
        output_dir / "calibration_by_model.csv", index=False
    )
    build_threshold_metrics(probabilities_by_model, y_test).to_csv(
        output_dir / "threshold_metrics.csv", index=False
    )

    metadata = {
        "source_file": str(input_path),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "models": list(models.keys()),
        "rate_exports": [f"rate_by_{column}.csv" for column in RATE_EXPORT_COLUMNS],
        "new_outputs": [
            "calibration_by_model.csv",
            "confusion_matrix_summary.csv",
            "threshold_metrics.csv",
        ],
    }
    (output_dir / "model_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
