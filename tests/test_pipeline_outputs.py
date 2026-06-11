from __future__ import annotations

from pathlib import Path

import pandas as pd

from heart_disease_risk_prediction.pipeline import run_pipeline


def test_run_pipeline_writes_dashboard_contract_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"

    run_pipeline(Path("data/heart.csv"), output_dir)

    expected = {
        "model_metrics.csv",
        "feature_importance_permutation.csv",
        "data_quality_summary.csv",
        "rate_by_ChestPainType.csv",
        "rate_by_ExerciseAngina.csv",
        "rate_by_ST_Slope.csv",
        "calibration_by_model.csv",
        "confusion_matrix_summary.csv",
        "threshold_metrics.csv",
        "model_run_metadata.json",
    }
    written = {path.name for path in output_dir.iterdir()}
    assert expected.issubset(written)


def test_model_metrics_contains_expected_models(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"

    run_pipeline(Path("data/heart.csv"), output_dir)
    metrics = pd.read_csv(output_dir / "model_metrics.csv")

    assert set(metrics["model"]) == {"Logistic Regression", "Random Forest"}
    assert {"accuracy", "roc_auc", "precision", "recall", "f1"}.issubset(metrics.columns)


def test_new_evaluation_outputs_have_expected_columns(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"

    run_pipeline(Path("data/heart.csv"), output_dir)

    calibration = pd.read_csv(output_dir / "calibration_by_model.csv")
    confusion = pd.read_csv(output_dir / "confusion_matrix_summary.csv")
    thresholds = pd.read_csv(output_dir / "threshold_metrics.csv")

    assert {"model", "bin", "mean_predicted_probability", "observed_rate"}.issubset(calibration.columns)
    assert {"model", "true_negative", "false_positive", "false_negative", "true_positive"}.issubset(confusion.columns)
    assert {"model", "threshold", "precision", "recall", "f1", "positive_predictions"}.issubset(thresholds.columns)
