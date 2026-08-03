"""Train, tune, evaluate, and serialize house-price regression models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.eda import run_eda
from src.preprocess import build_preprocessor, engineer_features
from src.utils import PROJECT_ROOT, RANDOM_STATE, configure_logging, find_data_file, infer_target_column, save_json

LOGGER = logging.getLogger(__name__)


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return regression metrics in original target units."""
    return {"RMSE": float(mean_squared_error(y_true, y_pred) ** .5), "MAE": float(mean_absolute_error(y_true, y_pred)), "R2": float(r2_score(y_true, y_pred))}


def train_project(data_path: str | Path | None = None) -> pd.DataFrame:
    """Run the complete reproducible training workflow and save artifacts."""
    root = PROJECT_ROOT
    models_dir, reports_dir, outputs_dir = root / "models", root / "reports", root / "outputs"
    for directory in (models_dir, reports_dir, outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(find_data_file(data_path))
    target = infer_target_column(data)
    if data[target].isna().any():
        LOGGER.warning("Dropping %d rows with missing target", data[target].isna().sum())
        data = data.dropna(subset=[target])
    run_eda(data, target, reports_dir, outputs_dir)
    data = engineer_features(data, target)
    features, y = data.drop(columns=[target]), pd.to_numeric(data[target], errors="raise")
    x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=.2, random_state=RANDOM_STATE)
    log_target = bool(y_train.skew() > 1 and (y_train > -1).all())
    train_target = np.log1p(y_train) if log_target else y_train
    bundle = build_preprocessor(x_train)
    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    holdout_metrics: dict[str, dict[str, float]] = {}
    holdout_predictions: dict[str, np.ndarray] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessing", bundle.transformer), ("model", estimator)])
        result = cross_validate(
            pipeline,
            x_train,
            train_target,
            cv=cv,
            n_jobs=-1,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
        )
        pipeline.fit(x_train, train_target)
        candidate_predictions = pipeline.predict(x_test)
        if log_target:
            candidate_predictions = np.expm1(candidate_predictions)
        holdout = _scores(y_test.to_numpy(), candidate_predictions)
        fitted_models[name] = pipeline
        holdout_metrics[name] = holdout
        holdout_predictions[name] = candidate_predictions
        rows.append({
            "Model": name,
            "CV_RMSE_TrainingScale": -result["test_rmse"].mean(),
            "CV_MAE_TrainingScale": -result["test_mae"].mean(),
            "CV_R2_TrainingScale": result["test_r2"].mean(),
            "Holdout_RMSE": holdout["RMSE"],
            "Holdout_MAE": holdout["MAE"],
            "Holdout_R2": holdout["R2"],
        })
    rf_pipeline = Pipeline([("preprocessing", bundle.transformer), ("model", candidates["RandomForest"])])
    search = GridSearchCV(rf_pipeline, {
        "model__n_estimators": [200, 400], "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 2],
    }, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, refit=True)
    search.fit(x_train, train_target)
    best_model = search.best_estimator_
    predicted = best_model.predict(x_test)
    if log_target:
        predicted = np.expm1(predicted)
    metrics = _scores(y_test.to_numpy(), predicted)
    fitted_models["RandomForest (tuned)"] = best_model
    holdout_metrics["RandomForest (tuned)"] = metrics
    holdout_predictions["RandomForest (tuned)"] = predicted
    rows.append({
        "Model": "RandomForest (tuned)",
        "CV_RMSE_TrainingScale": -search.best_score_,
        "CV_MAE_TrainingScale": np.nan,
        "CV_R2_TrainingScale": np.nan,
        "Holdout_RMSE": metrics["RMSE"],
        "Holdout_MAE": metrics["MAE"],
        "Holdout_R2": metrics["R2"],
    })
    comparison = pd.DataFrame(rows).sort_values("Holdout_RMSE")
    selected_name = str(comparison.iloc[0]["Model"])
    selected_model = fitted_models[selected_name]
    selected_metrics = holdout_metrics[selected_name]
    selected_predictions = holdout_predictions[selected_name]
    selected_estimator = selected_model.named_steps["model"]
    metrics_payload = {
        "Model": selected_estimator.__class__.__name__,
        **{key: round(value, 4) for key, value in selected_metrics.items()},
        "BestParameters": search.best_params_ if selected_name == "RandomForest (tuned)" else {},
        "TargetLogTransformed": log_target,
    }
    save_json(metrics_payload, outputs_dir / "metrics.json")
    joblib.dump(selected_model, models_dir / "model.pkl")
    joblib.dump(selected_model.named_steps["preprocessing"], models_dir / "preprocessing.pkl")
    save_json({"target": target, "features": features.columns.tolist()}, outputs_dir / "feature_columns.json")
    comparison.to_csv(outputs_dir / "model_comparison.csv", index=False)
    pd.DataFrame({"actual": y_test, "predicted": selected_predictions, "residual": y_test - selected_predictions}).to_csv(outputs_dir / "predictions.csv", index=False)
    from src.evaluate import create_evaluation_plots
    create_evaluation_plots(selected_model, x_test, y_test, selected_predictions, reports_dir)
    LOGGER.info("Training complete. Selected %s with holdout RMSE: %.4f", selected_name, selected_metrics["RMSE"])
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the house price prediction project.")
    parser.add_argument("--data-path", help="Optional Housing.csv path")
    arguments = parser.parse_args()
    configure_logging()
    train_project(arguments.data_path)
