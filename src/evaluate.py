"""Diagnostic visualizations for the fitted regression pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline


def create_evaluation_plots(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, predicted: np.ndarray, reports_dir: Path) -> None:
    """Save residual, prediction, error, and feature-importance diagnostics."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_test.to_numpy() - predicted
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=predicted, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted price")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(reports_dir / "residual_plot.png", dpi=160)
    plt.close()
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=predicted)
    limits = [min(y_test.min(), predicted.min()), max(y_test.max(), predicted.max())]
    plt.plot(limits, limits, "r--")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.tight_layout()
    plt.savefig(reports_dir / "actual_vs_predicted.png", dpi=160)
    plt.close()
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Prediction error")
    plt.tight_layout()
    plt.savefig(reports_dir / "error_distribution.png", dpi=160)
    plt.close()
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        names = model.named_steps["preprocessing"].get_feature_names_out()
        importance = pd.Series(estimator.feature_importances_, index=names).nlargest(20).sort_values()
    elif hasattr(estimator, "coef_"):
        names = model.named_steps["preprocessing"].get_feature_names_out()
        coefficients = np.abs(np.asarray(estimator.coef_).ravel())
        importance = pd.Series(coefficients, index=names).nlargest(20).sort_values()
    else:
        return
    if "importance" in locals():
        plt.figure(figsize=(8, 7))
        importance.plot.barh()
        plt.xlabel("Feature importance")
        plt.tight_layout()
        plt.savefig(reports_dir / "feature_importance.png", dpi=160)
        plt.close()
