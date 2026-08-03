"""Exploratory data analysis and executive report generation."""

from __future__ import annotations

import logging
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.preprocess import add_eda_price_per_sqft

LOGGER = logging.getLogger(__name__)


def _save_plot(path: Path) -> None:
    """Apply consistent plot layout and save it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_feature_distributions(numeric: pd.DataFrame, reports_dir: Path) -> None:
    """Create bounded boxplot and histogram grids for numeric columns."""
    columns = numeric.columns.tolist()
    columns_per_row = min(3, len(columns))
    rows = ceil(len(columns) / columns_per_row)
    for plot_type, output_name in (("box", "boxplots.png"), ("hist", "feature_distributions.png")):
        figure, axes = plt.subplots(rows, columns_per_row, figsize=(15, 4 * rows), squeeze=False)
        for axis, column in zip(axes.flat, columns):
            if plot_type == "box":
                sns.boxplot(y=numeric[column], ax=axis, color="#4C72B0")
            else:
                sns.histplot(numeric[column].dropna(), bins=25, ax=axis, color="#4C72B0")
            axis.set_title(column)
        for axis in axes.flat[len(columns):]:
            axis.set_visible(False)
        figure.tight_layout()
        figure.savefig(reports_dir / output_name, dpi=160, bbox_inches="tight")
        plt.close(figure)


def detect_geographic_inconsistencies(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Find inconsistent spelling/casing in geographic-looking categories."""
    findings: dict[str, list[str]] = {}
    for column in frame.select_dtypes(include="object"):
        if any(term in column.lower() for term in ("city", "state", "country", "location", "address", "area")):
            values = frame[column].dropna().astype(str)
            normalized = values.str.strip().str.lower()
            conflicts = values.groupby(normalized).unique()
            inconsistent = [", ".join(sorted(items)) for items in conflicts if len(items) > 1]
            if inconsistent:
                findings[column] = inconsistent
    return findings


def calculate_vif(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for numeric predictors after median imputation."""
    numeric = frame.select_dtypes(include="number").copy()
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "VIF"])
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    values: list[float] = []
    for index in range(numeric.shape[1]):
        try:
            values.append(float(variance_inflation_factor(numeric.values, index)))
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            values.append(float("inf"))
    return pd.DataFrame({"feature": numeric.columns, "VIF": values}).sort_values("VIF", ascending=False)


def run_eda(frame: pd.DataFrame, target_column: str, reports_dir: Path, outputs_dir: Path) -> dict[str, object]:
    """Generate EDA figures, CSV diagnostics, and an executive Markdown report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    analysis = add_eda_price_per_sqft(frame, target_column)
    # PricePerSqft includes the target and is therefore EDA-only. Excluding it
    # from correlation and VIF prevents it from appearing as a false predictor.
    numeric = frame.select_dtypes(include="number")
    missing = frame.isna().sum().sort_values(ascending=False)
    duplicates = int(frame.duplicated().sum())
    geographic = detect_geographic_inconsistencies(frame)
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3 - q1
    outlier_counts = ((numeric.lt(q1 - 1.5 * iqr)) | (numeric.gt(q3 + 1.5 * iqr))).sum().sort_values(ascending=False)
    correlations = numeric.corr(numeric_only=True)
    target_corr = correlations[target_column].drop(target_column).abs().sort_values(ascending=False) if target_column in correlations else pd.Series(dtype=float)
    # Keep only one half of the correlation matrix, avoiding duplicate pairs.
    pair_rows = []
    for index, left in enumerate(correlations.columns):
        for right in correlations.columns[index + 1:]:
            value = correlations.loc[left, right]
            if abs(value) >= .80:
                pair_rows.append({"feature_a": left, "feature_b": right, "correlation": value})
    pd.DataFrame(
        pair_rows,
        columns=["feature_a", "feature_b", "correlation"],
    ).to_csv(outputs_dir / "high_correlation_pairs.csv", index=False)
    vif = calculate_vif(numeric.drop(columns=[target_column], errors="ignore"))
    vif.to_csv(outputs_dir / "vif.csv", index=False)
    missing.rename("missing_count").to_csv(outputs_dir / "missing_values.csv")
    outlier_counts.rename("iqr_outlier_count").to_csv(outputs_dir / "outlier_counts.csv")

    if not correlations.empty:
        plt.figure(figsize=(max(8, len(correlations) * .7), max(6, len(correlations) * .6)))
        sns.heatmap(correlations, cmap="coolwarm", center=0, square=True)
        _save_plot(reports_dir / "correlation_heatmap.png")
    if not numeric.empty:
        _plot_feature_distributions(numeric, reports_dir)

    top_predictors = ", ".join(f"{name} ({value:.2f})" for name, value in target_corr.head(5).items()) or "Not available"
    missing_summary = ", ".join(f"{name}: {int(value)}" for name, value in missing[missing > 0].head(8).items()) or "No missing values"
    outlier_summary = ", ".join(f"{name}: {int(value)}" for name, value in outlier_counts[outlier_counts > 0].head(8).items()) or "No IQR outliers"
    report = f"""# Executive EDA Report\n\n## Dataset overview\n- Rows: {frame.shape[0]:,}\n- Columns: {frame.shape[1]:,}\n- Duplicate rows: {duplicates:,}\n- Target: `{target_column}`\n\n## Data quality\n{missing_summary}. Geographic spelling/casing inconsistencies: {geographic or 'none detected'}.\n\n## Missing values\nNumeric missing values are treated distribution-aware in the training pipeline: median for strongly skewed features and mean otherwise; categorical values use the most frequent category. All choices are learned from training data only.\n\n## Outliers\nIQR-rule counts — {outlier_summary}. Outliers are retained by default because property size and price extremes can be real market observations; model comparison determines sensitivity.\n\n## Distribution\nTarget skewness: {skew(pd.to_numeric(frame[target_column], errors='coerce').dropna()):.2f}. A log target transform is automatically evaluated when target skewness is positive and above 1.\n\n## Correlation and multicollinearity\nTop absolute target correlations: {top_predictors}. High-correlation pairs (absolute correlation ≥ 0.80) and VIF values are exported to `outputs/`.\n\n## Initial hypotheses\n1. The strongest numeric predictors will materially explain price.\n2. Nonlinear interactions between size and amenities may favor Random Forest over linear models.\n3. Categorical location/amenity variables may add predictive power after one-hot encoding.\n4. Price-per-square-foot is an EDA-only diagnostic and is excluded from modeling to prevent target leakage.\n"""
    (reports_dir / "executive_eda_report.md").write_text(report, encoding="utf-8")
    return {"duplicates": duplicates, "missing": missing.to_dict(), "top_predictors": target_corr.head(5).to_dict()}
