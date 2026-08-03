"""Feature engineering and leakage-safe preprocessing construction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingBundle:
    """Fitted-schema-independent preprocessing configuration."""

    transformer: ColumnTransformer
    numerical_columns: list[str]
    categorical_columns: list[str]
    engineered_columns: list[str]


def _find_area_column(frame: pd.DataFrame) -> str | None:
    """Find a likely area column, if one exists."""
    candidates = [column for column in frame.columns if any(
        token in column.lower().replace(" ", "_")
        for token in ("area", "sqft", "square_feet", "squarefeet")
    )]
    return candidates[0] if candidates else None


def engineer_features(frame: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """Create safe predictor features.

    Price-per-square-foot is useful only as a descriptive EDA measure. It is
    intentionally not created as a training feature because it uses the target
    and would leak the answer into the model.
    """
    result = frame.copy()
    area_column = _find_area_column(result)
    if area_column:
        result[area_column] = pd.to_numeric(result[area_column], errors="coerce")
    return result


def add_eda_price_per_sqft(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Return an EDA-only copy with price per square foot when possible."""
    result = frame.copy()
    area_column = _find_area_column(result)
    if area_column:
        area = pd.to_numeric(result[area_column], errors="coerce")
        price = pd.to_numeric(result[target_column], errors="coerce")
        result["PricePerSqft"] = price.div(area.where(area > 0)).replace([np.inf, -np.inf], np.nan)
    return result


def build_preprocessor(features: pd.DataFrame) -> PreprocessingBundle:
    """Build preprocessing using train-derived columns and imputation strategy.

    Numeric features with absolute skew above one receive median imputation;
    remaining numeric features use mean imputation. Categories use the mode.
    """
    numerical = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features.columns if column not in numerical]
    skewed = [column for column in numerical if abs(features[column].skew()) > 1]
    regular = [column for column in numerical if column not in skewed]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if regular:
        transformers.append(("numeric_mean", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())
        ]), regular))
    if skewed:
        transformers.append(("numeric_median", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())
        ]), skewed))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable features were found.")
    LOGGER.info("Preprocessor: %d numeric, %d categorical columns", len(numerical), len(categorical))
    return PreprocessingBundle(
        transformer=ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False),
        numerical_columns=numerical,
        categorical_columns=categorical,
        engineered_columns=[],
    )
