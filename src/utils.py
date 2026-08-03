"""Shared utilities for configuration, data discovery, and persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_logging() -> None:
    """Configure concise, application-wide logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def find_data_file(data_path: str | Path | None = None) -> Path:
    """Return a validated CSV path, preferring the conventional data folder.

    Args:
        data_path: Optional explicit dataset path.

    Raises:
        FileNotFoundError: If no CSV dataset can be found.
    """
    candidate = Path(data_path) if data_path else PROJECT_ROOT / "data" / "Housing.csv"
    if candidate.is_file():
        return candidate
    alternatives = list((PROJECT_ROOT / "data").glob("*.csv"))
    if len(alternatives) == 1:
        return alternatives[0]
    raise FileNotFoundError(
        f"Dataset not found at {candidate}. Place Housing.csv in {PROJECT_ROOT / 'data'} "
        "or provide --data-path."
    )


def infer_target_column(frame: pd.DataFrame) -> str:
    """Infer a price target column without requiring a fixed dataset schema."""
    preferred = ("price", "saleprice", "sale_price", "target")
    normalized = {column.lower().replace(" ", "_"): column for column in frame.columns}
    for name in preferred:
        if name in normalized:
            return normalized[name]
    matches = [column for column in frame.columns if "price" in column.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("Cannot infer target column. Expected a column named price, saleprice, or target.")


def json_default(value: Any) -> Any:
    """Convert NumPy and pandas scalars to JSON-compatible values."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(payload: dict[str, Any] | list[Any], path: Path) -> None:
    """Persist JSON with deterministic, readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
