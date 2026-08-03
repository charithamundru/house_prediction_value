"""Streamlit deployment interface for the house price prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
FEATURE_PATH = PROJECT_ROOT / "outputs" / "feature_columns.json"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"
EXPECTED_FEATURES = [
    "square_feet",
    "num_rooms",
    "age",
    "distance_to_city(km)",
]


@st.cache_resource(show_spinner="Loading trained model...")
def load_model() -> Any:
    """Load and cache the complete fitted Scikit-Learn pipeline.

    Raises:
        FileNotFoundError: If the serialized model has not been generated.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model artifact not found. Run the training command before starting the app."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON artifact as a dictionary.

    Args:
        path: Artifact location.

    Raises:
        FileNotFoundError: If the JSON artifact is unavailable.
        ValueError: If the JSON content is not an object.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return payload


def build_prediction_frame(
    square_feet: float,
    num_rooms: int,
    age: int,
    distance_to_city: float,
    configured_features: list[str],
) -> pd.DataFrame:
    """Create a validated single-row model input frame.

    Args:
        square_feet: Property area in square feet.
        num_rooms: Number of rooms.
        age: Property age in years.
        distance_to_city: City-centre distance in kilometres.
        configured_features: Feature names exported by training.

    Raises:
        ValueError: If values or the saved feature schema are invalid.
    """
    if square_feet <= 0:
        raise ValueError("Square feet must be greater than zero.")
    if num_rooms < 1:
        raise ValueError("Number of rooms must be at least one.")
    if age < 0:
        raise ValueError("House age cannot be negative.")
    if distance_to_city < 0:
        raise ValueError("Distance to city cannot be negative.")
    if configured_features != EXPECTED_FEATURES:
        raise ValueError(
            "The deployed form does not match the trained feature schema. "
            "Retrain the model or update the application inputs."
        )
    return pd.DataFrame([{
        "square_feet": square_feet,
        "num_rooms": num_rooms,
        "age": age,
        "distance_to_city(km)": distance_to_city,
    }], columns=configured_features)


def main() -> None:
    """Render the house-price prediction application."""
    st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")
    st.title("🏠 House Price Prediction")
    st.caption("Estimate a house price from its size, rooms, age, and city distance.")

    try:
        model = load_model()
        feature_config = load_json(FEATURE_PATH)
        metrics = load_json(METRICS_PATH)
        configured_features = feature_config.get("features")
        if not isinstance(configured_features, list):
            raise ValueError("feature_columns.json does not contain a features list.")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Application setup error: {error}")
        st.info("Run `.\\.venv\\Scripts\\python.exe -m src.train` to generate artifacts.")
        st.stop()

    with st.form("prediction_form"):
        square_feet = st.number_input(
            "Square feet", min_value=1.0, value=1800.0, step=50.0,
            help="Total property area in square feet.",
        )
        num_rooms = st.number_input(
            "Number of rooms", min_value=1, value=3, step=1,
            help="Total number of rooms in the property.",
        )
        age = st.number_input(
            "House age (years)", min_value=0, value=10, step=1,
            help="Age of the property in completed years.",
        )
        distance_to_city = st.number_input(
            "Distance to city (km)", min_value=0.0, value=10.0, step=0.5,
            help="Distance from the property to the city centre in kilometres.",
        )
        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        try:
            prediction_frame = build_prediction_frame(
                square_feet, num_rooms, age, distance_to_city, configured_features,
            )
            predicted_price = float(model.predict(prediction_frame)[0])
            st.success("Prediction generated")
            st.metric("Estimated house price", f"{predicted_price:,.2f}")
        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Could not create a prediction: {error}")
        except Exception:
            st.error("Prediction failed unexpectedly. Check the model and feature artifacts.")

    with st.expander("Model performance and disclaimer"):
        st.write(f"**Selected model:** {metrics.get('Model', 'Unavailable')}")
        st.write(f"**RMSE:** {metrics.get('RMSE', 'Unavailable')}")
        st.write(f"**MAE:** {metrics.get('MAE', 'Unavailable')}")
        st.write(f"**R²:** {metrics.get('R2', 'Unavailable')}")
        st.caption(
            "This is a machine-learning estimate based on the training data, "
            "not a formal property valuation or financial recommendation."
        )


if __name__ == "__main__":
    main()
