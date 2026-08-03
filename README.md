# House Price Prediction

Production-oriented, leakage-safe regression project using pandas and scikit-learn. It automatically detects common `price` target names and supports the common `Housing.csv` schema.

## Setup

1. Create a Python 3.12+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy the dataset to `data/Housing.csv`.
4. Run: `python -m src.train`

For a different location: `python -m src.train --data-path path/to/Housing.csv`.

## What the pipeline does

- Performs EDA, missing-value/outlier/correlation/VIF diagnostics, and writes `reports/executive_eda_report.md`.
- Splits before fitting imputers, scaling, and encoding. Numerical imputation is distribution-aware; categorical imputation uses the most frequent category.
- Compares Linear Regression, Ridge, and Random Forest with five-fold CV; tunes Random Forest with `GridSearchCV`.
- Uses `log1p` for a sufficiently positive-skewed target, then converts predictions back to price units.
- Prevents leakage: `PricePerSqft` is an EDA-only diagnostic because it incorporates the target.

## Output artifacts

- `models/model.pkl`: complete fitted pipeline.
- `models/preprocessing.pkl`: fitted `ColumnTransformer`.
- `outputs/metrics.json`, `outputs/feature_columns.json`, `outputs/model_comparison.csv`, diagnostic CSVs.
- `reports/`: executive report and EDA/evaluation figures.

## Project layout

`src/` contains independent EDA, preprocessing, training, evaluation, and utility modules. The notebook provides a narrated entry point and invokes the same tested pipeline, so there is no duplicated modeling code.

## Web deployment

Train the project once so `models/model.pkl` and the JSON artifacts exist, then launch the Streamlit interface:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app accepts a property's square feet, number of rooms, age, and distance to city, then uses the serialized pipeline to predict a price. It does not retrain the model when users make predictions.

For cloud deployment, commit `app.py`, `requirements.txt`, `.streamlit/config.toml`, `models/model.pkl`, and the required `outputs/*.json` artifacts. Configure the service to run `streamlit run app.py`.
