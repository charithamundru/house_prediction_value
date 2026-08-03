# Executive EDA Report

## Dataset overview
- Rows: 10,000
- Columns: 5
- Duplicate rows: 0
- Target: `price`

## Data quality
No missing values. Geographic spelling/casing inconsistencies: none detected.

## Missing values
Numeric missing values are treated distribution-aware in the training pipeline: median for strongly skewed features and mean otherwise; categorical values use the most frequent category. All choices are learned from training data only.

## Outliers
IQR-rule counts — square_feet: 83, price: 64. Outliers are retained by default because property size and price extremes can be real market observations; model comparison determines sensitivity.

## Distribution
Target skewness: 0.01. A log target transform is automatically evaluated when target skewness is positive and above 1.

## Correlation and multicollinearity
Top absolute target correlations: square_feet (0.76), distance_to_city(km) (0.42), num_rooms (0.34), age (0.29). High-correlation pairs (absolute correlation ≥ 0.80) and VIF values are exported to `outputs/`.

## Initial hypotheses
1. The strongest numeric predictors will materially explain price.
2. Nonlinear interactions between size and amenities may favor Random Forest over linear models.
3. Categorical location/amenity variables may add predictive power after one-hot encoding.
4. Price-per-square-foot is an EDA-only diagnostic and is excluded from modeling to prevent target leakage.
