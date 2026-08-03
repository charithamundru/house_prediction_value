# House Price Prediction — Presentation and Viva Notes

## 1. Project in one minute

This project predicts a house's **price** from its characteristics. The dataset contains 10,000 housing records with `square_feet`, `num_rooms`, `age`, `distance_to_city(km)`, and the target `price`.

The project performs exploratory data analysis (EDA), applies leakage-safe preprocessing, trains three regression models, compares them using cross-validation and a held-out test set, and saves the best final model for reuse.

**Final selected model:** Linear Regression  
**Test RMSE:** 19,658.17  
**Test MAE:** 15,596.12  
**Test R²:** 0.9601

In simple language: the selected model explains roughly 96% of the price variation in the test data. Its prediction is off by about 15.6k price units on average.

---

## 2. Project objective

**Problem statement:** Given the available property details, predict a house's selling price accurately and reproducibly.

**Business value:** A reliable estimate can help buyers, sellers, agents, and analysts compare properties, set prices, and identify unusually expensive or inexpensive listings.

**Machine-learning task:** Supervised learning, specifically **regression**, because price is a continuous numeric value.

---

## 3. Dataset explanation

| Column | Meaning | Type | Expected relationship with price |
|---|---|---|---|
| `square_feet` | Property size | Numeric | Larger houses usually cost more |
| `num_rooms` | Number of rooms | Numeric | More rooms may increase price |
| `age` | House age | Numeric | Older homes may have a lower price, all else equal |
| `distance_to_city(km)` | Distance from city centre | Numeric | Greater distance may reduce price, depending on the market |
| `price` | House price | Numeric target | Variable the model predicts |

The **features** are the input columns. The **target** is `price`.

---

## 4. End-to-end project workflow

```text
CSV dataset
    ↓
EDA and data-quality checks
    ↓
Train/test split (80% / 20%)
    ↓
Preprocessing fitted on training data only
    ↓
5-fold cross-validation and model comparison
    ↓
Random Forest hyperparameter tuning
    ↓
Test-set evaluation
    ↓
Select best model and save artifacts
```

### Why split before preprocessing?

This prevents **data leakage**. If information from the test set is used when calculating means, scaling values, or deciding transformations, test performance is unrealistically optimistic. The test set must behave like completely unseen future data.

---

## 5. EDA: what was checked and why

The EDA module creates an executive report and charts in `reports/`.

| Check | Why it matters |
|---|---|
| Shape and schema | Confirms number of observations, columns, and data types |
| Descriptive statistics | Shows ranges, average values, and unusual values |
| Missing values | Identifies whether imputation is needed |
| Duplicate rows | Detects repeated records that can bias analysis |
| Geographic consistency | Detects location spelling/case variations when geographic columns exist |
| Boxplots and IQR | Identifies possible outliers |
| Histograms | Shows feature distributions and skewness |
| Correlation heatmap | Shows linear relationships between numeric variables |
| VIF | Detects multicollinearity between predictors |

### Outliers

The IQR rule marks a value as a possible outlier when it is below:

`Q1 - 1.5 × IQR`

or above:

`Q3 + 1.5 × IQR`

where `IQR = Q3 - Q1`.

Outliers are not automatically deleted. Very large or expensive homes can be genuine data, not mistakes. Removing them without investigation can make the model worse in real use.

### Correlation and VIF

- **Correlation** ranges from -1 to +1. Positive means two variables tend to rise together; negative means one tends to fall as the other rises.
- **VIF (Variance Inflation Factor)** measures whether a predictor is strongly explained by other predictors. A high VIF can make linear-model coefficients unstable. Values above 5–10 often need investigation.

### Important leakage note

`PricePerSqft = price / square_feet` is meaningful as an EDA business measure, but it includes the target (`price`). It is excluded from model predictors, correlations, and VIF because using it to predict price would leak the answer.

---

## 6. Preprocessing explanation

Preprocessing is implemented using a Scikit-Learn `Pipeline` and `ColumnTransformer`.

### Numerical columns

1. Missing values are imputed.
   - Strongly skewed numeric columns: median.
   - Other numeric columns: mean.
2. `StandardScaler` standardizes each numeric feature:

`z = (x - mean) / standard deviation`

Scaling makes features comparable, which is especially important for Ridge Regression.

### Categorical columns

If categorical columns exist, the pipeline:

1. Replaces missing values with the most frequent category.
2. Uses `OneHotEncoder(handle_unknown="ignore")`.

One-hot encoding converts a category such as `furnishing = furnished` into binary indicator columns. `handle_unknown="ignore"` ensures the model does not crash when a new category appears in future data.

### Target transformation

The pipeline checks target skewness. If the price target is strongly positively skewed, it uses `log1p(price)` for training and converts predictions back with `expm1()`. This can make highly skewed prices easier to model. In this dataset, the target did not meet the condition, so no log transform was used.

---

## 7. Models used

### Linear Regression

Linear Regression estimates a linear relationship:

`price = intercept + b1 × square_feet + b2 × rooms + ...`

It is simple, fast, interpretable, and performs very well when the data relationship is mostly linear.

### Ridge Regression

Ridge is Linear Regression with L2 regularization. It penalizes excessively large coefficients and can be more stable when predictors are correlated.

### Random Forest Regressor

Random Forest builds many decision trees and averages their predictions. It can learn nonlinear relationships and interactions, but may not beat a well-specified linear model.

### Hyperparameter tuning

`GridSearchCV` tunes the Random Forest using five-fold cross-validation across:

- `n_estimators`: number of trees
- `max_depth`: maximum tree depth
- `min_samples_split`: samples required to split a node
- `min_samples_leaf`: samples required at a leaf

The tuned Random Forest was retained in the comparison, but not selected because Linear Regression had the lower test RMSE.

---

## 8. Validation and metrics

### 5-fold cross-validation

The training data is split into five folds. The model trains on four folds and validates on the remaining fold, repeating five times. The average score is more reliable than a single training split.

### Holdout test set

Twenty percent of records are held out until final evaluation. This test set estimates expected performance on new, unseen houses.

### Metrics

| Metric | Meaning | Better value |
|---|---|---|
| RMSE | Square root of the average squared error; penalizes large mistakes more | Lower |
| MAE | Average absolute difference between actual and predicted prices | Lower |
| R² | Fraction of target variation explained by the model | Higher, closer to 1 |

Formulas:

- `MAE = average(|actual - predicted|)`
- `RMSE = sqrt(average((actual - predicted)²))`
- `R² = 1 - (model squared error / baseline squared error)`

### Results

| Model | RMSE | MAE | R² | Decision |
|---|---:|---:|---:|---|
| Linear Regression | 19,658.17 | 15,596.12 | 0.9601 | Selected |
| Ridge | 19,658.23 | 15,595.83 | 0.9601 | Very close second |
| Tuned Random Forest | 22,266.57 | 17,816.04 | 0.9488 | Not selected |
| Baseline Random Forest | 22,560.43 | 18,021.65 | 0.9474 | Not selected |

### Why was Linear Regression selected?

It achieved the lowest RMSE and highest R² on the held-out test set. It is also easier to interpret and smaller to deploy. The model choice is based on measured performance, not the complexity of the algorithm.

---

## 9. How to read the charts

| File | What to look for |
|---|---|
| `correlation_heatmap.png` | Strong positive/negative relationships; very similar predictor pairs |
| `boxplots.png` | Values far beyond whiskers are possible outliers |
| `feature_distributions.png` | Skewed, multi-peaked, or unusual feature shapes |
| `actual_vs_predicted.png` | Points close to the diagonal line indicate accurate predictions |
| `residual_plot.png` | Errors should appear randomly spread around zero |
| `error_distribution.png` | Errors should be concentrated near zero and reasonably balanced |
| `feature_importance.png` | Variables with greater influence in the final model; for linear models this uses absolute coefficient magnitude after scaling |

If residuals form a curve or funnel shape, it can indicate that the model misses a nonlinear pattern or that error variance changes with price.

---

## 10. Saved deployment artifacts

| Artifact | Purpose |
|---|---|
| `models/model.pkl` | Complete fitted pipeline: preprocessing plus selected model |
| `models/preprocessing.pkl` | Fitted preprocessing transformer |
| `outputs/metrics.json` | Selected-model metrics and tuning information |
| `outputs/feature_columns.json` | Expected feature names and target name |
| `outputs/model_comparison.csv` | Results for every candidate model |
| `outputs/predictions.csv` | Actual, predicted, and residual values for the test set |
| `reports/executive_eda_report.md` | Executive data-quality and EDA summary |

To predict with the saved model in Python:

```python
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
new_houses = pd.DataFrame([
    {
        "square_feet": 2000,
        "num_rooms": 3,
        "age": 10,
        "distance_to_city(km)": 8.5,
    }
])
prediction = model.predict(new_houses)
print(prediction[0])
```

Use the exact predictor names shown in `outputs/feature_columns.json`.

---

## 11. Execution commands

Run all commands from the project folder:

```powershell
cd "C:\Users\91901\OneDrive\Desktop\house_price _priciction"
```

### First-time setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Train the project

```powershell
.\.venv\Scripts\python.exe -m src.train
```

### Train using a specific dataset path

```powershell
.\.venv\Scripts\python.exe -m src.train --data-path "C:\path\to\Housing.csv"
```

### View final metrics

```powershell
Get-Content outputs\metrics.json
```

### View the model comparison table

```powershell
Import-Csv outputs\model_comparison.csv | Format-Table -AutoSize
```

### Start the notebook

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\HousePricePrediction.ipynb
```

### Open the reports folder in File Explorer

```powershell
explorer reports
```

---

## 12. Likely questions and answers

### What kind of ML problem is this?

It is supervised regression because it learns from historical feature-price pairs and predicts a continuous numeric target, house price.

### Why did you use an 80/20 train-test split?

The 80% training data provides enough examples to learn patterns, while 20% stays unseen until final testing. This gives an honest estimate of generalization performance.

### What is data leakage, and how did you prevent it?

Data leakage occurs when the model learns information that would not be available when predicting a new house. I split the data before fitting preprocessing, put preprocessing inside a pipeline, and excluded target-derived `PricePerSqft` from predictors.

### Why use a Pipeline and ColumnTransformer?

They make preprocessing reproducible, apply the same transformations during training and prediction, and reduce leakage risk. `ColumnTransformer` lets numeric and categorical columns receive different transformations.

### Why StandardScaler?

It transforms numeric values to a comparable scale. This is important for regularized linear models such as Ridge, because they are sensitive to feature magnitudes.

### Why use OneHotEncoder with `handle_unknown="ignore"`?

Machine-learning models need numbers rather than text categories. One-hot encoding creates binary columns, and `handle_unknown="ignore"` prevents prediction failure when an unseen category appears later.

### Why calculate both RMSE and MAE?

MAE gives typical average error. RMSE gives more weight to large errors. Reporting both gives a fuller view of prediction quality.

### Why is a lower RMSE better?

RMSE measures prediction error in the same units as price. A lower value means predictions are closer to actual prices.

### What does R² = 0.9601 mean?

It means the model explains about 96.01% of variation in house price in the held-out test data. It does not mean that 96% of every individual prediction is correct.

### Why did Linear Regression beat Random Forest?

The current data appears to have a strongly linear relationship between the features and price. A simpler model can generalize better than a more complex model when nonlinear interactions add little useful information.

### Why tune Random Forest if it was not selected?

Tuning provides a fair comparison against a well-configured nonlinear model. The final choice should be based on validation performance, not an assumption that a complex algorithm is always better.

### What is multicollinearity?

It is when predictors are strongly correlated with one another. It can make linear-model coefficients unstable. Correlation checks and VIF were used to detect it.

### What are the project limitations?

The dataset has only four input variables and may not include important real-market factors such as location quality, property condition, bathrooms, parking, neighborhood amenities, market trends, and date of sale. The model should be retrained and monitored when new data becomes available.

### How could the project be improved?

Add richer property and location features, validate the data source, test XGBoost, use time-aware validation if data contains sale dates, perform residual analysis by price band/location, create an API or web application, and monitor prediction error after deployment.

---

## 13. Suggested closing statement

> I built a reproducible house-price regression pipeline that starts with EDA and data-quality checks, uses leakage-safe preprocessing, compares linear and ensemble models with five-fold cross-validation, and evaluates them on unseen test data. Linear Regression was selected because it achieved the best test performance: RMSE of 19,658, MAE of 15,596, and R² of 0.9601. The final pipeline and reports are serialized for repeatable deployment and future predictions.
## execution commands
.\.venv\Scripts\python.exe -m streamlit run app.py
