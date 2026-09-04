# Food Delivery Time Prediction

An end-to-end machine learning project for predicting **food delivery time (`Time_taken(min)`)** from order, delivery-person, traffic, weather, vehicle, location, and contextual features.

The project focuses not only on model training, but on the complete tabular ML workflow: **data cleaning, missing-value treatment, feature engineering, exploratory analysis, outlier handling, categorical encoding, leakage-safe scaling, model comparison, and feature-importance analysis**.

---

## Project Overview

Food delivery time is influenced by several interacting factors such as:

- Delivery-person attributes
- Restaurant-to-customer distance
- Road traffic density
- Weather conditions
- Vehicle condition and type
- Number of simultaneous deliveries
- Festival periods
- City
- Order context

The goal is to learn these relationships from historical delivery records and build a regression model that predicts the delivery time for unseen orders.

### Business objective

A reliable delivery-time prediction model can support:

- More accurate ETA estimation
- Better customer communication
- Delivery operations planning
- Rider allocation and dispatch decisions
- Identification of operational bottlenecks
- More data-driven service-level monitoring

---

## Key Results

Four regression algorithms were trained and evaluated on the same held-out test split.

| Model | Test RMSE | Test R² | Interpretation |
|---|---:|---:|---|
| Linear Regression | 6.22 | 0.568 | Baseline; limited at modelling non-linear interactions |
| Decision Tree | 5.46 | 0.667 | Better than linear regression but strongly overfits |
| Random Forest | 4.06 | 0.816 | Large generalization improvement through bagging |
| **XGBoost** | **3.99** | **0.823** | **Best-performing model** |

### Main finding

**XGBoost achieved the best recorded result with RMSE = 3.99 minutes and R² = 0.823.**

This shows that the delivery-time problem contains substantial **non-linear relationships and feature interactions** that are not captured well by a simple linear model.

The single Decision Tree achieved almost perfect training performance (`R² ≈ 0.99998`) but much lower test performance (`R² ≈ 0.667`), providing a clear example of overfitting. Ensemble tree methods substantially improved generalization.

---

## Dataset

The notebook uses a food-delivery dataset containing operational, geographic, temporal, weather, traffic, vehicle, and order information.

### Raw columns used in the project

| Column | Description |
|---|---|
| `ID` | Order identifier |
| `Delivery_person_ID` | Delivery-person identifier |
| `Delivery_person_Age` | Delivery person's age |
| `Delivery_person_Ratings` | Delivery person's rating |
| `Restaurant_latitude` | Restaurant latitude |
| `Restaurant_longitude` | Restaurant longitude |
| `Delivery_location_latitude` | Customer/delivery location latitude |
| `Delivery_location_longitude` | Customer/delivery location longitude |
| `Order_Date` | Order date |
| `Time_Orderd` | Time at which the order was placed |
| `Time_Order_picked` | Time at which the order was picked up |
| `Weatherconditions` | Weather condition |
| `Road_traffic_density` | Traffic level |
| `Vehicle_condition` | Vehicle condition score |
| `Type_of_order` | Order category |
| `Type_of_vehicle` | Delivery vehicle type |
| `multiple_deliveries` | Number of deliveries handled simultaneously |
| `Festival` | Festival indicator |
| `City` | City category |
| `Time_taken(min)` | **Target: delivery time in minutes** |

> The uploaded `test.csv` contains the feature columns but does not contain the target column. The merged notebook follows the final project version in which a single combined food-delivery CSV is loaded first and the train/test split is performed later with `train_test_split`.

---

# Machine Learning Pipeline

```text
Raw Food Delivery Data
        │
        ▼
Data Inspection
        │
        ▼
Data Cleaning
  ├─ Strip whitespace
  ├─ Convert "NaN" strings → np.nan
  └─ Clean target/weather text
        │
        ▼
Missing Value Treatment
  ├─ Numeric → median
  └─ Categorical → mode
        │
        ▼
Feature Engineering
  ├─ order_hour
  └─ distance_km (Haversine)
        │
        ▼
Feature Selection
  ├─ Remove IDs
  ├─ Remove raw coordinates
  └─ Remove redundant raw time columns
        │
        ▼
EDA + Outlier Handling
  ├─ Distance validation
  └─ Rating capping
        │
        ▼
Categorical Encoding
  ├─ Ordinal encoding: traffic density
  └─ One-hot encoding: nominal categories
        │
        ▼
80/20 Train-Test Split
        │
        ▼
StandardScaler
  ├─ fit on training data only
  └─ transform train + test
        │
        ▼
Model Training
  ├─ Linear Regression
  ├─ Decision Tree
  ├─ Random Forest
  └─ XGBoost
        │
        ▼
Evaluation
  ├─ RMSE
  └─ R²
        │
        ▼
Best Model: XGBoost
```

---

# 1. Data Cleaning

The original dataset contains several quality issues that need to be corrected before modelling.

### Whitespace and missing values

Object/string columns may contain leading or trailing spaces, and missing values can appear as the literal string `"NaN"` rather than an actual NumPy missing value.

The pipeline therefore:

1. Strips whitespace from object columns.
2. Replaces the string `"NaN"` with `np.nan`.

This is important because later missing-value operations such as `fillna()` require genuine missing values.

### Target cleaning

The target is stored in the form:

```text
(min) 24
```

The pipeline extracts the numeric component so that:

```text
(min) 24  →  24
```

The result becomes a numeric regression target.

### Weather cleaning

Weather values are represented as strings such as:

```text
conditions Sunny
conditions Rainy
conditions Fog
```

The pipeline extracts the actual condition:

```text
conditions Sunny → Sunny
```

---

# 2. Missing Value Treatment

Missing values are handled using a simple and robust strategy.

### Numeric columns → Median

The following numeric fields are filled using their median:

- `Delivery_person_Age`
- `Delivery_person_Ratings`
- `multiple_deliveries`
- `Vehicle_condition`

Median imputation is preferred here because it is less sensitive to extreme values than mean imputation.

### Categorical columns → Mode

Categorical columns are filled using the most frequent category:

- `City`
- `Festival`
- `Road_traffic_density`
- `Weatherconditions`

This preserves valid categorical values without introducing arbitrary new categories.

---

# 3. Feature Engineering

Feature engineering is one of the most important parts of this project.

Two meaningful features are engineered from the raw data.

## 3.1 `order_hour`

The raw `Time_Orderd` field is converted to a datetime representation and the hour is extracted:

```text
Time_Orderd
     │
     ▼
datetime
     │
     ▼
order_hour
```

This allows the model to capture time-of-day effects.

The notebook's exploratory analysis showed a rush-hour pattern, with delivery times increasing around lunch and evening periods.

An additional experimental feature called `order_period` was created:

```text
11–14  → Lunch
17–21  → Dinner
else   → Normal
```

However, in the final Part 4 modelling run, both `order_hour` and `order_period` were dropped before training. The experiment is intentionally retained in the notebook because it documents the modelling investigation and provides a natural direction for future experimentation.

---

## 3.2 `distance_km` — Haversine Distance

Latitude and longitude values are not directly as useful to the model as the physical distance between the restaurant and delivery location.

The project therefore computes the great-circle distance using the **Haversine formula**.

For two geographical points:

- Restaurant: `(lat1, lon1)`
- Delivery location: `(lat2, lon2)`

the implementation computes:

```text
φ1 = radians(lat1)
φ2 = radians(lat2)

Δφ = radians(lat2 - lat1)
Δλ = radians(lon2 - lon1)

a = sin²(Δφ/2)
    + cos(φ1) · cos(φ2) · sin²(Δλ/2)

c = 2 · arcsin(√a)

distance = R · c
```

where:

```text
R = 6371 km
```

The calculation is vectorized with NumPy and applied to the complete pandas Series.

### Why this feature matters

The raw coordinates represent absolute geographic positions. What matters operationally for delivery time is much closer to the **distance that a rider must cover**.

The exploratory analysis showed a positive relationship between `distance_km` and delivery time (approximately 0.32 correlation in the notebook), confirming that distance contains useful predictive signal.

---

# 4. Feature Selection

Identifier columns are not useful predictive signals and can encourage memorization, so the modelling feature set excludes:

- `ID`
- `Delivery_person_ID`

Raw latitude/longitude columns are also not directly used after creating `distance_km`.

Similarly, raw time information is transformed rather than feeding the original time string directly to the model.

The engineered modelling matrix is built from delivery-person attributes, distance, traffic, weather, vehicle, order, festival, and city information.

---

# 5. Exploratory Data Analysis and Outlier Handling

The project includes per-feature inspection before model training.

## Distance outliers

Some rows produce impossible food-delivery distances in the thousands of kilometres.

The notebook identifies these as corrupted geographic records, likely caused by malformed or sign-flipped latitude/longitude values.

Rows with:

```text
distance_km >= 100
```

are removed because such distances are unrealistic for the delivery problem represented by the dataset.

Both `X` and `y` are filtered using the same mask and their indices are reset so their alignment is preserved.

---

## Delivery-person ratings

Ratings are expected to be on a 1–5 scale, but some records contain values greater than 5.

The project therefore caps the upper end:

```python
X["Delivery_person_Ratings"] = X["Delivery_person_Ratings"].clip(upper=5)
```

This prevents obvious data-entry noise from disproportionately affecting the model.

---

## Weather analysis

The EDA compares delivery-time statistics across weather categories.

The notebook records:

- Sunny as the fastest group at approximately 21.9 minutes mean delivery time.
- Cloudy/Fog among the slowest groups at approximately 28.9 minutes.

This supports retaining weather as a predictive feature.

---

## Traffic analysis

The EDA shows a strong ordered relationship:

```text
Low     ≈ 21.4 min
Medium  ≈ 26.7 min
High    ≈ 27.2 min
Jam     ≈ 31.2 min
```

Because these categories have a natural order, traffic density is treated as an **ordinal variable** rather than a nominal variable.

---

## Festival analysis

Festival status shows a particularly strong effect in the notebook:

```text
Normal day   ≈ 25.9 min
Festival     ≈ 45.5 min
```

This makes `Festival` one of the most meaningful categorical predictors in the dataset.

---

# 6. Categorical Encoding

Two encoding strategies are deliberately used.

## Ordinal Encoding

`Road_traffic_density` has an intrinsic order:

```text
Low → Medium → High → Jam
```

It is therefore mapped to:

```python
{
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Jam": 3
}
```

This preserves the ordering information.

## One-Hot Encoding

The following variables do not have a natural numeric ordering:

- `Weatherconditions`
- `Type_of_order`
- `Type_of_vehicle`
- `Festival`
- `City`

They are one-hot encoded using:

```python
pd.get_dummies(..., drop_first=True)
```

`drop_first=True` removes one reference category from each categorical group and avoids perfect multicollinearity in the resulting design matrix, which is particularly relevant for the Linear Regression baseline.

---

# 7. Train/Test Split and Data Leakage Prevention

The final prepared dataset is split using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This creates an:

```text
80% training
20% testing
```

split.

## Standardization

`StandardScaler` is applied before model training.

The critical detail is that the scaler is:

```text
fit → training set only
transform → training set
transform → test set
```

The project explicitly avoids fitting the scaler on the complete dataset.

This prevents **test-set information leakage** into the training process.

The scaled NumPy arrays are converted back into DataFrames so that feature names remain available during feature-importance analysis.

---

# 8. Models

## 8.1 Linear Regression

Linear Regression is used as the baseline.

Recorded performance:

```text
RMSE ≈ 6.22
R²   ≈ 0.568
```

The model provides a useful reference point but cannot adequately model complex interactions such as:

```text
traffic × distance × festival
```

or other non-linear relationships.

---

## 8.2 Decision Tree Regressor

The Decision Tree improves the test result:

```text
Test RMSE ≈ 5.46
Test R²   ≈ 0.667
```

However, it shows extreme overfitting.

Training performance:

```text
Train RMSE ≈ 0.04
Train R²   ≈ 0.99998
```

while test performance is substantially worse.

This is a classic example of a high-capacity, unregularized tree memorizing the training data.

---

## 8.3 Random Forest Regressor

Random Forest uses an ensemble of trees and averages their predictions to reduce variance.

Configuration used in the notebook:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

Recorded result:

```text
RMSE ≈ 4.06
R²   ≈ 0.816
```

This is a large improvement over both Linear Regression and the single Decision Tree.

### Recorded important features

The original notebook identifies the strongest Random Forest feature importances approximately as:

```text
Delivery_person_Ratings  0.210
distance_km              0.147
Road_traffic_density     0.127
Vehicle_condition        0.121
Delivery_person_Age      0.106
```

---

## 8.4 XGBoost

XGBoost is the best-performing model in the recorded experiment.

Configuration:

```python
XGBRegressor(
    n_estimators=100,
    random_state=42
)
```

Recorded result:

```text
RMSE ≈ 3.99
R²   ≈ 0.823
```

### Why it performs well

Gradient boosting builds trees sequentially, with later trees learning to correct errors made by previous trees.

This allows XGBoost to represent non-linear patterns and interactions more effectively than the linear baseline while also generalizing better than the single unrestricted Decision Tree.

### Recorded important features

The notebook's XGBoost importance analysis highlights:

```text
multiple_deliveries   ≈ 0.160
Vehicle_condition     ≈ 0.132
Road_traffic_density  ≈ 0.126
Festival_Yes          ≈ 0.109
```

Feature-importance rankings are model-specific, so the Random Forest and XGBoost rankings should be interpreted as complementary rather than identical.

---

# 9. Model Comparison

```text
                           Test RMSE      Test R²
Linear Regression             6.22         0.568
Decision Tree                 5.46         0.667
Random Forest                 4.06         0.816
XGBoost                       3.99         0.823   ← Best
```

### What the comparison tells us

The progression is important:

```text
Linear Regression
      ↓
Decision Tree
      ↓
Random Forest
      ↓
XGBoost
```

The biggest practical lesson is that **ensemble tree methods capture the non-linear structure of delivery-time prediction much better than a simple linear model or a single unrestricted tree**.

---

# 10. What Drives Delivery Time?

Across the feature-importance analysis, several variables repeatedly emerge as important signals:

- Delivery-person rating
- Delivery-person age
- Restaurant-to-delivery distance
- Road traffic density
- Vehicle condition
- Number of simultaneous deliveries
- Festival status

These variables represent a mixture of:

```text
Human factors
    +
Geographic factors
    +
Traffic / operational factors
    +
Vehicle factors
    +
Contextual demand effects
```

This combination explains why a purely linear model is insufficient for the problem.

---

# 11. Repository Structure

A recruiter-friendly repository can use the following structure:

```text
food-delivery-time-prediction/
│
├── notebooks/
│   └── Project_1_Complete_Merged.ipynb
│
├── data/
│   └── test.csv
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE                # optional
```

If the original training dataset cannot legally or practically be redistributed, keep it out of the repository and document the expected local path or dataset source instead.

For a public recruiter-facing repository, this is generally preferable to committing a large or license-restricted raw dataset.

---

# 12. How to Run the Project

## Step 1 — Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/food-delivery-time-prediction.git
cd food-delivery-time-prediction
```

## Step 2 — Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

## Step 4 — Launch Jupyter

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Open:

```text
notebooks/Project_1_Complete_Merged.ipynb
```

### Important dataset note

The merged notebook currently reflects the final assignment version in which the combined food-delivery CSV is loaded from a Google Drive path.

For a fully portable GitHub version, replace that machine-specific path with the repository's dataset path, for example:

```python
train_df = pd.read_csv("../data/food_delivery.csv")
```

or an appropriate path based on the notebook location.

Do not leave a personal Google Drive path in the recruiter-facing version.

---

# 13. Requirements

The project uses Python with the following major libraries:

- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- Jupyter Notebook / JupyterLab

A minimal `requirements.txt` is:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

For maximum reproducibility, dependencies can later be pinned to tested versions.

---

# 14. Reproducibility

The project uses fixed random seeds where applicable:

```python
random_state=42
```

This is used for the train/test split and tree-based model initialization.

However, the reported metrics should be understood as results from the documented experiment rather than a claim of production-level benchmark performance.

A stronger experimental protocol would include:

- K-fold cross-validation
- Hyperparameter search
- Repeated experiments with different seeds
- Validation-set tracking
- More extensive error analysis

---

# 15. Limitations

This project is a strong end-to-end ML exercise, but it is not yet a production prediction system.

Current limitations include:

### Single train/test split

The primary evaluation is based on one 80/20 split with `random_state=42`.

Cross-validation would give a more robust estimate of generalization.

### Limited hyperparameter tuning

The models use straightforward configurations rather than an extensive search over parameters.

### Decision Tree comparison is intentionally unregularized

The tree was allowed to grow without a `max_depth`, which demonstrates overfitting clearly but is not a fair optimized-tree benchmark.

### Order-period experiment was not retained

The notebook explores `order_period`, but the final run drops both `order_hour` and `order_period`. This should be re-tested systematically during future feature selection.

### Data quality assumptions

The 100 km distance threshold is domain-specific to the food-delivery context and should ideally be validated against the geographical coverage and business operating region.

---

# 16. Future Improvements

The next iteration could include:

## Model tuning

Use:

```text
GridSearchCV
RandomizedSearchCV
```

or a dedicated hyperparameter optimization framework to tune XGBoost.

Potential parameters include:

- `max_depth`
- `learning_rate`
- `n_estimators`
- `subsample`
- `colsample_bytree`
- Regularization parameters

## Cross-validation

Replace reliance on a single split with K-fold cross-validation to estimate model stability.

## Better temporal features

Re-test:

- `order_hour`
- `order_period`
- Day of week
- Month
- Weekend/weekday
- Pickup delay
- Time between order and pickup

The current notebook specifically documents that `order_period` was engineered but removed before the final modelling run, making it a natural candidate for controlled experimentation.

## Better geographic features

Potential extensions:

- Delivery-zone clustering
- City-specific distance distributions
- Restaurant density
- Distance bins
- Geospatial clusters

## Error analysis

Analyze:

```text
Prediction error by city
Prediction error by weather
Prediction error by traffic density
Prediction error by festival status
Prediction error by distance range
```

This can reveal where the model performs well and where operational data needs improvement.

## Explainability

Add:

- SHAP
- Partial Dependence Plots
- Permutation Importance

to make individual predictions and global model behaviour easier to explain to non-technical stakeholders.

---

# 17. Skills Demonstrated

This project demonstrates practical knowledge of:

### Data Science

- Data inspection
- Data cleaning
- Missing-value handling
- Exploratory Data Analysis
- Outlier detection
- Statistical comparison
- Feature selection

### Feature Engineering

- Datetime parsing
- Time-based feature extraction
- Haversine distance
- Domain-driven feature construction
- Categorical feature transformation

### Machine Learning

- Regression
- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting / XGBoost
- Ensemble learning
- Model comparison

### ML Engineering Fundamentals

- Train/test splitting
- Reproducibility
- Leakage prevention
- Scaling
- Feature-importance analysis
- Experiment documentation

### Python / Libraries

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- Jupyter

---

# 18. Interview Discussion Points

This project provides several strong topics for an ML/Data Science interview.

### Why use Haversine distance?

Because latitude and longitude are geographic coordinates; the distance between the restaurant and delivery location is a more meaningful operational feature than four raw coordinate columns.

### Why median imputation for numeric variables?

Median is more robust to skew and extreme values than mean, which is useful for variables such as age and ratings when the data contains noise.

### Why ordinal encoding for traffic?

Because:

```text
Low < Medium < High < Jam
```

has genuine semantic order.

### Why one-hot encode the other categorical features?

Weather, vehicle type, order type, city, and festival status do not have a meaningful numeric order.

### Why was scaling applied?

Scaling is important for Linear Regression and provides a consistent input representation across the compared models. Tree-based models generally do not require scaling.

### How was data leakage avoided?

The scaler was fitted only on `X_train`:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The test set therefore does not influence the learned scaling parameters.

### Why did the Decision Tree overfit?

An unrestricted tree can keep splitting until it nearly memorizes the training data. The recorded training R² of approximately 0.99998 versus test R² of approximately 0.667 demonstrates the generalization gap.

### Why did ensemble models perform better?

Random Forest reduces variance by aggregating multiple trees, while XGBoost builds trees sequentially to correct previous errors. Both approaches represent non-linear relationships more effectively than Linear Regression and generalize much better than the single unrestricted tree.

---

# 19. Final Conclusion

This project demonstrates a complete tabular machine-learning workflow for food delivery time prediction.

The key result is that **XGBoost achieved the best recorded performance with RMSE ≈ 3.99 minutes and R² ≈ 0.823**.

The experiment also highlights an important modelling lesson: simply increasing model complexity is not enough. The unrestricted Decision Tree nearly memorized the training data, while ensemble methods provided a much better balance between flexibility and generalization.

The strongest practical signals identified by the project include **distance, traffic, delivery-person attributes, vehicle condition, multiple deliveries, and festival context**.

---

## Author

**Krish Savaj**

Machine Learning / Data Science Project

---

## Repository Presentation Checklist

Before sharing the GitHub link with a recruiter:

- Keep the repository name professional and easy to understand.
- Rename the notebook to a clean filename.
- Remove personal Google Drive paths from the notebook.
- Add `README.md`.
- Add `requirements.txt`.
- Add `.gitignore`.
- Keep raw datasets out of GitHub when licensing/size is a concern.
- Add a short repository description on GitHub.
- Pin the best model/result in the README.
- Make sure the notebook opens and reads cleanly from top to bottom.
