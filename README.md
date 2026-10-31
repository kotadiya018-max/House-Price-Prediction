# 🏠 House Price Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

A machine learning project that predicts the **sale price of residential houses** from 80 property features such as size, quality, location, age and garage details.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Workflow](#-workflow)
6. [Complete Code with Line-by-Line Explanation](#-complete-code-with-line-by-line-explanation)
7. [How to Run](#-how-to-run)
8. [Results](#-results)
9. [Future Improvements](#-future-improvements)

---

## 📖 Project Overview

Buying or selling a house requires knowing its fair market value. This project builds a regression model that learns the relationship between a house's characteristics and its final sale price.

**Goal:** predict `SalePrice` (the target) for a given house.

**Key points**
- End-to-end ML pipeline: data loading, EDA, preprocessing, modeling, evaluation
- Handles missing values and mixed numeric/categorical features
- Compares multiple models using cross-validation
- Uses a log-transformed target to reduce the effect of skewed prices

---

## 📂 Dataset

| Property | Value |
|---|---|
| File | `train.csv` |
| Rows | 1,460 |
| Columns | 81 (80 features + 1 target) |
| Target | `SalePrice` |

**Example features**

| Feature | Meaning |
|---|---|
| `OverallQual` | Overall material and finish quality (1-10) |
| `GrLivArea` | Above-ground living area (sq ft) |
| `GarageCars` | Garage capacity (number of cars) |
| `TotalBsmtSF` | Total basement area (sq ft) |
| `YearBuilt` | Original construction year |
| `Neighborhood` | Physical location within the city |

The dataset contains numeric columns, categorical columns, and many missing values (for example `Alley`, `PoolQC`, `Fence`, `MiscFeature`, where a missing value usually means the feature does not exist).

---

## 🛠 Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" />
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" />
</p>

| Library | Purpose |
|---|---|
| **Pandas** | Loading and manipulating tabular data |
| **NumPy** | Numerical operations, log transform |
| **Matplotlib / Seaborn** | Data visualization |
| **Scikit-learn** | Preprocessing, models, cross-validation, metrics |
| **Jupyter Notebook** | Interactive development |

---

## 🗂 Project Structure

```
house-price-prediction/
│
├── main.ipynb        # Main notebook with the full pipeline
├── train.csv         # Training dataset
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

---

## 🔄 Workflow

```
Load Data
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Data Cleaning (missing values)
   │
   ▼
Feature Preprocessing (impute, encode)
   │
   ▼
Train / Test Split
   │
   ▼
Model Training (Ridge, Random Forest, Gradient Boosting)
   │
   ▼
Cross-Validation & Model Comparison
   │
   ▼
Evaluation (RMSE, R²) & Feature Importance
```

1. **Load data**: read `train.csv` with Pandas.
2. **EDA**: inspect shape, data types, missing values, target distribution and correlations.
3. **Cleaning**: fill missing values (median for numbers, `"None"` for categories).
4. **Preprocessing**: one-hot encode categorical columns inside a Scikit-learn pipeline.
5. **Split**: 80% training, 20% testing.
6. **Modeling**: train several regressors on the log of `SalePrice`.
7. **Evaluation**: compare models with cross-validation and measure the best on the test set.
8. **Interpretation**: plot the most important features.

---

## 💻 Complete Code with Line-by-Line Explanation

Every step below shows the code first, then explains **every line** of it. Run the steps in order, each in its own notebook cell.

---

### Step 1: Import libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

**Explanation**

| Line | What it does |
|---|---|
| `import numpy as np` | Loads NumPy for math on arrays (log, exp, square root). `np` is the short name. |
| `import pandas as pd` | Loads Pandas for working with tables (DataFrames). |
| `import matplotlib.pyplot as plt` | Loads Matplotlib's plotting module to draw charts. |
| `import seaborn as sns` | Loads Seaborn, which makes nicer statistical plots on top of Matplotlib. |
| `train_test_split` | Splits data into a training part and a testing part. |
| `cross_val_score` | Tests a model on several different splits to get a reliable score. |
| `ColumnTransformer` | Applies different preprocessing to different columns (numbers vs text). |
| `Pipeline` | Chains steps (preprocessing, then model) into one object. |
| `SimpleImputer` | Fills in missing values. |
| `OneHotEncoder` | Converts text categories into 0/1 columns. |
| `Ridge` | A linear regression model with regularization (a simple baseline). |
| `RandomForestRegressor` | An ensemble of many decision trees, averaged together. |
| `GradientBoostingRegressor` | Builds trees one after another, each fixing the previous one's errors. |
| `mean_squared_error` | Measures the average squared prediction error. |
| `r2_score` | Measures how much of the price variation the model explains (1.0 is perfect). |

---

### Step 2: Load and inspect the data

```python
df = pd.read_csv("train.csv")

print(df.shape)
df.info()
df.describe()
df.head()
```

**Explanation**

| Line | What it does |
|---|---|
| `df = pd.read_csv("train.csv")` | Reads the CSV file into a DataFrame called `df`. The file must be in the same folder as the notebook. |
| `print(df.shape)` | Prints `(rows, columns)`. Expected: `(1460, 81)`. |
| `df.info()` | Lists every column with its data type and how many values are not missing. |
| `df.describe()` | Shows count, mean, min, max and quartiles for each numeric column. |
| `df.head()` | Shows the first 5 rows so you can see what the data looks like. |

---

### Step 3: Explore the target (SalePrice)

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["SalePrice"], kde=True, ax=axes[0])
axes[0].set_title("SalePrice")

sns.histplot(np.log1p(df["SalePrice"]), kde=True, ax=axes[1])
axes[1].set_title("log(SalePrice)")

plt.show()
```

**Explanation**

| Line | What it does |
|---|---|
| `fig, axes = plt.subplots(1, 2, figsize=(12, 4))` | Creates one figure with 1 row and 2 side-by-side charts, 12 inches wide and 4 tall. `axes` holds the two chart areas. |
| `sns.histplot(df["SalePrice"], kde=True, ax=axes[0])` | Draws a histogram of prices in the left chart. `kde=True` adds a smooth curve over the bars. |
| `axes[0].set_title("SalePrice")` | Titles the left chart. |
| `np.log1p(df["SalePrice"])` | Computes `log(1 + price)` for every price. This compresses large values. |
| `sns.histplot(..., ax=axes[1])` | Draws the histogram of the log prices in the right chart. |
| `axes[1].set_title("log(SalePrice)")` | Titles the right chart. |
| `plt.show()` | Displays the figure. |

**Why:** raw prices are skewed to the right (a few very expensive houses). After the log, the shape is close to a bell curve, which most models handle better.

---

### Step 4: Check missing values

```python
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

missing.plot(kind="bar", figsize=(12, 4), title="Missing values per column")
plt.show()
```

**Explanation**

| Line | What it does |
|---|---|
| `df.isnull()` | Makes a table of True/False, True where a value is missing. |
| `.sum()` | Counts the True values in each column, giving missing count per column. |
| `missing[missing > 0]` | Keeps only columns that have at least one missing value. |
| `.sort_values(ascending=False)` | Sorts so the columns with the most missing values come first. |
| `print(missing)` | Prints the list. |
| `missing.plot(kind="bar", ...)` | Draws a bar chart of missing counts, 12x4 inches, with a title. |
| `plt.show()` | Displays the chart. |

---

### Step 5: Find which features relate to price

```python
corr = df.select_dtypes(include="number").corr()["SalePrice"].sort_values(ascending=False)
print(corr.head(10))

top_features = corr.index[:10]

plt.figure(figsize=(8, 6))
sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Top correlated features")
plt.show()
```

**Explanation**

| Line | What it does |
|---|---|
| `df.select_dtypes(include="number")` | Keeps only numeric columns (correlation needs numbers). |
| `.corr()` | Computes the correlation between every pair of numeric columns (-1 to +1). |
| `["SalePrice"]` | Takes only the column showing each feature's correlation with `SalePrice`. |
| `.sort_values(ascending=False)` | Strongest positive correlations first. `SalePrice` itself is first (1.0). |
| `print(corr.head(10))` | Prints the top 10. |
| `top_features = corr.index[:10]` | Saves the names of those top 10 features. |
| `plt.figure(figsize=(8, 6))` | Creates a blank 8x6 figure. |
| `sns.heatmap(df[top_features].corr(), ...)` | Draws a colored grid of correlations among those features. `annot=True` writes the numbers, `fmt=".2f"` shows 2 decimals, `cmap="coolwarm"` uses blue-to-red colors. |
| `plt.title(...)` and `plt.show()` | Adds a title and displays the plot. |

---

### Step 6: Define features (X) and target (y)

```python
X = df.drop(columns=["Id", "SalePrice"])
y = np.log1p(df["SalePrice"])

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

print("Numeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))
```

**Explanation**

| Line | What it does |
|---|---|
| `X = df.drop(columns=["Id", "SalePrice"])` | `X` is every column except `Id` (just a row number, no predictive value) and `SalePrice` (what we want to predict). |
| `y = np.log1p(df["SalePrice"])` | `y` is the target, log-transformed. |
| `X.select_dtypes(include="number").columns` | Names of all numeric feature columns. |
| `X.select_dtypes(exclude="number").columns` | Names of all text (categorical) feature columns. |
| `print(...)` lines | Show how many columns are in each group. |

---

### Step 7: Build the preprocessing pipeline

```python
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])
```

**Explanation**

| Line | What it does |
|---|---|
| `numeric_pipe = Pipeline([...])` | A mini-pipeline for numeric columns. |
| `("imputer", SimpleImputer(strategy="median"))` | Step named `imputer`: replaces each missing number with that column's median. The median is not distorted by extreme values. |
| `categorical_pipe = Pipeline([...])` | A mini-pipeline for text columns. |
| `SimpleImputer(strategy="constant", fill_value="None")` | Replaces missing text with the word `"None"`. In this dataset a missing value usually means the house has no such feature (no pool, no alley). |
| `OneHotEncoder(handle_unknown="ignore")` | Turns each category into its own 0/1 column. `handle_unknown="ignore"` prevents errors if a new category appears at prediction time. |
| `preprocessor = ColumnTransformer([...])` | Combines both mini-pipelines. |
| `("num", numeric_pipe, numeric_cols)` | Apply `numeric_pipe` to the numeric columns. |
| `("cat", categorical_pipe, categorical_cols)` | Apply `categorical_pipe` to the text columns. |

---

### Step 8: Split into training and testing sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
```

**Explanation**

| Part | What it does |
|---|---|
| `train_test_split(X, y, ...)` | Randomly divides the data into two groups. |
| `test_size=0.2` | 20% goes to testing (292 rows), 80% to training (1168 rows). |
| `random_state=42` | Fixes the randomness so you get the same split every run. |
| `X_train, X_test, y_train, y_test` | The four results: training features, testing features, training target, testing target. |
| `print(...)` lines | Confirm the sizes of both sets. |

The model learns from the training set only. The test set is kept hidden to check how well it works on unseen houses.

---

### Step 9: Compare models using cross-validation

```python
models = {
    "Ridge": Ridge(alpha=10),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
    ),
}

results = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    scores = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"
    )
    results[name] = -scores.mean()
    print(f"{name:20s} CV RMSE (log): {-scores.mean():.4f}")
```

**Explanation**

| Line | What it does |
|---|---|
| `models = {...}` | A dictionary of the three models to compare, each with its settings. |
| `Ridge(alpha=10)` | Linear model. `alpha` controls regularization strength (higher means simpler model). |
| `RandomForestRegressor(n_estimators=300, ...)` | Builds 300 trees. `random_state=42` makes it repeatable. `n_jobs=-1` uses all CPU cores. |
| `GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, ...)` | Builds 500 small trees (depth 3), each correcting the previous. A small learning rate means careful, gradual learning. |
| `results = {}` | An empty dictionary to store each model's score. |
| `for name, model in models.items():` | Loops through the models one at a time. |
| `pipe = Pipeline([("prep", preprocessor), ("model", model)])` | Joins preprocessing and the model so both run together. |
| `cross_val_score(pipe, X_train, y_train, cv=5, ...)` | Splits the training data into 5 parts. Trains on 4, tests on 1, and repeats 5 times. |
| `scoring="neg_root_mean_squared_error"` | Uses RMSE as the metric. Scikit-learn returns it as a negative number (higher is better internally). |
| `results[name] = -scores.mean()` | Averages the 5 scores and flips the sign back to positive RMSE. |
| `print(f"{name:20s} ... {...:.4f}")` | Prints the model name (padded to 20 characters) and its RMSE with 4 decimals. |

**Why use the pipeline inside cross-validation:** preprocessing is learned only from the training folds each time. This prevents data leakage, which would make scores look better than they really are.

**Reading the score:** lower RMSE is better. Because the target is log-transformed, an RMSE of 0.13 means predictions are off by roughly 13%.

---

### Step 10: Train the best model and evaluate on the test set

```python
best_name = min(results, key=results.get)
print("Best model:", best_name)

best_pipe = Pipeline([("prep", preprocessor), ("model", models[best_name])])
best_pipe.fit(X_train, y_train)

pred = best_pipe.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"Test RMSE (log): {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
```

**Explanation**

| Line | What it does |
|---|---|
| `best_name = min(results, key=results.get)` | Finds the model name with the lowest RMSE in `results`. |
| `best_pipe = Pipeline([...models[best_name]])` | Builds a pipeline using the winning model. |
| `best_pipe.fit(X_train, y_train)` | Trains preprocessing and the model on the full training set. |
| `pred = best_pipe.predict(X_test)` | Predicts (log) prices for the unseen test houses. |
| `mean_squared_error(y_test, pred)` | Average of squared differences between real and predicted values. |
| `np.sqrt(...)` | Square root gives RMSE, back in the same units as the target. |
| `r2_score(y_test, pred)` | R² score: closer to 1.0 means the model explains more of the price variation. |
| `print(...)` lines | Show the final test results. Copy these into the Results table. |

---

### Step 11: Plot actual vs predicted prices

```python
actual = np.expm1(y_test)
predicted = np.expm1(pred)

plt.figure(figsize=(6, 6))
plt.scatter(actual, predicted, alpha=0.5)

lims = [actual.min(), actual.max()]
plt.plot(lims, lims, "r--")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()
```

**Explanation**

| Line | What it does |
|---|---|
| `actual = np.expm1(y_test)` | Reverses the log transform (`expm1` is the opposite of `log1p`) to get real dollar prices. |
| `predicted = np.expm1(pred)` | Same for the predictions. |
| `plt.figure(figsize=(6, 6))` | Creates a square figure. |
| `plt.scatter(actual, predicted, alpha=0.5)` | Plots one dot per test house: actual price on x, predicted on y. `alpha=0.5` makes dots semi-transparent so overlaps are visible. |
| `lims = [actual.min(), actual.max()]` | The lowest and highest actual prices, used as the line's start and end. |
| `plt.plot(lims, lims, "r--")` | Draws a red dashed diagonal line where predicted equals actual. |
| `plt.xlabel / ylabel / title` | Label the axes and the chart. |
| `plt.show()` | Displays the plot. |

**How to read it:** the closer the dots are to the red line, the better the predictions.

---

### Step 12: See which features matter most

```python
if best_name != "Ridge":
    feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
    importances = best_pipe.named_steps["model"].feature_importances_

    imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

    imp.sort_values().plot(kind="barh", figsize=(8, 6), title="Top 15 Features")
    plt.show()
```

**Explanation**

| Line | What it does |
|---|---|
| `if best_name != "Ridge":` | Only runs for tree models, because Ridge has no `feature_importances_`. |
| `best_pipe.named_steps["prep"]` | Gets the preprocessing step from the pipeline by its name. |
| `.get_feature_names_out()` | Returns the names of all columns after encoding (for example `num__GrLivArea`, `cat__Neighborhood_NridgHt`). |
| `best_pipe.named_steps["model"].feature_importances_` | Gets how much each column contributed to the model's decisions. |
| `pd.Series(importances, index=feature_names)` | Pairs each importance value with its feature name. |
| `.sort_values(ascending=False).head(15)` | Keeps the 15 most important features. |
| `imp.sort_values().plot(kind="barh", ...)` | Draws a horizontal bar chart. Sorted ascending so the biggest bar appears at the top. |
| `plt.show()` | Displays the chart. |

---

### Step 13 (optional): Predict a price for a single house

```python
sample = X_test.iloc[[0]]
price = np.expm1(best_pipe.predict(sample))[0]

print(f"Predicted price: ${price:,.0f}")
print(f"Actual price:    ${np.expm1(y_test.iloc[0]):,.0f}")
```

**Explanation**

| Line | What it does |
|---|---|
| `X_test.iloc[[0]]` | Picks the first test house. The double brackets keep it as a one-row table, which is what the model expects. |
| `best_pipe.predict(sample)` | Predicts its log price. |
| `np.expm1(...)[0]` | Converts back to dollars and takes the single value out of the array. |
| `f"${price:,.0f}"` | Formats the number with commas and no decimals, for example `$208,500`. |
| Last `print` | Shows the real price for comparison. |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file with:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

Then run:

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter notebook main.ipynb
```

Make sure `train.csv` is in the same folder as `main.ipynb`, then choose **Kernel → Restart & Run All**.

---

## 📊 Results

| Metric | Value |
|---|---|
| Test RMSE (log scale) | _fill in after running_ |
| Test R² | _fill in after running_ |

Add screenshots of your plots here:

```
![Target distribution](images/target_distribution.png)
![Actual vs Predicted](images/actual_vs_predicted.png)
![Feature importance](images/feature_importance.png)
```

---

## 🔮 Future Improvements

- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- Feature engineering (for example `TotalSF`, `HouseAge`, `TotalBathrooms`)
- Outlier removal for very large houses
- Try XGBoost, LightGBM or CatBoost
- Stack or blend multiple models
- Deploy as a web app with Streamlit or Flask

---

## 🤝 Contributing

Contributions are welcome. Fork the repository, create a branch, and open a pull request.

## 📜 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
GitHub: [@your-username](https://github.com/your-username)
