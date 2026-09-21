"""
Trains and compares a baseline (Linear Regression) against XGBoost for
UK house price prediction, then saves the best model + preprocessing
pipeline to models/model.joblib for the API to load.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

DATA_PATH = "data/uk_housing.csv"
MODEL_PATH = "models/model.joblib"

df = pd.read_csv(DATA_PATH)

# --- Feature engineering -----------------------------------------------
# 1) property age is more meaningful to a model than raw year_built
df["property_age"] = 2026 - df["year_built"]
df = df.drop(columns=["year_built"])

# 2) price per sqft-adjacent feature: bedrooms-to-sqft ratio flags
#    unusually cramped/spacious properties for their bedroom count
df["sqft_per_bedroom"] = df["sqft"] / df["bedrooms"]

TARGET = "price"
categorical_features = ["region", "property_type"]
numeric_features = [
    "bedrooms", "bathrooms", "sqft", "dist_to_station_km",
    "has_garden", "crime_score", "property_age", "sqft_per_bedroom",
]

X = df[categorical_features + numeric_features]
y = df[TARGET]

# 80/20 train/test split, then take a validation slice out of train for
# model comparison so the test set stays untouched until final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])


def evaluate(name, pipe):
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    print(f"{name:20s}  MAE=£{mae:,.0f}   RMSE=£{rmse:,.0f}   R2={r2:.3f}")
    return mae, rmse, r2


print("--- Model comparison (validation set) ---")

baseline = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
baseline_mae, _, baseline_r2 = evaluate("LinearRegression", baseline)

xgb = Pipeline([("prep", preprocessor), ("model", XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42,
))])
xgb_mae, _, xgb_r2 = evaluate("XGBoost", xgb)

# --- Pick the winner and refit on ALL training data (train+val) --------
best_name, best_pipe = ("XGBoost", xgb) if xgb_mae < baseline_mae else ("LinearRegression", baseline)
print(f"\nBest model: {best_name}")

best_pipe.fit(X_train, y_train)
test_preds = best_pipe.predict(X_test)
test_mae = mean_absolute_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)
print(f"Final held-out TEST performance -> MAE=£{test_mae:,.0f}   R2={test_r2:.3f}")

joblib.dump({
    "pipeline": best_pipe,
    "model_name": best_name,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "test_mae": test_mae,
    "test_r2": test_r2,
}, MODEL_PATH)
print(f"\nSaved model to {MODEL_PATH}")
