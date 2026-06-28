"""
House Price Prediction - Training Pipeline
Trains a GradientBoosting model with feature engineering & saves artifacts.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Feature selection ──────────────────────────────────────────────────────
# Key features chosen by domain knowledge + correlation analysis
NUMERICAL_FEATURES = [
    "OverallQual", "GrLivArea", "GarageCars", "GarageArea",
    "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd",
    "YearBuilt", "YearRemodAdd", "MasVnrArea", "Fireplaces",
    "BsmtFinSF1", "LotFrontage", "WoodDeckSF", "OpenPorchSF",
    "2ndFlrSF", "HalfBath", "LotArea", "BsmtFullBath",
    "BsmtUnfSF", "BedroomAbvGr", "ScreenPorch", "PoolArea", "MoSold", "YrSold"
]

CATEGORICAL_FEATURES = [
    "MSZoning", "Street", "LotShape", "LandContour", "LotConfig",
    "Neighborhood", "BldgType", "HouseStyle", "RoofStyle", "Exterior1st",
    "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
    "BsmtExposure", "HeatingQC", "CentralAir", "KitchenQual",
    "Functional", "GarageType", "GarageFinish", "SaleType", "SaleCondition"
]

TARGET = "SalePrice"

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
y = np.log1p(df[TARGET])   # log-transform target for better distribution

# Fill numeric NaNs with median
num_imputer = SimpleImputer(strategy="median")
X[NUMERICAL_FEATURES] = num_imputer.fit_transform(X[NUMERICAL_FEATURES])

# Fill categorical NaNs with mode / "Unknown"
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].fillna("Unknown")

# Encode categoricals
label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Feature engineering
X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
X["RemodelAge"] = X["YrSold"] - X["YearRemodAdd"]
X["TotalBath"] = X["FullBath"] + 0.5 * X["HalfBath"] + X["BsmtFullBath"]
X["PorchArea"] = X["WoodDeckSF"] + X["OpenPorchSF"] + X["ScreenPorch"]
ENGINEERED = ["TotalSF", "HouseAge", "RemodelAge", "TotalBath", "PorchArea"]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[ALL_FEATURES])

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── 5. Model training ─────────────────────────────────────────────────────────
print("\nTraining GradientBoostingRegressor …")
model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

# Back-transform from log scale
y_pred_orig = np.expm1(y_pred)
y_test_orig = np.expm1(y_test)

rmse  = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae   = mean_absolute_error(y_test_orig, y_pred_orig)
r2    = r2_score(y_test_orig, y_pred_orig)
mape  = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

print(f"\n{'='*45}")
print(f"  R²  Score       : {r2:.4f}")
print(f"  RMSE            : ${rmse:,.0f}")
print(f"  MAE             : ${mae:,.0f}")
print(f"  MAPE            : {mape:.2f}%")
print(f"  CV R² (5-fold)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"{'='*45}")

metrics = {
    "r2": round(r2, 4),
    "rmse": round(rmse, 2),
    "mae": round(mae, 2),
    "mape": round(mape, 2),
    "cv_r2_mean": round(cv_scores.mean(), 4),
    "cv_r2_std": round(cv_scores.std(), 4),
    "train_size": len(X_train),
    "test_size": len(X_test),
}

# ── 7. Save artifacts ─────────────────────────────────────────────────────────
with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{MODEL_DIR}/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open(f"{MODEL_DIR}/num_imputer.pkl", "wb") as f:
    pickle.dump(num_imputer, f)

meta = {
    "numerical_features": NUMERICAL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "engineered_features": ENGINEERED,
    "all_features": ALL_FEATURES,
    "target": TARGET,
    "metrics": metrics,
}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nAll artifacts saved to: {MODEL_DIR}/")
print("Training complete ✓")
