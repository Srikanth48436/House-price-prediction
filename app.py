"""
House Price Prediction – Flask Application
REST API + HTML frontend
"""

from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")

with open(f"{MODEL_DIR}/model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)
with open(f"{MODEL_DIR}/label_encoders.pkl", "rb") as f:
    LABEL_ENCODERS = pickle.load(f)
with open(f"{MODEL_DIR}/num_imputer.pkl", "rb") as f:
    NUM_IMPUTER = pickle.load(f)
with open(f"{MODEL_DIR}/metadata.json", "r") as f:
    META = json.load(f)

NUM_FEATURES = META["numerical_features"]
CAT_FEATURES = META["categorical_features"]
ALL_FEATURES  = META["all_features"]

# ── Helper: build feature vector ──────────────────────────────────────────────
def preprocess(data: dict) -> np.ndarray:
    import pandas as pd

    # Build a single-row DataFrame
    row = {}

    # Numerical defaults (use median from a rough reference or 0)
    defaults_num = {
        "OverallQual": 5, "GrLivArea": 1500, "GarageCars": 2,
        "GarageArea": 480, "TotalBsmtSF": 1000, "1stFlrSF": 1000,
        "FullBath": 2, "TotRmsAbvGrd": 7, "YearBuilt": 1980,
        "YearRemodAdd": 2000, "MasVnrArea": 0, "Fireplaces": 1,
        "BsmtFinSF1": 400, "LotFrontage": 70, "WoodDeckSF": 0,
        "OpenPorchSF": 40, "2ndFlrSF": 0, "HalfBath": 0,
        "LotArea": 9000, "BsmtFullBath": 0, "BsmtUnfSF": 400,
        "BedroomAbvGr": 3, "ScreenPorch": 0, "PoolArea": 0,
        "MoSold": 6, "YrSold": 2008,
    }
    for col in NUM_FEATURES:
        row[col] = float(data.get(col, defaults_num.get(col, 0)))

    # Categorical defaults
    defaults_cat = {
        "MSZoning": "RL", "Street": "Pave", "LotShape": "Reg",
        "LandContour": "Lvl", "LotConfig": "Inside", "Neighborhood": "NAmes",
        "BldgType": "1Fam", "HouseStyle": "2Story", "RoofStyle": "Gable",
        "Exterior1st": "VinylSd", "ExterQual": "TA", "ExterCond": "TA",
        "Foundation": "PConc", "BsmtQual": "TA", "BsmtCond": "TA",
        "BsmtExposure": "No", "HeatingQC": "Ex", "CentralAir": "Y",
        "KitchenQual": "TA", "Functional": "Typ", "GarageType": "Attchd",
        "GarageFinish": "RFn", "SaleType": "WD", "SaleCondition": "Normal",
    }
    for col in CAT_FEATURES:
        val = str(data.get(col, defaults_cat.get(col, "Unknown")))
        le = LABEL_ENCODERS[col]
        if val not in le.classes_:
            val = le.classes_[0]
        row[col] = le.transform([val])[0]

    df = pd.DataFrame([row])

    # Apply num imputer
    df[NUM_FEATURES] = NUM_IMPUTER.transform(df[NUM_FEATURES])

    # Feature engineering
    df["TotalSF"]    = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"]   = df["YrSold"] - df["YearBuilt"]
    df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalBath"]  = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"]
    df["PorchArea"]  = df["WoodDeckSF"] + df["OpenPorchSF"] + df["ScreenPorch"]

    X = SCALER.transform(df[ALL_FEATURES])
    return X


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", metrics=META["metrics"])


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body: JSON with house features
    Returns: { "predicted_price": float, "confidence_range": [low, high] }
    """
    try:
        data = request.get_json(force=True)
        X = preprocess(data)
        log_price = MODEL.predict(X)[0]
        price = float(np.expm1(log_price))

        # Approx ±10% confidence band (based on MAPE)
        mape = META["metrics"]["mape"] / 100
        return jsonify({
            "success": True,
            "predicted_price": round(price),
            "confidence_range": [
                round(price * (1 - mape)),
                round(price * (1 + mape)),
            ],
            "model_r2": META["metrics"]["r2"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/metrics", methods=["GET"])
def metrics():
    """GET /api/metrics – returns model performance metrics"""
    return jsonify(META["metrics"])


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "GradientBoostingRegressor"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
