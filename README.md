# 🏠 House Price Prediction System

An end-to-end Machine Learning web application for predicting house prices using the Ames Housing dataset.

## Tech Stack
- **ML**: Python · Scikit-learn (GradientBoostingRegressor)
- **Backend**: Flask REST API
- **Frontend**: HTML5 · CSS3 · Vanilla JS

## Model Performance
| Metric | Value |
|---|---|
| R² Score | 0.8933 |
| RMSE | $28,610 |
| MAE | $17,080 |
| MAPE | 10.14% |
| CV R² (5-fold) | 0.8935 ± 0.0124 |

## Project Structure
```
house_price_app/
├── data.csv              ← Ames Housing dataset
├── train_model.py        ← ML training pipeline
├── app.py                ← Flask application
├── requirements.txt
├── model/
│   ├── model.pkl         ← Trained GBR model
│   ├── scaler.pkl        ← StandardScaler
│   ├── label_encoders.pkl
│   ├── num_imputer.pkl
│   └── metadata.json     ← Features + metrics
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (only needed once, artifacts already included)
python train_model.py

# 3. Start the Flask server
python app.py
# Open http://localhost:5000
```

## REST API

### `POST /api/predict`
Predicts house price from JSON feature payload.

**Request body** (all fields optional, sensible defaults apply):
```json
{
  "OverallQual": 7,
  "GrLivArea": 1800,
  "GarageCars": 2,
  "YearBuilt": 2000,
  "Neighborhood": "CollgCr",
  ...
}
```
**Response**:
```json
{
  "success": true,
  "predicted_price": 215430,
  "confidence_range": [193887, 236973],
  "model_r2": 0.8933
}
```

### `GET /api/metrics`
Returns model evaluation metrics.

### `GET /api/health`
Health check.

## Feature Engineering
- **TotalSF** = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
- **HouseAge** = YrSold − YearBuilt
- **RemodelAge** = YrSold − YearRemodAdd
- **TotalBath** = FullBath + 0.5×HalfBath + BsmtFullBath
- **PorchArea** = WoodDeckSF + OpenPorchSF + ScreenPorch

## ML Pipeline
1. Load & explore Ames Housing data (1,460 rows × 81 cols)
2. Select 26 numerical + 24 categorical features
3. Impute missing values (median / "Unknown")
4. Label-encode categoricals
5. Engineer 5 derived features
6. StandardScaler normalization
7. Train GradientBoostingRegressor (500 trees, lr=0.05)
8. Log-transform target for better distribution
9. Evaluate with RMSE, MAE, MAPE, 5-fold CV R²
10. Serialize model + preprocessors with pickle
