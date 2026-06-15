from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
import torch
import pickle
import os
import json
from xgboost import XGBRegressor

from src.models.hybrid_model import HybridModel, StandaloneLSTM
from src.api.schemas import ForecastOverrideRequest, ForecastOverrideResponse

router = APIRouter()

# Global Cache for Models & Metadata
MODELS_CACHE = {}
METRICS_CACHE = None

def load_sector_assets(sector: str):
    if sector in MODELS_CACHE:
        return MODELS_CACHE[sector]
        
    meta_path = f"models/saved/{sector}_meta.pkl"
    lstm_path = f"models/saved/{sector}_lstm.pth"
    hybrid_path = f"models/saved/{sector}_hybrid.pth"
    xgb_path = f"models/saved/{sector}_xgb.json"
    
    if not (os.path.exists(meta_path) and os.path.exists(lstm_path) and os.path.exists(hybrid_path) and os.path.exists(xgb_path)):
        raise HTTPException(status_code=500, detail=f"Models for sector '{sector}' are not trained yet. Run training first.")
        
    # Load metadata
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        
    # Load Standalone LSTM
    lstm_model = StandaloneLSTM(input_dim=1, hidden_dim=64)
    lstm_model.load_state_dict(torch.load(lstm_path))
    lstm_model.eval()
    
    # Load Hybrid Model
    hybrid_model = HybridModel(input_dim=10, cnn_channels=64, lstm_hidden=64)
    hybrid_model.load_state_dict(torch.load(hybrid_path))
    hybrid_model.eval()
    
    # Load XGBoost
    xgb_model = XGBRegressor()
    xgb_model.load_model(xgb_path)
    
    # Load and parse CSV dataframe
    csv_path = f"data/processed/final_{sector}.csv"
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Slice to match training size
    if len(df) > 25000:
        df = df.iloc[-25000:].copy()
        
    assets = {
        "meta": meta,
        "lstm": lstm_model,
        "hybrid": hybrid_model,
        "xgb": xgb_model,
        "df": df
    }
    MODELS_CACHE[sector] = assets
    return assets

@router.get("/api/metrics")
def get_metrics():
    global METRICS_CACHE
    if METRICS_CACHE is not None:
        return METRICS_CACHE
    metrics_path = "models/results/global_metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Global metrics not found. Run model evaluation first.")
    with open(metrics_path, "r") as f:
        METRICS_CACHE = json.load(f)
    return METRICS_CACHE

@router.get("/api/forecast-data/{sector}")
def get_forecast_data(sector: str):
    if sector not in ["residential", "commercial", "industrial"]:
        raise HTTPException(status_code=400, detail="Invalid sector name.")
        
    # Load assets
    assets = load_sector_assets(sector)
    meta = assets["meta"]
    target_col = meta["target_col"]
    split_idx = meta["split_idx"]
    target_scaler = meta["target_scaler"]
    res_scaler = meta["res_scaler"]
    weather_scaler = meta["weather_scaler"]
    seasonal_baseline = meta["seasonal_baseline"]
    
    df = assets["df"]
    
    # Extract test dataframe (take last 150 hours to display)
    test_df = df.iloc[split_idx:].copy()
    show_hours = 150
    test_df_sliced = test_df.iloc[24:24+show_hours]
    
    # Dates & Actuals
    dates = test_df_sliced.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
    actuals = test_df_sliced[target_col].values.tolist()
    
    # 1. Seasonal Baseline
    seasonal_preds = seasonal_baseline.predict_season(test_df_sliced.index).tolist()
    
    # 2. XGBoost
    feature_cols = meta["feature_cols"]
    X_xgb = test_df_sliced[feature_cols]
    xgb_preds = assets["xgb"].predict(X_xgb).tolist()
    
    # 3. LSTM
    # We need the 24 hour history for each of the sliced hours
    test_scaled_target = target_scaler.transform(test_df[[target_col]])
    X_lstm = []
    # Aligned with the slice: 24 to 24+show_hours
    for i in range(24, 24+show_hours):
        X_lstm.append(test_scaled_target[i-24:i])
    X_lstm = np.array(X_lstm)
    
    with torch.no_grad():
        tensor_x = torch.tensor(X_lstm, dtype=torch.float32)
        lstm_scaled_out = assets["lstm"](tensor_x).numpy()
    lstm_preds = target_scaler.inverse_transform(lstm_scaled_out).flatten().tolist()
    
    # 4. Hybrid
    # Compute residuals
    test_seasonal_preds_full = seasonal_baseline.predict_season(test_df.index)
    test_residuals_full = test_df[target_col].values - test_seasonal_preds_full
    test_scaled_res = res_scaler.transform(test_residuals_full.reshape(-1, 1))
    test_scaled_weather = weather_scaler.transform(test_df[['air_temperature', 'wind_speed', 'cloud_coverage']])
    calendar_features = test_df[['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']].values
    
    test_features = np.hstack([test_scaled_res, test_scaled_weather, calendar_features])
    
    X_hybrid = []
    for i in range(24, 24+show_hours):
        X_hybrid.append(test_features[i-24:i])
    X_hybrid = np.array(X_hybrid)
    
    with torch.no_grad():
        tensor_x_hybrid = torch.tensor(X_hybrid, dtype=torch.float32)
        hybrid_res_scaled = assets["hybrid"](tensor_x_hybrid).numpy()
    hybrid_res_preds = res_scaler.inverse_transform(hybrid_res_scaled).flatten()
    hybrid_preds = (np.array(seasonal_preds) + hybrid_res_preds).tolist()
    
    return {
        "sector": sector,
        "timestamps": dates,
        "actuals": actuals,
        "seasonal": [round(x, 3) for x in seasonal_preds],
        "xgb": [round(x, 3) for x in xgb_preds],
        "lstm": [round(x, 3) for x in lstm_preds],
        "hybrid": [round(x, 3) for x in hybrid_preds]
    }

@router.post("/api/predict-interactive", response_model=ForecastOverrideResponse)
def predict_interactive(req: ForecastOverrideRequest):
    sector = req.sector
    if sector not in ["residential", "commercial", "industrial"]:
        raise HTTPException(status_code=400, detail="Invalid sector name.")
        
    assets = load_sector_assets(sector)
    meta = assets["meta"]
    target_col = meta["target_col"]
    target_scaler = meta["target_scaler"]
    res_scaler = meta["res_scaler"]
    weather_scaler = meta["weather_scaler"]
    seasonal_baseline = meta["seasonal_baseline"]
    
    df = assets["df"]
    
    recent_df = df.iloc[-24:].copy()
    
    # Target Hour = 1 hour after the end of dataset
    target_dt = recent_df.index[-1] + pd.Timedelta(hours=1)
    
    # 1. Seasonal Baseline Prediction
    S_t = seasonal_baseline.predict_season([target_dt])[0]
    
    # 2. LSTM Prediction
    loads = recent_df[target_col].values
    loads_scaled = target_scaler.transform(loads.reshape(-1, 1))
    X_lstm = loads_scaled.reshape(1, 24, 1)
    with torch.no_grad():
        tensor_x = torch.tensor(X_lstm, dtype=torch.float32)
        lstm_out_scaled = assets["lstm"](tensor_x).numpy()
    pred_lstm = float(target_scaler.inverse_transform(lstm_out_scaled)[0][0])
    
    # 3. XGBoost Prediction
    # Create feature record at target time t
    # Replace the weather variables with user's overrides
    lag_1 = recent_df[target_col].values[-1]
    lag_2 = recent_df[target_col].values[-2]
    lag_24 = recent_df[target_col].values[-24]
    rolling_mean_24 = recent_df[target_col].mean()
    rolling_std_24 = recent_df[target_col].std()
    
    hour = target_dt.hour
    dayofweek = target_dt.dayofweek
    month = target_dt.month
    
    xgb_features = {
        'lag_1': [lag_1],
        'lag_2': [lag_2],
        'lag_24': [lag_24],
        'rolling_mean_24': [rolling_mean_24],
        'rolling_std_24': [rolling_std_24],
        'air_temperature': [req.air_temperature],
        'wind_speed': [req.wind_speed],
        'cloud_coverage': [req.cloud_coverage],
        'hour_sin': [np.sin(2 * np.pi * hour / 24.0)],
        'hour_cos': [np.cos(2 * np.pi * hour / 24.0)],
        'dayofweek_sin': [np.sin(2 * np.pi * dayofweek / 7.0)],
        'dayofweek_cos': [np.cos(2 * np.pi * dayofweek / 7.0)],
        'month_sin': [np.sin(2 * np.pi * month / 12.0)],
        'month_cos': [np.cos(2 * np.pi * month / 12.0)]
    }
    X_xgb = pd.DataFrame(xgb_features)
    pred_xgb = float(assets["xgb"].predict(X_xgb)[0])
    
    # 4. Hybrid Prediction
    # We substitute the weather of the last timestep (t-1) with the user overrides
    # to show instant sensitivity of the neural net to weather changes
    recent_df.loc[recent_df.index[-1], 'air_temperature'] = req.air_temperature
    recent_df.loc[recent_df.index[-1], 'wind_speed'] = req.wind_speed
    recent_df.loc[recent_df.index[-1], 'cloud_coverage'] = req.cloud_coverage
    
    # Re-calculate residuals and features
    res = recent_df[target_col].values - seasonal_baseline.predict_season(recent_df.index)
    res_scaled = res_scaler.transform(res.reshape(-1, 1))
    weather_scaled = weather_scaler.transform(recent_df[['air_temperature', 'wind_speed', 'cloud_coverage']])
    calendar_features = recent_df[['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']].values
    
    features = np.hstack([res_scaled, weather_scaled, calendar_features])
    X_hybrid = features.reshape(1, 24, 10)
    
    with torch.no_grad():
        tensor_x_hybrid = torch.tensor(X_hybrid, dtype=torch.float32)
        hybrid_res_scaled = assets["hybrid"](tensor_x_hybrid).numpy()
    pred_res = float(res_scaler.inverse_transform(hybrid_res_scaled)[0][0])
    pred_hybrid = S_t + pred_res
    
    return ForecastOverrideResponse(
        sector=sector,
        air_temperature=req.air_temperature,
        wind_speed=req.wind_speed,
        cloud_coverage=req.cloud_coverage,
        baseline_prediction=round(float(S_t), 3),
        xgb_prediction=round(pred_xgb, 3),
        lstm_prediction=round(pred_lstm, 3),
        hybrid_prediction=round(pred_hybrid, 3)
    )

def preload_all_assets():
    global METRICS_CACHE
    # Preload global metrics
    metrics_path = "models/results/global_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            METRICS_CACHE = json.load(f)
    
    # Preload sector assets (models, metadata, dataframes)
    for sector in ["residential", "commercial", "industrial"]:
        try:
            load_sector_assets(sector)
            print(f"Preloaded assets for sector '{sector}' successfully.")
        except Exception as e:
            print(f"Failed to preload assets for sector '{sector}': {e}")

