import pandas as pd
import numpy as np
import pickle
import os
import torch
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.models.hybrid_model import HybridModel, StandaloneLSTM

def calculate_mape(y_true, y_pred):
    # Avoid division by zero
    y_true = np.clip(y_true, 1e-5, None)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_sector(csv_path, sector_name):
    print(f"\n======================================")
    print(f"Evaluating models for {sector_name.upper()}...")
    print(f"======================================")
    
    # Load metadata
    meta_path = f"models/saved/{sector_name}_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        
    target_col = meta["target_col"]
    feature_cols = meta["feature_cols"]
    split_idx = meta["split_idx"]
    target_scaler = meta["target_scaler"]
    res_scaler = meta["res_scaler"]
    weather_scaler = meta["weather_scaler"]
    seasonal_baseline = meta["seasonal_baseline"]
    
    # Load dataset
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Slice to match training size
    if len(df) > 25000:
        df = df.iloc[-25000:].copy()
        
    # Extract test dataframe
    test_df = df.iloc[split_idx:].copy()
    test_dates = test_df.index[24:]
    actuals = test_df[target_col].values[24:]
    
    # ----------------------------------------------------
    # 1. Seasonal Baseline Prediction
    # ----------------------------------------------------
    print("Computing Seasonal Baseline predictions...")
    seasonal_preds = seasonal_baseline.predict_season(test_dates)

    # ----------------------------------------------------
    # 2. XGBoost Prediction
    # ----------------------------------------------------
    print("Computing XGBoost predictions...")
    xgb_model = XGBRegressor()
    xgb_model.load_model(f"models/saved/{sector_name}_xgb.json")
    
    X_test_xgb = test_df[feature_cols].iloc[24:]
    xgb_preds = xgb_model.predict(X_test_xgb)

    # ----------------------------------------------------
    # 3. Standalone LSTM Prediction
    # ----------------------------------------------------
    print("Computing Standalone LSTM predictions...")
    lstm_model = StandaloneLSTM(input_dim=1, hidden_dim=64)
    lstm_model.load_state_dict(torch.load(f"models/saved/{sector_name}_lstm.pth"))
    lstm_model.eval()
    
    # Prepare sequence input
    test_scaled_target = target_scaler.transform(test_df[[target_col]])
    X_lstm = []
    for i in range(24, len(test_scaled_target)):
        X_lstm.append(test_scaled_target[i-24:i])
    X_lstm = np.array(X_lstm)
    
    with torch.no_grad():
        tensor_x = torch.tensor(X_lstm, dtype=torch.float32)
        lstm_out_scaled = lstm_model(tensor_x).numpy()
    
    lstm_preds = target_scaler.inverse_transform(lstm_out_scaled).flatten()

    # ----------------------------------------------------
    # 4. Hybrid CNN-BiLSTM-Attention Prediction
    # ----------------------------------------------------
    print("Computing Hybrid model predictions...")
    # Calculate actual test residuals
    test_seasonal_preds_full = seasonal_baseline.predict_season(test_df.index)
    test_residuals_full = test_df[target_col].values - test_seasonal_preds_full
    
    # Scale residuals
    test_scaled_res = res_scaler.transform(test_residuals_full.reshape(-1, 1))
    
    # Scale weather
    test_scaled_weather = weather_scaler.transform(test_df[['air_temperature', 'wind_speed', 'cloud_coverage']])
    
    # Calendar features
    calendar_features = test_df[['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']].values
    
    # Combine features
    test_features = np.hstack([
        test_scaled_res,
        test_scaled_weather,
        calendar_features
    ])
    
    X_hybrid = []
    for i in range(24, len(test_features)):
        X_hybrid.append(test_features[i-24:i])
    X_hybrid = np.array(X_hybrid)
    
    hybrid_model = HybridModel(input_dim=X_hybrid.shape[2], cnn_channels=64, lstm_hidden=64)
    hybrid_model.load_state_dict(torch.load(f"models/saved/{sector_name}_hybrid.pth"))
    hybrid_model.eval()
    
    with torch.no_grad():
        tensor_x_hybrid = torch.tensor(X_hybrid, dtype=torch.float32)
        hybrid_res_scaled = hybrid_model(tensor_x_hybrid).numpy()
        
    hybrid_res_preds = res_scaler.inverse_transform(hybrid_res_scaled).flatten()
    
    # Hybrid prediction = Seasonal Baseline + Predicted Residual
    hybrid_preds = seasonal_preds + hybrid_res_preds

    # ----------------------------------------------------
    # Metric Calculation
    # ----------------------------------------------------
    metrics = {}
    models = {
        "Seasonal_Baseline": seasonal_preds,
        "XGBoost": xgb_preds,
        "LSTM": lstm_preds,
        "Hybrid_Model": hybrid_preds
    }
    
    for name, preds in models.items():
        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mape = calculate_mape(actuals, preds)
        metrics[name] = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "MAPE": round(float(mape), 4)
        }
        print(f"Model: {name:<20} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
        
    # Save metrics JSON
    os.makedirs("models/results", exist_ok=True)
    with open(f"models/results/{sector_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # ----------------------------------------------------
    # Generate Visualizations (First 150 hours of test set)
    # ----------------------------------------------------
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plot Actual
    show_hours = 150
    ax.plot(test_dates[:show_hours], actuals[:show_hours], label="Actual Load", color="#ffffff", linewidth=2.5, alpha=0.9)
    
    # Plot predictions
    ax.plot(test_dates[:show_hours], seasonal_preds[:show_hours], label="Seasonal Baseline", color="#95a5a6", linestyle="--", alpha=0.7)
    ax.plot(test_dates[:show_hours], lstm_preds[:show_hours], label="LSTM Baseline", color="#e74c3c", alpha=0.8)
    ax.plot(test_dates[:show_hours], xgb_preds[:show_hours], label="XGBoost", color="#3498db", alpha=0.8)
    ax.plot(test_dates[:show_hours], hybrid_preds[:show_hours], label="Hybrid CNN-BiLSTM-Attention", color="#1abc9c", linewidth=2.0)
    
    ax.set_title(f"Multi-Model Energy Forecasting Comparison - {sector_name.upper()} Sector", fontsize=16, pad=15)
    ax.set_xlabel("Datetime", fontsize=12, labelpad=10)
    ax.set_ylabel(f"Consumption ({'kW' if sector_name == 'residential' else 'kWh' if sector_name == 'commercial' else 'MW'})", fontsize=12, labelpad=10)
    ax.legend(loc="upper right", framealpha=0.5, fontsize=11)
    ax.grid(True, color="#2c3e50", linestyle=":", alpha=0.6)
    
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(f"models/results/{sector_name}_comparison.png", dpi=150)
    plt.close()
    print(f"Comparison plot saved for {sector_name.upper()}!")

    return metrics

if __name__ == "__main__":
    all_results = {}
    all_results["residential"] = evaluate_sector("data/processed/final_residential.csv", "residential")
    all_results["commercial"] = evaluate_sector("data/processed/final_commercial.csv", "commercial")
    all_results["industrial"] = evaluate_sector("data/processed/final_industrial.csv", "industrial")
    
    # Save a global metrics comparison
    with open("models/results/global_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print("\nModel evaluation completed for all sectors! Metrics and charts exported.")
