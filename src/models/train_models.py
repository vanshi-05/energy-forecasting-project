import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from src.models.hybrid_model import HybridModel, StandaloneLSTM, SeasonalBaseline

def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, 0])  # Target is the first column
    return np.array(X), np.array(y)

def train_pytorch_model(model, X_train, y_train, epochs=5, batch_size=64, lr=0.001):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert to PyTorch Tensors
    tensor_x = torch.tensor(X_train, dtype=torch.float32)
    tensor_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"  Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")

def train_sector_models(csv_path, target_col, sector_name):
    print(f"\n======================================")
    print(f"Training models for {sector_name.upper()} sector...")
    print(f"======================================")
    
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # 80/20 Train/Test Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # ----------------------------------------------------
    # 1. Fit Seasonal Baseline Model
    # ----------------------------------------------------
    print("Fitting Seasonal Baseline...")
    seasonal_model = SeasonalBaseline()
    seasonal_model.fit(train_df, target_col)
    
    # Compute residuals
    train_seasonal_preds = seasonal_model.predict_season(train_df.index)
    train_residuals = train_df[target_col].values - train_seasonal_preds
    
    test_seasonal_preds = seasonal_model.predict_season(test_df.index)
    test_residuals = test_df[target_col].values - test_seasonal_preds

    # ----------------------------------------------------
    # 2. Train XGBoost Model (Tabular Supervised Learning)
    # ----------------------------------------------------
    print("Training XGBoost Regressor...")
    feature_cols = [
        'lag_1', 'lag_2', 'lag_24', 'rolling_mean_24', 'rolling_std_24',
        'air_temperature', 'wind_speed', 'cloud_coverage',
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos'
    ]
    
    X_train_xgb = train_df[feature_cols]
    y_train_xgb = train_df[target_col]
    
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=6, random_state=42)
    xgb_model.fit(X_train_xgb, y_train_xgb)
    xgb_model.save_model(f"models/saved/{sector_name}_xgb.json")

    # ----------------------------------------------------
    # 3. Train Standalone LSTM (On Target Load Directly)
    # ----------------------------------------------------
    print("Training Standalone LSTM...")
    # Scale target load
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled_target = target_scaler.fit_transform(train_df[[target_col]])
    
    X_lstm, y_lstm = create_sequences(train_scaled_target, seq_length=24)
    
    # Instantiate PyTorch LSTM
    lstm_model = StandaloneLSTM(input_dim=1, hidden_dim=64)
    train_pytorch_model(lstm_model, X_lstm, y_lstm, epochs=5, batch_size=64)
    torch.save(lstm_model.state_dict(), f"models/saved/{sector_name}_lstm.pth")

    # ----------------------------------------------------
    # 4. Train Proposed Hybrid CNN-BiLSTM-Attention Model (On Residuals)
    # ----------------------------------------------------
    print("Training Hybrid CNN-BiLSTM-Attention Model (on Residuals)...")
    
    # Scale Residuals
    res_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled_res = res_scaler.fit_transform(train_residuals.reshape(-1, 1))
    
    # Scale Weather Features
    weather_scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled_weather = weather_scaler.fit_transform(train_df[['air_temperature', 'wind_speed', 'cloud_coverage']])
    
    # Combined inputs: residual + weather + cyclical time features
    calendar_features = train_df[['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']].values
    
    # Create input feature matrix
    train_features = np.hstack([
        train_scaled_res,
        train_scaled_weather,
        calendar_features
    ]) # shape: (len, 10)
    
    X_hybrid, y_hybrid = create_sequences(train_features, seq_length=24)
    
    # Build and train PyTorch Hybrid model
    hybrid_model = HybridModel(input_dim=X_hybrid.shape[2], cnn_channels=64, lstm_hidden=64)
    train_pytorch_model(hybrid_model, X_hybrid, y_hybrid, epochs=5, batch_size=64)
    torch.save(hybrid_model.state_dict(), f"models/saved/{sector_name}_hybrid.pth")
    
    # Save scalers, baseline, and metadata
    metadata = {
        "target_scaler": target_scaler,
        "res_scaler": res_scaler,
        "weather_scaler": weather_scaler,
        "seasonal_baseline": seasonal_model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "split_idx": split_idx
    }
    with open(f"models/saved/{sector_name}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    print(f"All models and scalers for {sector_name.upper()} saved successfully!")

if __name__ == "__main__":
    os.makedirs("models/saved", exist_ok=True)
    
    # Train Residential
    train_sector_models(
        csv_path="data/processed/final_residential.csv",
        target_col="Global_active_power",
        sector_name="residential"
    )
    
    # Train Commercial
    train_sector_models(
        csv_path="data/processed/final_commercial.csv",
        target_col="meter_reading",
        sector_name="commercial"
    )
    
    # Train Industrial
    train_sector_models(
        csv_path="data/processed/final_industrial.csv",
        target_col="National Hourly Demand",
        sector_name="industrial"
    )
