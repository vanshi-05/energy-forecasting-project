import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def train_model(csv_path, target_col, model_name):
    df = pd.read_csv(csv_path)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[target_col]])

    X, y = create_sequences(scaled)

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32)

    model.save(f"models/saved/{model_name}.h5")
    print(f"{model_name} model trained and saved!")


if __name__ == "__main__":
    train_model(
        "data/processed/final_residential.csv",
        "Global_active_power",
        "residential_lstm"
    )

    train_model(
        "data/processed/final_commercial.csv",
        "meter_reading",
        "commercial_lstm"
    )

    train_model(
        "data/processed/final_industrial.csv",
        "National Hourly Demand",
        "industrial_lstm"
    )
