import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def evaluate(csv_path, model_path, target_col):
    df = pd.read_csv(csv_path)

# Auto-detect time column
    time_col = 'datetime' if 'datetime' in df.columns else 'timestamp'

    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[[target_col]])

    X, y = [], []
    for i in range(24, len(data)):
        X.append(data[i-24:i])
        y.append(data[i])

    X, y = np.array(X), np.array(y)

    model = load_model(model_path, compile=False)
    preds = model.predict(X)

    preds = scaler.inverse_transform(preds)
    y = scaler.inverse_transform(y)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    print(f"\nResults for {model_path}")
    print("MAE :", mae)
    print("RMSE:", rmse)

    plt.figure(figsize=(12,6))
    plt.plot(y[:200], label="Actual")
    plt.plot(preds[:200], label="Predicted")
    plt.legend()
    plt.title(model_path)
    plt.show()


if __name__ == "__main__":
    evaluate(
        "data/processed/final_residential.csv",
        "models/saved/residential_lstm.h5",
        "Global_active_power"
    )

    evaluate(
        "data/processed/final_commercial.csv",
        "models/saved/commercial_lstm.h5",
        "meter_reading"
    )

    evaluate(
        "data/processed/final_industrial.csv",
        "models/saved/industrial_lstm.h5",
        "National Hourly Demand"
    )
