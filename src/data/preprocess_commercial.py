import pandas as pd

def preprocess_commercial():
    # Load files
    df = pd.read_csv("data/raw/commercial_raw.csv")
    meta = pd.read_csv("data/raw/building_metadata.csv")
    weather = pd.read_csv("data/raw/weather_commercial.csv")

    # Keep only electricity meter
    df = df[df['meter'] == 0]

    # Merge with metadata
    df = df.merge(meta, on='building_id')

    # Group by timestamp (total commercial consumption)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    grouped = df.groupby('timestamp')['meter_reading'].sum().reset_index()

    # Merge weather
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    final = grouped.merge(weather, on='timestamp')

    final.set_index('timestamp', inplace=True)
    final.to_csv("data/processed/commercial_hourly.csv")

    print("Commercial preprocessing done!")

if __name__ == "__main__":
    preprocess_commercial()
