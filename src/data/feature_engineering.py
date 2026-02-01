import pandas as pd


def add_features(input_path, output_path, time_col, target_col):
    df = pd.read_csv(input_path)

    # Standardize datetime
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)

    # Lag features
    df['lag_1'] = df[target_col].shift(1)
    df['lag_24'] = df[target_col].shift(24)

    # Rolling mean
    df['rolling_mean_24'] = df[target_col].rolling(24).mean()

    # Time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month

    df.dropna(inplace=True)
    df.to_csv(output_path)

    print(f"Features added for {output_path}")


if __name__ == "__main__":

    # Residential
    add_features(
        "data/processed/residential_hourly.csv",
        "data/processed/final_residential.csv",
        time_col="datetime",
        target_col="Global_active_power"
    )

    # Commercial
    add_features(
        "data/processed/commercial_hourly.csv",
        "data/processed/final_commercial.csv",
        time_col="timestamp",
        target_col="meter_reading"
    )

    # Industrial
    add_features(
        "data/processed/industrial_hourly.csv",
        "data/processed/final_industrial.csv",
        time_col="datetime",
        target_col="National Hourly Demand"
    )
