import pandas as pd

def preprocess_residential(input_path, output_path):
    df = pd.read_csv(
        input_path,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        low_memory=False
    )

    df = df[['datetime', 'Global_active_power']]
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')

    df.set_index('datetime', inplace=True)

    # Resample to hourly
    hourly = df.resample('H').mean().dropna()

    hourly.to_csv(output_path)
    print("Residential preprocessing done!")

if __name__ == "__main__":
    preprocess_residential(
        "data/raw/residential_raw.txt",
        "data/processed/residential_hourly.csv"
    )
