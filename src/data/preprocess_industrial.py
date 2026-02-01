import pandas as pd

def preprocess_industrial():
    df = pd.read_excel("data/raw/industrial_india_raw.xlsx")

    # Keep only required columns
    df = df[['datetime', 'National Hourly Demand']]

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    df.to_csv("data/processed/industrial_hourly.csv")

    print("Industrial preprocessing done!")

if __name__ == "__main__":
    preprocess_industrial()
