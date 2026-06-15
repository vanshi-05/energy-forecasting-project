import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np

def download_file(url, output_path):
    print(f"Downloading {url}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Saved to {output_path}")

def download_and_extract_uci():
    print("\n----------------------------------------------------")
    print("1. Downloading UCI Residential Dataset...")
    print("----------------------------------------------------")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    zip_path = "data/raw/household_power_consumption.zip"
    
    os.makedirs("data/raw", exist_ok=True)
    download_file(url, zip_path)
    
    print("Extracting ZIP file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract("household_power_consumption.txt", "data/raw")
        
    # Rename extracted file to residential_raw.txt
    dest_path = "data/raw/residential_raw.txt"
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename("data/raw/household_power_consumption.txt", dest_path)
    os.remove(zip_path)
    print("Residential dataset extracted successfully: data/raw/residential_raw.txt")

def download_bdg_commercial():
    print("\n----------------------------------------------------")
    print("2. Downloading BDG2 Commercial Dataset (Panther Site)...")
    print("----------------------------------------------------")
    
    # URL references
    meta_url = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/metadata/metadata.csv"
    weather_url = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/weather/weather.csv"
    elec_url = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/meters/cleaned/electricity_cleaned.csv"
    
    os.makedirs("data/raw", exist_ok=True)
    
    # 2.1 Download and process building metadata
    print("Fetching metadata and filtering for 'Panther' site...")
    meta_df = pd.read_csv(meta_url)
    # Filter for site 'Panther' and building types we want (e.g. Office, Education)
    panther_meta = meta_df[meta_df['site_id'] == 'Panther'].head(5).copy()
    
    # Map building name columns to ids 1 to 5
    building_names = panther_meta['building_id'].tolist()
    panther_meta['old_building_name'] = building_names
    panther_meta['building_id'] = range(1, len(building_names) + 1)
    
    # Rename metadata columns to match preprocess_commercial.py expectations
    # Expected columns: building_id, primary_use, square_feet, year_built, floor_count
    # BDGP2 columns: building_id, primaryspaceusage, sqft, yearbuilt, floorcount
    panther_meta_clean = panther_meta[[
        'building_id', 'primaryspaceusage', 'sqft', 'yearbuilt', 'numberoffloors', 'old_building_name'
    ]].rename(columns={
        'primaryspaceusage': 'primary_use',
        'sqft': 'square_feet',
        'yearbuilt': 'year_built',
        'numberoffloors': 'floor_count'
    })
    
    panther_meta_clean.to_csv("data/raw/building_metadata.csv", index=False)
    print("Saved data/raw/building_metadata.csv")

    # 2.2 Download and process weather data
    print("Fetching weather data and filtering for 'Panther' site...")
    weather_df = pd.read_csv(weather_url)
    panther_weather = weather_df[weather_df['site_id'] == 'Panther'].copy()
    
    # Rename columns to match preprocess_commercial.py
    # Expected: timestamp, air_temperature, dew_temperature, sea_level_pressure, wind_speed, wind_direction, cloud_coverage
    # BDGP2: timestamp, airTemperature, cloudCoverage, dewTemperature, precipDepth1HR, seaLevelPressure, windDirection, windSpeed
    panther_weather_clean = panther_weather[[
        'timestamp', 'airTemperature', 'dewTemperature', 'seaLvlPressure', 'windSpeed', 'windDirection', 'cloudCoverage'
    ]].rename(columns={
        'airTemperature': 'air_temperature',
        'dewTemperature': 'dew_temperature',
        'seaLvlPressure': 'sea_level_pressure',
        'windSpeed': 'wind_speed',
        'windDirection': 'wind_direction',
        'cloudCoverage': 'cloud_coverage'
    })
    panther_weather_clean.to_csv("data/raw/weather_commercial.csv", index=False)
    print("Saved data/raw/weather_commercial.csv")

    # 2.3 Download electricity meter readings (LFS file - we stream it to save memory/bandwidth)
    print("Streaming electricity meter data and extracting Panther building loads...")
    
    # Get columns we need (timestamp + the 5 selected buildings)
    col_indices = {}
    r = requests.get(elec_url, stream=True)
    r.raise_for_status()
    
    # Create the generator once to prevent stream misalignment
    lines_gen = r.iter_lines()
    
    # Read the header line to map building columns
    header_line = next(lines_gen).decode('utf-8')
    header_cols = header_line.split(',')
    
    # Map from old building name to building_id
    name_to_id = dict(zip(building_names, range(1, len(building_names) + 1)))
    
    target_indices = []
    building_ids_ordered = []
    
    # Column 0 is timestamp
    timestamp_idx = 0
    
    for idx, col in enumerate(header_cols):
        if col in name_to_id:
            target_indices.append(idx)
            building_ids_ordered.append(name_to_id[col])
            
    print(f"Target building columns found at indices: {target_indices}")
    
    # Read remaining rows and format into building_id, meter, timestamp, meter_reading
    # Expected columns: building_id, meter, timestamp, meter_reading
    records = []
    line_count = 0
    
    for line in lines_gen:
        if not line:
            continue
        line_str = line.decode('utf-8')
        row_vals = line_str.split(',')
        
        timestamp = row_vals[timestamp_idx]
        
        for t_idx, b_id in zip(target_indices, building_ids_ordered):
            try:
                reading_str = row_vals[t_idx]
                # If reading is empty or null, skip it or set to NaN
                if reading_str == "" or reading_str.lower() == "nan":
                    continue
                reading = float(reading_str)
                records.append({
                    "building_id": b_id,
                    "meter": 0, # Electricity
                    "timestamp": timestamp,
                    "meter_reading": reading
                })
            except (ValueError, IndexError):
                continue
                
        line_count += 1
        if line_count % 5000 == 0:
            print(f"Processed {line_count} hourly steps...")
            
    com_df = pd.DataFrame(records)
    com_df.to_csv("data/raw/commercial_raw.csv", index=False)
    print(f"Commercial dataset compiled successfully: data/raw/commercial_raw.csv (Shape: {com_df.shape})")

def download_pjm_industrial():
    print("\n----------------------------------------------------")
    print("3. Downloading PJM Industrial Load Dataset...")
    print("----------------------------------------------------")
    url = "https://raw.githubusercontent.com/soumilshah1995/Data-Analysis-Over-10-years-of-hourly-energy-consumption-data-from-PJM-in-Megawatts/master/AEP_hourly.csv"
    csv_path = "data/raw/AEP_hourly.csv"
    
    download_file(url, csv_path)
    
    # Process into industrial_india_raw.xlsx format
    # Expected columns: datetime, National Hourly Demand
    print("Formatting PJM dataset as Excel...")
    df = pd.read_csv(csv_path)
    df.rename(columns={'Datetime': 'datetime', 'AEP_MW': 'National Hourly Demand'}, inplace=True)
    df = df[['datetime', 'National Hourly Demand']]
    
    # Save as Excel
    excel_path = "data/raw/industrial_india_raw.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"Industrial dataset formatted and saved: {excel_path}")
    
    # Cleanup temporary CSV
    os.remove(csv_path)

if __name__ == "__main__":
    download_and_extract_uci()
    download_bdg_commercial()
    download_pjm_industrial()
    print("\nAll real-world research datasets successfully downloaded and structured!")
