import os
import numpy as np
import pandas as pd
from datetime import datetime

def generate_data(start_date="2023-01-01", end_date="2025-12-31"):
    print("Generating synthetic datasets...")
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("models/results", exist_ok=True)

    # Date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    n_hours = len(date_range)
    
    # Generate Weather Data
    # Temperature: annual cycle + diurnal cycle + noise
    doy = date_range.dayofyear
    hour = date_range.hour
    temp_annual = 20 + 10 * np.sin(2 * np.pi * (doy - 120) / 365)  # Peak in summer
    temp_diurnal = 5 * np.sin(2 * np.pi * (hour - 8) / 24)        # Peak in afternoon
    noise = np.random.normal(0, 2, n_hours)
    temperature = temp_annual + temp_diurnal + noise
    
    # Humidity: inverse to temperature
    humidity = 60 - 20 * np.sin(2 * np.pi * (hour - 8) / 24) + np.random.normal(0, 5, n_hours)
    humidity = np.clip(humidity, 10, 100)

    # ----------------------------------------------------
    # 1. Residential Raw Data (Date;Time;Global_active_power;...)
    # ----------------------------------------------------
    print("Generating residential data...")
    res_base = 1.2 # kW baseline
    res_diurnal = 0.8 * np.exp(-((hour - 8)/2)**2) + 1.5 * np.exp(-((hour - 20)/3)**2) # Peaks at 8am, 8pm
    res_weather = 0.05 * np.maximum(0, temperature - 22) + 0.04 * np.maximum(0, 15 - temperature) # AC & heating
    res_noise = np.random.normal(0, 0.3, n_hours)
    global_active_power = res_base + res_diurnal + res_weather + res_noise
    global_active_power = np.clip(global_active_power, 0.1, 8.0)

    res_df = pd.DataFrame({
        "Date": date_range.strftime("%d/%m/%Y"),
        "Time": date_range.strftime("%H:%M:%S"),
        "Global_active_power": np.round(global_active_power, 3),
        "Global_reactive_power": np.round(global_active_power * 0.1 + np.random.normal(0, 0.02, n_hours), 3),
        "Voltage": np.round(230 + np.random.normal(0, 2, n_hours), 1),
        "Global_intensity": np.round(global_active_power * 1000 / 230, 2),
        "Sub_metering_1": np.round(np.where(np.isin(hour, [7,8,18,19,20]), np.random.uniform(0, 10, n_hours), 0), 1),
        "Sub_metering_2": np.round(np.where(np.isin(hour, [8,12,13,20]), np.random.uniform(0, 5, n_hours), 0), 1),
        "Sub_metering_3": np.round(global_active_power * 5 + np.random.normal(0, 1, n_hours), 1)
    })
    
    # Save as semi-colon separated txt file
    res_df.to_csv("data/raw/residential_raw.txt", sep=";", index=False)
    print("Saved data/raw/residential_raw.txt")

    # ----------------------------------------------------
    # 2. Commercial Raw Data & Metadata & Weather
    # ----------------------------------------------------
    print("Generating commercial data...")
    # 5 commercial buildings
    n_buildings = 5
    building_ids = list(range(1, n_buildings + 1))
    
    # Metadata
    meta_df = pd.DataFrame({
        "building_id": building_ids,
        "primary_use": ["Office", "Office", "Education", "Education", "Retail"],
        "square_feet": [50000, 75000, 120000, 60000, 45000],
        "year_built": [1995, 2005, 2012, 1988, 2018],
        "floor_count": [4, 6, 8, 4, 2]
    })
    meta_df.to_csv("data/raw/building_metadata.csv", index=False)
    print("Saved data/raw/building_metadata.csv")
    
    # Weather
    weather_df = pd.DataFrame({
        "timestamp": date_range.strftime("%Y-%m-%d %H:%M:%S"),
        "air_temperature": np.round(temperature, 1),
        "dew_temperature": np.round(temperature - (100 - humidity)/5, 1),
        "sea_level_pressure": np.round(1013 + np.random.normal(0, 3, n_hours), 1),
        "wind_speed": np.round(np.random.rayleigh(3, n_hours), 1),
        "wind_direction": np.random.randint(0, 360, n_hours),
        "cloud_coverage": np.random.randint(0, 9, n_hours)
    })
    weather_df.to_csv("data/raw/weather_commercial.csv", index=False)
    print("Saved data/raw/weather_commercial.csv")

    # Readings for each building
    records = []
    is_weekend = date_range.dayofweek >= 5
    for b_id in building_ids:
        sq_ft = meta_df.loc[meta_df["building_id"] == b_id, "square_feet"].values[0]
        base_load = sq_ft * 0.05  # Wh per sq ft baseline
        
        # Operational multiplier: high during weekdays 8am-6pm, lower on weekends
        op_hours = (hour >= 8) & (hour <= 18)
        
        for idx, dt in enumerate(date_range):
            wknd = is_weekend[idx]
            hr = hour[idx]
            temp = temperature[idx]
            
            # Operational profiles
            if wknd:
                op_mult = 0.3 + 0.1 * np.sin(2 * np.pi * hr / 24)
            else:
                if hr >= 8 and hr <= 18:
                    op_mult = 1.5 + 0.3 * np.sin(2 * np.pi * (hr - 8) / 10)
                else:
                    op_mult = 0.5 + 0.1 * np.sin(2 * np.pi * hr / 24)
                    
            # Weather AC effect
            cool_mult = 1.0 + 0.05 * max(0, temp - 22) + 0.02 * max(0, 15 - temp)
            
            reading = base_load * op_mult * cool_mult + np.random.normal(0, base_load * 0.05)
            reading = max(10, reading)
            
            records.append({
                "building_id": b_id,
                "meter": 0, # Electricity
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "meter_reading": np.round(reading / 1000, 2) # convert to kWh
            })
            
    com_raw_df = pd.DataFrame(records)
    com_raw_df.to_csv("data/raw/commercial_raw.csv", index=False)
    print("Saved data/raw/commercial_raw.csv")

    # ----------------------------------------------------
    # 3. Industrial Raw Data (Excel)
    # ----------------------------------------------------
    print("Generating industrial data...")
    # Industrial: Heavy base load, continuous operation with 3 shifts
    # Base load is high (e.g., 50,000 kW)
    ind_base = 45000.0
    
    # Shift handovers create minor dips/peaks
    # Shift 1: 06:00 - 14:00, Shift 2: 14:00 - 22:00, Shift 3: 22:00 - 06:00
    shift_effect = np.zeros(n_hours)
    # Dip at shift change hours: 6, 14, 22
    for sc in [6, 14, 22]:
        shift_effect += -1500 * np.exp(-((hour - sc)/0.5)**2)
        
    # Weekday baseline is steady, weekend drops slightly (e.g. by 15% for maintenance)
    ind_weekend = np.where(is_weekend, -6000, 0)
    
    # Slight season variation (minimal compared to residential/commercial)
    ind_seasonal = 1000 * np.sin(2 * np.pi * doy / 365)
    
    ind_noise = np.random.normal(0, 800, n_hours)
    
    national_hourly_demand = ind_base + shift_effect + ind_weekend + ind_seasonal + ind_noise
    national_hourly_demand = np.clip(national_hourly_demand, 25000, 70000)

    ind_df = pd.DataFrame({
        "datetime": date_range,
        "National Hourly Demand": np.round(national_hourly_demand, 1),
        "Frequency": np.round(49.9 + np.random.normal(0, 0.05, n_hours), 2),
        "Grid_Loss_MW": np.round(national_hourly_demand * 0.03 + np.random.normal(0, 50, n_hours), 1)
    })
    
    ind_df.to_excel("data/raw/industrial_india_raw.xlsx", index=False)
    print("Saved data/raw/industrial_india_raw.xlsx")
    print("All synthetic data successfully generated!")

if __name__ == "__main__":
    generate_data()
