import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load historical data
try:
    with open("historical_aqi_and_weather_data.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    logging.error("The data file 'historical_aqi_and_weather_data.json' was not found.")
    exit(1)

# Flatten data and convert to DataFrame
records = []
for entry in data:
    try:
        timestamp = entry["aqi"]["list"][0]["dt"]
        dt = datetime.utcfromtimestamp(timestamp)
        components = entry["aqi"]["list"][0]["components"]
        main_aqi = entry["aqi"]["list"][0]["main"]["aqi"]
        weather = entry["weather"]["forecast"]["forecastday"][0]["hour"][0]

        records.append({
            "timestamp": timestamp,
            "datetime": dt,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "weekday": dt.weekday(),
            "aqi": main_aqi,
            "co": components.get("co"),
            "no": components.get("no"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "nh3": components.get("nh3"),
            "temperature": weather["temp_c"],
            "humidity": weather["humidity"],
            "wind_speed": weather["wind_kph"],
            "precipitation": weather["precip_mm"],
        })
    except KeyError as e:
        logging.warning(f"Missing key in entry: {e}")

df = pd.DataFrame(records)

# Remove rows with constant values
constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
if constant_columns:
    logging.info(f"Removing constant columns: {constant_columns}")
    df.drop(columns=constant_columns, inplace=True)

# Feature Engineering
df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

rolling_window = 24
for col in ["aqi", "pm2_5", "pm10"]:
    df[f"{col}_rolling_mean"] = df[col].rolling(window=rolling_window).mean()
    df[f"{col}_rolling_std"] = df[col].rolling(window=rolling_window).std()

lag_features = [1, 24, 48]
for col in ["aqi", "pm2_5", "pm10"]:
    for lag in lag_features:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)

for col in ["aqi", "pm2_5", "pm10"]:
    df[f"{col}_ema"] = df[col].ewm(span=24, adjust=False).mean()

df.dropna(inplace=True)

# Scaling numerical features
scaler = StandardScaler()
scaled_columns = ["temperature", "humidity", "wind_speed", "precipitation", "aqi", "pm2_5", "pm10"]
df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

df.to_csv("processed_features_advanced.csv", index=False)
logging.info("Feature computation completed and saved.")
