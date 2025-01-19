import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

# Load historical data
with open("historical_aqi_and_weather_data.json", "r") as f:
    data = json.load(f)

# Flatten the data and convert to a DataFrame
records = []
for entry in data:
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
        "temperature": weather["temp_c"],  # Temperature from WeatherAPI
        "humidity": weather["humidity"],   # Humidity from WeatherAPI
        "wind_speed": weather["wind_kph"], # Wind speed from WeatherAPI
        "precipitation": weather["precip_mm"],  # Precipitation from WeatherAPI
    })

df = pd.DataFrame(records)

# Feature Engineering

## 1. Time-Based Features
df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

## 2. Rolling Statistics (Advanced)
rolling_window = 24  # 1-day rolling window
for col in ["aqi", "pm2_5", "pm10"]:
    df[f"{col}_rolling_mean"] = df[col].rolling(window=rolling_window).mean()
    df[f"{col}_rolling_std"] = df[col].rolling(window=rolling_window).std()
    df[f"{col}_rolling_max"] = df[col].rolling(window=rolling_window).max()
    df[f"{col}_rolling_min"] = df[col].rolling(window=rolling_window).min()

## 3. Lag Features (Advanced)
lag_features = [1, 24, 48]  # Lag of 1, 24, 48 hours
for col in ["aqi", "pm2_5", "pm10"]:
    for lag in lag_features:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)  # Hourly, daily, and 2-day lag

## 4. Exponential Moving Average (EMA)
for col in ["aqi", "pm2_5", "pm10"]:
    df[f"{col}_ema"] = df[col].ewm(span=24, adjust=False).mean()  # 1-day EMA

## 5. Feature Interactions
df["temp_aqi_interaction"] = df["pm2_5"] * df["aqi"]
df["humidity_wind_interaction"] = df["humidity"] * df["wind_speed"]

## 6. Aggregation Features (Advanced)
df["daily_avg_aqi"] = df.groupby("day")["aqi"].transform("mean")
df["monthly_avg_aqi"] = df.groupby("month")["aqi"].transform("mean")
df["weekly_avg_aqi"] = df.groupby(df["datetime"].dt.isocalendar().week)["aqi"].transform("mean")


## 7. Statistical Features (Advanced)
df["aqi_median"] = df["aqi"].rolling(window=24).median()
df["aqi_max"] = df["aqi"].rolling(window=24).max()
df["aqi_min"] = df["aqi"].rolling(window=24).min()

## 8. Time Series Decomposition (Trend and Seasonality)
def decompose_timeseries(series):
    result = STL(series, period=24).fit()
    return result.trend, result.seasonal, result.resid

df["aqi_trend"], df["aqi_seasonal"], df["aqi_residual"] = decompose_timeseries(df["aqi"])

## 9. Anomaly Detection Features
df["is_aqi_spike"] = (df["aqi"] > df["aqi_rolling_mean"] + 2 * df["aqi_rolling_std"]).astype(int)
df["is_aqi_dip"] = (df["aqi"] < df["aqi_rolling_mean"] - 2 * df["aqi_rolling_std"]).astype(int)

## 10. Scaling Numerical Features
scaler = StandardScaler()
scaled_columns = ["temperature", "humidity", "wind_speed", "precipitation", "aqi", "pm2_5", "pm10"]
df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

# Drop NaN values created by rolling, lag, and other operations
df.dropna(inplace=True)

# Save processed features to CSV
df.to_csv("processed_features_advanced.csv", index=False)

print("Feature computation and engineering completed. Processed features saved to processed_features_advanced.csv.")
