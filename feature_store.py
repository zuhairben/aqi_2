import sqlite3
import pandas as pd

# Load features from the updated processed CSV file
# Load features from the updated processed CSV file
features_file = "processed_features_advanced.csv"
features_df = pd.read_csv(features_file)

# Ensure all new features are included and column names match
features_df = features_df.rename(columns={
    "datetime": "date",
    "pm2_5": "components_pm2_5",
    "pm10": "components_pm10",
    "temperature": "weather_temperature",
    "humidity": "weather_humidity",
    "wind_speed": "weather_wind_speed",
    "precipitation": "weather_precipitation"
})

# Check if 'aqi_category' exists, otherwise, create it
if 'aqi_category' not in features_df.columns:
    # If missing, create the aqi_category column
    features_df["aqi_category"] = pd.cut(features_df["aqi"], bins=[0, 1, 2, 3, 4, 5], labels=["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"], right=False)

# Add derived features explicitly to match feature engineering
features_df = features_df[[
    "date",
    "components_pm2_5",
    "components_pm10",
    "aqi",
    "hour",
    "day",
    "month",
    "aqi_rolling_mean",
    "aqi_rolling_std",
    "aqi_rolling_max",
    "aqi_rolling_min",
    "aqi_lag_1",
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_ema",
    "aqi_category",  # This should now always exist
    "is_aqi_spike",
    "is_aqi_dip",
    "temp_aqi_interaction",
    "humidity_wind_interaction",
    "daily_avg_aqi",
    "monthly_avg_aqi",
    "weekly_avg_aqi",
    "aqi_median",
    "aqi_max",
    "aqi_min",
    "aqi_trend",
    "aqi_seasonal",
    "aqi_residual",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weather_temperature",
    "weather_humidity",
    "weather_wind_speed",
    "weather_precipitation"
]]

# Create a SQLite database (or connect to it if it already exists)
db_file = "feature_store.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create a table for features if it doesn't exist
create_table_query = """
CREATE TABLE IF NOT EXISTS features (
    date TEXT PRIMARY KEY,
    components_pm2_5 REAL,
    components_pm10 REAL,
    aqi INTEGER,
    hour INTEGER,
    day INTEGER,
    month INTEGER,
    aqi_rolling_mean REAL,
    aqi_rolling_std REAL,
    aqi_rolling_max REAL,
    aqi_rolling_min REAL,
    aqi_lag_1 REAL,
    aqi_lag_24 REAL,
    aqi_lag_48 REAL,
    aqi_ema REAL,
    aqi_category TEXT,
    is_aqi_spike INTEGER,
    is_aqi_dip INTEGER,
    temp_aqi_interaction REAL,
    humidity_wind_interaction REAL,
    daily_avg_aqi REAL,
    monthly_avg_aqi REAL,
    weekly_avg_aqi REAL,
    aqi_median REAL,
    aqi_max REAL,
    aqi_min REAL,
    aqi_trend REAL,
    aqi_seasonal REAL,
    aqi_residual REAL,
    is_weekend INTEGER,
    hour_sin REAL,
    hour_cos REAL,
    weather_temperature REAL,
    weather_humidity REAL,
    weather_wind_speed REAL,
    weather_precipitation REAL
)
"""
cursor.execute(create_table_query)

# Insert the features into the database
features_df.to_sql("features", conn, if_exists="replace", index=False)

# Close the connection
conn.commit()
conn.close()

print(f"Features successfully stored in {db_file}.")
