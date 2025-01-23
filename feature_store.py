import sqlite3
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
features_file = "processed_features_advanced.csv"
db_file = "feature_store.db"

try:
    # Load features from CSV file
    logging.info(f"Loading features from {features_file}")
    features_df = pd.read_csv(features_file)

    # Ensure column names and features match expected format
    features_df = features_df.rename(columns={
        "datetime": "date",
        "pm2_5": "components_pm2_5",
        "pm10": "components_pm10",
        "temperature": "weather_temperature",
        "humidity": "weather_humidity",
        "wind_speed": "weather_wind_speed",
        "precipitation": "weather_precipitation"
    })

    # Add `aqi_category` if missing
    if 'aqi_category' not in features_df.columns:
        logging.info("Adding missing `aqi_category` column")
        features_df["aqi_category"] = pd.cut(
            features_df["aqi"], 
            bins=[0, 1, 2, 3, 4, 5], 
            labels=["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"], 
            right=False
        )

    # Define required columns
    required_columns = [
        "date", "components_pm2_5", "components_pm10", "aqi", "hour", "day", "month",
        "aqi_rolling_mean", "aqi_rolling_std", "aqi_rolling_max", "aqi_rolling_min",
        "aqi_lag_1", "aqi_lag_24", "aqi_lag_48", "aqi_ema", "aqi_category", "is_aqi_spike",
        "is_aqi_dip", "temp_aqi_interaction", "humidity_wind_interaction", "daily_avg_aqi",
        "monthly_avg_aqi", "weekly_avg_aqi", "aqi_median", "aqi_max", "aqi_min",
        "aqi_trend", "aqi_seasonal", "aqi_residual", "is_weekend", "hour_sin",
        "hour_cos", "weather_temperature", "weather_humidity", "weather_wind_speed",
        "weather_precipitation"
    ]

    # Filter columns based on availability in the dataset
    available_columns = [col for col in required_columns if col in features_df.columns]
    missing_columns = [col for col in required_columns if col not in features_df.columns]

    if missing_columns:
        logging.warning(f"The following required columns are missing: {missing_columns}")

    # Retain only available columns
    features_df = features_df[available_columns]

    # Connect to SQLite database
    logging.info(f"Connecting to database: {db_file}")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS features (
        date TEXT PRIMARY KEY,
        components_pm2_5 REAL, components_pm10 REAL, aqi INTEGER, hour INTEGER, day INTEGER,
        month INTEGER, aqi_rolling_mean REAL, aqi_rolling_std REAL, aqi_rolling_max REAL,
        aqi_rolling_min REAL, aqi_lag_1 REAL, aqi_lag_24 REAL, aqi_lag_48 REAL, aqi_ema REAL,
        aqi_category TEXT, is_aqi_spike INTEGER, is_aqi_dip INTEGER, temp_aqi_interaction REAL,
        humidity_wind_interaction REAL, daily_avg_aqi REAL, monthly_avg_aqi REAL, weekly_avg_aqi REAL,
        aqi_median REAL, aqi_max REAL, aqi_min REAL, aqi_trend REAL, aqi_seasonal REAL, aqi_residual REAL,
        is_weekend INTEGER, hour_sin REAL, hour_cos REAL, weather_temperature REAL, weather_humidity REAL,
        weather_wind_speed REAL, weather_precipitation REAL
    )
    """
    cursor.execute(create_table_query)

    # Insert features into the database
    logging.info("Inserting features into the database")
    features_df.to_sql("features", conn, if_exists="replace", index=False)

    # Commit changes and close connection
    conn.commit()
    conn.close()
    logging.info(f"Features successfully stored in {db_file}")

except Exception as e:
    logging.error(f"An error occurred: {e}")
    raise
