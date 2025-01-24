import os
import requests
import json
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration: Replace with environment variables or configuration files
AQI_API_KEY = os.getenv("AQI_API_KEY", "d718e38e4583f8530ef42da3e31f0994")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "6887eeae47d04070bc281137251801")
CITY = "Karachi"

GEO_URL = f"http://api.openweathermap.org/geo/1.0/direct?q={CITY}&limit=1&appid={AQI_API_KEY}"

# Function to fetch coordinates
def fetch_coordinates():
    while True:
        response = requests.get(GEO_URL)
        if response.status_code == 200:
            data = response.json()
            if data:
                logging.info("Coordinates fetched successfully.")
                return data[0]['lat'], data[0]['lon']
            else:
                logging.error("City not found.")
                raise ValueError("City not found.")
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            logging.warning(f"Rate limit hit. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            logging.error(f"Failed to fetch coordinates: HTTP {response.status_code}")
            raise Exception(f"Failed to fetch coordinates: {response.status_code}")

# Fetch historical data for a specific timestamp
def fetch_historical_data(lat, lon, timestamp):
    AQI_URL = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={timestamp}&end={timestamp+3600}&appid={AQI_API_KEY}"
    WEATHER_URL = f"http://api.weatherapi.com/v1/history.json?key={WEATHER_API_KEY}&q={lat},{lon}&dt={datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')}"

    while True:
        aqi_response = requests.get(AQI_URL)
        weather_response = requests.get(WEATHER_URL)

        if aqi_response.status_code == 200 and weather_response.status_code == 200:
            logging.info(f"Data fetched successfully for timestamp {timestamp}.")
            return {
                "aqi": aqi_response.json(),
                "weather": weather_response.json()
            }
        elif aqi_response.status_code == 429 or weather_response.status_code == 429:
            retry_after = int(aqi_response.headers.get("Retry-After", 60))
            logging.warning(f"Rate limit hit for timestamp {timestamp}. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            logging.error(f"Failed to fetch data for timestamp {timestamp}: AQI - {aqi_response.status_code}, Weather - {weather_response.status_code}")
            return None

# Main script logic
if __name__ == "__main__":
    try:
        latitude, longitude = fetch_coordinates()
    except Exception as e:
        logging.error(f"Error fetching coordinates: {e}")
        exit(1)

    # Set up date range for the past 2-3 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    data = []

    current_date = start_date
    while current_date <= end_date:
        timestamp = int(current_date.timestamp())
        historical_data = fetch_historical_data(latitude, longitude, timestamp)
        if historical_data:
            # Ensure no duplicates
            if all(h["aqi"]["list"][0]["dt"] != historical_data["aqi"]["list"][0]["dt"] for h in data):
                data.append(historical_data)
        current_date += timedelta(hours=1)  # Increment by 1 hour

    # Save data to file
    with open("historical_aqi_and_weather_data.json", "w") as f:
        json.dump(data, f)

    logging.info("Historical AQI and weather data fetched and saved successfully.")
