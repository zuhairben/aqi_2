import requests
import json
from datetime import datetime, timedelta
import time

# Configuration
AQI_API_KEY = "d718e38e4583f8530ef42da3e31f0994"  # AQI API key
WEATHER_API_KEY = "6887eeae47d04070bc281137251801"  # WeatherAPI key
CITY = "Karachi"
# Geocoding URL to get latitude and longitude
GEO_URL = f"http://api.openweathermap.org/geo/1.0/direct?q={CITY}&limit=1&appid={AQI_API_KEY}"

# Function to fetch coordinates
def fetch_coordinates():
    while True:
        response = requests.get(GEO_URL)
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]['lat'], data[0]['lon']
            else:
                raise Exception("City not found.")
        elif response.status_code == 429:  # Too Many Requests
            retry_after = int(response.headers.get("Retry-After", 60))  # Default to 60 seconds
            print(f"Rate limit hit. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            raise Exception(f"Failed to fetch coordinates: {response.status_code}")

# Historical Weather and AQI URL
def fetch_historical_data(lat, lon, timestamp):
    # AQI data URL
    AQI_URL = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={timestamp}&end={timestamp+3600}&appid={AQI_API_KEY}"
    # Weather data URL (WeatherAPI)
    WEATHER_URL = f"http://api.weatherapi.com/v1/history.json?key={WEATHER_API_KEY}&q={lat},{lon}&dt={datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')}"
    
    while True:
        # Fetch AQI data
        aqi_response = requests.get(AQI_URL)
        weather_response = requests.get(WEATHER_URL)

        if aqi_response.status_code == 200 and weather_response.status_code == 200:
            aqi_data = aqi_response.json()
            weather_data = weather_response.json()
            return {
                "aqi": aqi_data,
                "weather": weather_data
            }
        elif aqi_response.status_code == 429 or weather_response.status_code == 429:  # Too Many Requests
            retry_after = int(aqi_response.headers.get("Retry-After", 60))  # Default to 60 seconds
            print(f"Rate limit hit for timestamp {timestamp}. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            print(f"Failed to fetch data for timestamp {timestamp}: AQI - {aqi_response.status_code}, Weather - {weather_response.status_code}")
            return None

# Fetch coordinates
try:
    latitude, longitude = fetch_coordinates()
except Exception as e:
    print(f"Error fetching coordinates: {e}")
    exit(1)

# Backfill data for the past 2–3 months
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
data = []

current_date = start_date
while current_date <= end_date:
    timestamp = int(current_date.timestamp())
    historical_data = fetch_historical_data(latitude, longitude, timestamp)
    if historical_data:
        data.append(historical_data)
    current_date += timedelta(hours=1)  # Increment by 1 hour

# Save the backfilled data
with open("historical_aqi_and_weather_data.json", "w") as f:
    json.dump(data, f)

print("Historical AQI and weather data fetched and saved successfully.")
