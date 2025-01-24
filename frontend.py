import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Backend API URLs
PREDICTION_API_URL = "http://127.0.0.1:5000/predict"
CUSTOM_API_URL = "http://127.0.0.1:5000/predict_custom"
TREND_API_URL = "http://127.0.0.1:5000/aqi_trend"

st.set_page_config(page_title="AQI Predictor", layout="wide")

st.title("Air Quality Index (AQI) Predictor")
st.markdown(
    "This dashboard provides real-time AQI data predictions and trends, including forecasts for the next three days."
)

# Fetch AQI predictions from the backend
def fetch_predictions():
    try:
        response = requests.get(PREDICTION_API_URL)
        if response.status_code == 200:
            return response.json()["predictions"]
        else:
            st.error("Failed to fetch predictions from the backend.")
            return None
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return None

# Fetch AQI trend data from the backend
def fetch_aqi_trend():
    try:
        response = requests.get(TREND_API_URL)
        if response.status_code == 200:
            return response.json()["aqi_trend"]
        else:
            st.error("Failed to fetch AQI trend data from the backend.")
            return None
    except Exception as e:
        st.error(f"Error fetching AQI trend data: {e}")
        return None

# Display AQI trend
st.subheader("AQI Trend Over Time")
aqi_trend_data = fetch_aqi_trend()
if aqi_trend_data:
    trend_df = pd.DataFrame(aqi_trend_data)
    trend_df["date"] = pd.to_datetime(trend_df["date"])

    # Create a line graph
    fig = px.line(
        trend_df,
        x="date",
        y="aqi",
        title="AQI Trend (Including Predictions for Next 3 Days)",
        labels={"date": "Date", "aqi": "AQI"},
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# Display predictions
predictions = fetch_predictions()
if predictions:
    st.subheader("Predicted AQI for the Next 3 Days")
    forecast_df = pd.DataFrame(predictions)
    st.dataframe(forecast_df)

    # Visualize predictions with a bar chart
    fig = px.bar(
        forecast_df,
        x="date",
        y="predicted_aqi",
        color="predicted_aqi",
        title="AQI Forecast for the Next 3 Days",
        labels={"predicted_aqi": "Predicted AQI"},
        color_continuous_scale="RdYlGn_r",
    )
    st.plotly_chart(fig, use_container_width=True)

# Add a real-time interactive input form
st.sidebar.header("Custom Input for AQI Prediction")
st.sidebar.write("Adjust features to see how they affect AQI predictions.")

# Sample feature input form
temperature = st.sidebar.number_input("Temperature (°C)", value=25.0, step=0.1)
humidity = st.sidebar.number_input("Humidity (%)", value=50.0, step=1.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", value=10.0, step=0.5)

if st.sidebar.button("Predict"):
    # Send custom input to the backend
    try:
        custom_prediction = requests.post(CUSTOM_API_URL, json={
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed
        })

        if custom_prediction.status_code == 200:
            prediction_result = custom_prediction.json()
            if "predicted_aqi" in prediction_result:
                st.sidebar.success(f"Predicted AQI: {prediction_result['predicted_aqi']}")
            else:
                st.sidebar.error("Unexpected response format from the backend.")
        else:
            st.sidebar.error(f"Error: {custom_prediction.status_code} - {custom_prediction.text}")
    except Exception as e:
        st.sidebar.error(f"Error sending custom input: {e}")
