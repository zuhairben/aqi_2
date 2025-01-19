import streamlit as st
import requests
import pandas as pd
import plotly.express as px

BACKEND_URL = "http://localhost:8000"

st.title("Karachi AQI Prediction Dashboard")
st.sidebar.header("Navigation")
options = ["Real-time AQI", "Forecast AQI"]
choice = st.sidebar.radio("Choose a page:", options)

if choice == "Real-time AQI":
    st.header("Current and Historical AQI")
    response = requests.get(f"{BACKEND_URL}/historical_aqi")
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        st.dataframe(data)
        fig = px.line(data, x="timestamp", y="aqi", title="Historical AQI")
        st.plotly_chart(fig)
    else:
        st.error("Failed to fetch historical AQI data.")

elif choice == "Forecast AQI":
    st.header("Next 3 Days AQI Prediction")
    response = requests.get(f"{BACKEND_URL}/predict_next_3_days")
    if response.status_code == 200:
        data = response.json()
        predictions = pd.DataFrame({"Date": data["dates"], "AQI Prediction": data["predictions"]})
        st.dataframe(predictions)
        fig = px.bar(predictions, x="Date", y="AQI Prediction", title="Next 3 Days AQI")
        st.plotly_chart(fig)
    else:
        st.error("Failed to fetch prediction data.")