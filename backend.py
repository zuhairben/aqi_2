from flask import Flask, jsonify, request
import joblib
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Load the best model
model_path = "model_registry/linear_regression.pkl"
model = joblib.load(model_path)

# Connect to the feature store
db_file = "feature_store.db"

# Preprocessing helpers
label_encoders = {}  # Store label encoders for non-numeric columns

def preprocess_data(X):
    """Preprocess input features for prediction."""
    # Handle missing values for numeric and non-numeric columns
    numeric_cols = X.select_dtypes(include=["number"]).columns
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns

    # Fill missing values
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    X[non_numeric_cols] = X[non_numeric_cols].fillna("Unknown")

    # Label encode non-numeric columns (fit if not done already)
    for col in non_numeric_cols:
        if col not in label_encoders:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            X[col] = le.transform(X[col])

    # Ensure feature alignment with the model
    model_features = model.feature_names_in_
    X = X.reindex(columns=model_features, fill_value=0)

    return X

def fetch_latest_features():
    """Fetch the most recent features from the Feature Store."""
    conn = sqlite3.connect(db_file)
    query = """
    SELECT *
    FROM features
    ORDER BY date DESC
    LIMIT 1
    """
    latest_features = pd.read_sql_query(query, conn)
    conn.close()
    return latest_features

@app.route("/predict", methods=["GET"])
def predict():
    """Predict AQI for the next 3 days."""
    latest_features = fetch_latest_features()
    if latest_features.empty:
        return jsonify({"error": "No feature data available."}), 400

    # Drop non-feature columns
    X = latest_features.drop(columns=["aqi", "date"])

    # Preprocess data
    X_processed = preprocess_data(X)

    # Generate predictions for the next three days
    predictions = []
    for i in range(3):
        predicted_aqi = model.predict(X_processed)[0]
        predictions.append({
            "date": (datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
            "predicted_aqi": round(predicted_aqi, 2)
        })

    return jsonify({"predictions": predictions})

@app.route("/predict_custom", methods=["POST"])
def predict_custom():
    """Predict AQI for custom user input."""
    try:
        input_data = request.json
        logging.debug("Received custom input: %s", input_data)

        # Map custom input keys to expected feature names
        feature_mapping = {
            "temperature": "weather_temperature",
            "humidity": "weather_humidity",
            "wind_speed": "weather_wind_speed",
        }

        # Transform input data to match expected feature names
        transformed_data = {feature_mapping[key]: value for key, value in input_data.items() if key in feature_mapping}

        # Validate input data
        required_features = ["weather_temperature", "weather_humidity", "weather_wind_speed"]
        if not all(feature in transformed_data for feature in required_features):
            return jsonify({"error": f"Missing required features. Expected: {required_features}"}), 400

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([transformed_data], columns=required_features)

        # Preprocess custom input data
        X_processed = preprocess_data(input_df)

        # Generate prediction
        predicted_aqi = model.predict(X_processed)[0]
        return jsonify({"predicted_aqi": round(predicted_aqi, 2)})
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return jsonify({"error": f"Error during prediction: {e}"}), 400

if __name__ == "__main__":
    app.run(debug=True)
