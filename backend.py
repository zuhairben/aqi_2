from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import sqlite3
import pandas as pd
from typing import List

app = FastAPI()

# Load the model
MODEL_PATH = "model_registry/linear_regression.pkl"
model = joblib.load(MODEL_PATH)

# Database connection
DB_PATH = "feature_store.db"
def fetch_features():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM features ORDER BY timestamp DESC LIMIT 3"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

class PredictionRequest(BaseModel):
    features: List[float]

@app.get("/predict_next_3_days")
def predict_next_3_days():
    features = fetch_features()
    predictions = model.predict(features.iloc[:, 1:].values)  # Assuming features start after timestamp
    result = {
        "dates": features["timestamp"].tolist(),
        "predictions": predictions.tolist(),
    }
    return result

@app.get("/historical_aqi")
def historical_aqi():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM features"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data.to_dict(orient="records")