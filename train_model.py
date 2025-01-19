import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

# Load preprocessed data
file_path = "processed_data.csv"
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=["aqi"])
y = data["aqi"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2

# Initialize models
models = {
    "random_forest": RandomForestRegressor(random_state=42),
    "ridge_regression": Ridge(alpha=1.0),
    "linear_regression": LinearRegression(),
    "svr": SVR(),
    "xgboost": XGBRegressor(random_state=42),
    "lightgbm": lgb.LGBMRegressor(random_state=42)
}

# Dictionary to store the performance of each model
model_performance = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    rmse, mae, r2 = evaluate_model(model, X_test, y_test)
    model_performance[model_name] = {"rmse": rmse, "mae": mae, "r2": r2}
    print(f"{model_name} - RMSE: {rmse}, MAE: {mae}, R²: {r2}")

# Identify the best model based on R² score
best_model_name = max(model_performance, key=lambda x: model_performance[x]["r2"])
best_model = models[best_model_name]
best_model_r2 = model_performance[best_model_name]["r2"]

print(f"Best model: {best_model_name} with R²: {best_model_r2}")

# Create a directory for the model registry
model_registry_dir = "model_registry"
os.makedirs(model_registry_dir, exist_ok=True)

# Save the best model
model_file = os.path.join(model_registry_dir, f"{best_model_name}.pkl")
joblib.dump(best_model, model_file)

print(f"Best model ({best_model_name}) saved to {model_file}.")
