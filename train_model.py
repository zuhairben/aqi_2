import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import os
import sqlite3

# Load raw data (from feature store database or directly from a CSV file)
file_path = "feature_store.db"  # Adjust as needed
query = "SELECT * FROM features"
data = pd.read_sql_query(query, sqlite3.connect(file_path))

# Preprocess the data
X = data.drop(columns=["aqi", "date"])
y = data["aqi"]

# Handle missing values separately for numeric and non-numeric columns
numeric_columns = X.select_dtypes(include=["number"]).columns
non_numeric_columns = X.select_dtypes(exclude=["number"]).columns

# Fill missing values for numeric columns with their mean
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

# Fill missing values for non-numeric columns with a placeholder
X[non_numeric_columns] = X[non_numeric_columns].fillna("Unknown")

# Label Encoding for categorical data
label_encoders = {}
for col in non_numeric_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Save label encoders for use in the web app
joblib.dump(label_encoders, "model_registry/label_encoders.pkl")

# Standardize features for models sensitive to scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for use in the web app
joblib.dump(scaler, "model_registry/scaler.pkl")

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
    "elastic_net": ElasticNet(random_state=42),
    "svr": SVR(),
    "xgboost": XGBRegressor(random_state=42),
    "lightgbm": lgb.LGBMRegressor(random_state=42),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
    "catboost": CatBoostRegressor(verbose=0, random_state=42)
}

# Dictionary to store the performance of each model
model_performance = {}

# Train and evaluate each model
for model_name, model in models.items():
    try:
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        rmse, mae, r2 = evaluate_model(model, X_test, y_test)
        model_performance[model_name] = {"rmse": rmse, "mae": mae, "r2": r2}
        print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Identify the best model based on R² score
best_model_name = max(model_performance, key=lambda x: model_performance[x]["r2"])
best_model = models[best_model_name]
best_model_r2 = model_performance[best_model_name]["r2"]

print(f"Best model: {best_model_name} with R²: {best_model_r2:.2f}")

# Create a directory for the model registry
model_registry_dir = "model_registry"
os.makedirs(model_registry_dir, exist_ok=True)

# Save the best model
model_file = os.path.join(model_registry_dir, f"{best_model_name}.pkl")
joblib.dump(best_model, model_file)
print(f"Best model ({best_model_name}) saved to {model_file}.")

# Save the feature names for use in the web app
joblib.dump(list(X.columns), "model_registry/feature_names.pkl")

# Save model performance metrics to a CSV file
performance_df = pd.DataFrame(model_performance).T.reset_index()
performance_df.columns = ["model", "rmse", "mae", "r2"]
performance_file = os.path.join(model_registry_dir, "model_performance.csv")
performance_df.to_csv(performance_file, index=False)
print(f"Model performance metrics saved to {performance_file}.")
