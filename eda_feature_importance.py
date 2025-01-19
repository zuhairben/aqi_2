import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sqlite3

# Connect to the SQLite database to fetch the features
db_file = "feature_store.db"
conn = sqlite3.connect(db_file)

# Fetch the data
query = "SELECT * FROM features"
features_df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Quick overview of the data
print("Data Overview:")
print(features_df.info())
print(features_df.describe())

# --- EDA ---
# Univariate Analysis
plt.figure(figsize=(12, 6))
sns.histplot(features_df["aqi"], bins=20, kde=True)
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.show()

# Bivariate Analysis
plt.figure(figsize=(12, 6))
sns.scatterplot(data=features_df, x="weather_temperature", y="aqi", alpha=0.7)
plt.title("AQI vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("AQI")
plt.show()

numeric_features_df = features_df.select_dtypes(include=[np.number])

# Correlation Matrix
plt.figure(figsize=(14, 8))
correlation_matrix = numeric_features_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# --- Feature Importance ---
# Prepare the data
X = features_df.drop(columns=["aqi", "date"])
y = features_df["aqi"]

# Handle missing values separately for numeric and non-numeric columns
numeric_columns = X.select_dtypes(include=["number"]).columns
non_numeric_columns = X.select_dtypes(exclude=["number"]).columns

# Fill missing values for numeric columns with their mean
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

# Fill missing values for non-numeric columns with a placeholder
X[non_numeric_columns] = X[non_numeric_columns].fillna("Unknown")

# Label Encoding for categorical data
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in non_numeric_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Save processed data
processed_data = pd.concat([X, y], axis=1)
processed_data.to_csv("processed_data.csv", index=False)
print("Processed data saved to 'processed_data.csv'.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest for Feature Importance
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print(f"Random Forest - RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"Random Forest - MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"Random Forest - R²: {r2_score(y_test, y_pred)}")

# Feature Importance
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("Top 10 Important Features:")
print(feature_importances.head(10))

# Plot Feature Importances
plt.figure(figsize=(12, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importances.head(10))
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
