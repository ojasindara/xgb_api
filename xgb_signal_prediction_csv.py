# xgb_signal_prediction_supabase.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from supabase import create_client, Client

# ---------------- CONFIG ----------------
MODEL_OUT = "xgb_signal_strength_model.pkl"
TARGET = "download_mbps"  # Target variable
RANDOM_STATE = 42

# ---------------- Supabase Config ----------------
SUPABASE_URL = "https://lofhphqjdfairgjqhjvp.supabase.co"   # Replace with your Supabase project URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxvZmhwaHFqZGZhaXJnanFoanZwIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDk5NTk0MSwiZXhwIjoyMDc2NTcxOTQxfQ.tLSf-lAxJEmQYPT5g0-JHLO_ndIL2UDBEdMzFNq23ag"                      # Replace with your Supabase service key

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
TABLE_NAME = "speed_logs"

# ---------------- Fetch Data from Supabase ----------------
response = supabase.table(TABLE_NAME).select("*").execute()
records_raw = response.data

if not records_raw:
    raise ValueError("No valid network logs found in Supabase.")

records = []
for data in records_raw:
    try:
        records.append({
            "signal_dbm": int(data.get("signal_strength", -1)),
            "latitude": float(data.get("latitude", 0.0)),
            "longitude": float(data.get("longitude", 0.0)),
            "download_mbps": float(data.get("download_speed", 0.0)),
            "upload_mbps": float(data.get("upload_speed", 0.0)),
            "timestamp": pd.to_datetime(data.get("timestamp"))
        })
    except Exception as e:
        print(f"Skipping record due to error: {e}")

df = pd.DataFrame(records)

# ---------------- Data Cleaning ----------------
df = df.dropna(subset=["signal_dbm", "latitude", "longitude", "download_mbps", "upload_mbps", "timestamp"])

# ---------------- Feature Engineering (Time) ----------------
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

# Encode cyclic hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ---------------- Features & Target ----------------
features = ["signal_dbm", "latitude", "longitude", "hour_sin", "hour_cos", "weekday"]
target = TARGET

X = df[features]
y = df[target]

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------- Train XGBoost Model ----------------
model = XGBRegressor(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)

# ---------------- Evaluate Model ----------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully")
print(f"📉 Mean Squared Error: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")

# ---------------- Save Model ----------------
joblib.dump(model, MODEL_OUT)
print(f"💾 Model saved as '{MODEL_OUT}'")

# ---------------- Example Prediction ----------------
example = pd.DataFrame({
    "signal_dbm": [-80],
    "latitude": [6.2546],
    "longitude": [5.2345],
    "hour_sin": [np.sin(2 * np.pi * 14 / 24)],
    "hour_cos": [np.cos(2 * np.pi * 14 / 24)],
    "weekday": [2]
})
pred_speed = model.predict(example)[0]
print(f"🌐 Predicted Download Speed: {pred_speed:.2f} Mbps")
