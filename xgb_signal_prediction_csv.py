# xgb_signal_prediction_firebase.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- CONFIG ----------------
MODEL_OUT = "xgb_signal_strength_model.pkl"
TARGET = "download_mbps"  # Target variable
RANDOM_STATE = 42
FIREBASE_CRED_PATH = "C:\Users\Admin\AndroidStudioProjects\network_predicter\python_backend_prediction\firestore\databases\networkpredicter\networkpredictor-firebase-adminsdk-fbsvc-ab5e905e3b.json"  # <-- change this
COLLECTION_NAME = "networkLogs"
# ----------------------------------------

# ---------------- Initialize Firebase ----------------
cred = credentials.Certificate(FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------------- Fetch Data from Firebase ----------------
docs = db.collection(COLLECTION_NAME).stream()
records = []

for doc in docs:
    data = doc.to_dict()
    try:
        records.append({
            "signal_dbm": int(data.get("signal_dbm", -1)),
            "latitude": float(data.get("latitude", 0.0)),
            "longitude": float(data.get("longitude", 0.0)),
            "download_kbps": float(data.get("download_kbps", 0.0)),
            "upload_kbps": float(data.get("upload_kbps", 0.0)),
            "timestamp": pd.to_datetime(data.get("timestamp"))
        })
    except Exception as e:
        print(f"Skipping doc {doc.id} due to error: {e}")

df = pd.DataFrame(records)
if df.empty:
    raise ValueError("No valid network logs found in Firebase.")

# ---------------- Data Cleaning ----------------
df = df.dropna(subset=["signal_dbm", "latitude", "longitude", "download_kbps", "upload_kbps", "timestamp"])

# Convert speeds to Mbps
df["download_mbps"] = df["download_kbps"] / 1000
df["upload_mbps"] = df["upload_kbps"] / 1000

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
    "latitude": [6.2546],    # Example latitude
    "longitude": [5.2345],   # Example longitude
    "hour_sin": [np.sin(2 * np.pi * 14 / 24)],  # 2 PM
    "hour_cos": [np.cos(2 * np.pi * 14 / 24)],
    "weekday": [2]  # Wednesday
})
pred_speed = model.predict(example)[0]
print(f"🌐 Predicted Download Speed: {pred_speed:.2f} Mbps")
