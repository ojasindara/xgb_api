# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import datetime

app = Flask(__name__)

# Load model once at startup
model = joblib.load("xgb_signal_strength_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Expect JSON with signal_dbm, latitude, longitude

    try:
        signal_dbm = float(data["signal_dbm"])
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])

        # Use provided timestamp or current time
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            timestamp = datetime.datetime.fromisoformat(timestamp_str)
        else:
            timestamp = datetime.datetime.now()

        # Compute time features
        hour = timestamp.hour
        weekday = timestamp.weekday()
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Prepare features in same order as training
        df = pd.DataFrame([{
            "signal_dbm": signal_dbm,
            "latitude": latitude,
            "longitude": longitude,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "weekday": weekday
        }])

        # Make prediction
        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_speed": round(float(prediction), 2),
            "timestamp_used": timestamp.isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "✅ XGB Signal Prediction API is running!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
