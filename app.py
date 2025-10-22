# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model once at startup
model = joblib.load("xgb_signal_strength_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Expect JSON with signal_dbm, latitude, longitude
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({"predicted_speed": float(prediction)})

# Health check endpoint
@app.route("/", methods=["GET"])
def index():
    return "XGB Signal Prediction API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
