from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Try to load the model at import time, but don't let the app crash if it fails.
model = None
try:
    model = load_model("model/wave_height_lstm_model.keras")
except Exception as e:
    print(f"Warning: could not load model at startup: {e}")

# Defer loading/processing the CSV until a request so missing/renamed columns
# won't crash the app at import time. We'll try a best-effort to find a column
# to use for wave heights.
data = None
wave = None
scaler = None
scaled = None

def ensure_data_loaded():
    global data, wave, scaler, scaled
    if scaled is not None:
        return
    try:
        data = pd.read_csv(
            "data/wave_data.csv",
            sep=r"\s+",
            comment="#",
            header=None,
            names=[
                "YY",
                "MM",
                "DD",
                "hh",
                "mm",
                "WDIR",
                "WSPD",
                "GST",
                "WVHT",
                "DPD",
                "APD",
                "MWD",
                "PRES",
                "ATMP",
                "WTMP",
                "DEWP",
                "VIS",
                "TIDE",
            ],
        )
    except Exception as e:
        raise RuntimeError(f"failed to read CSV: {e}")

    # Replace placeholder values (99.00, 999, etc.) with NaN
    data.replace({99.00: np.nan, 999: np.nan, 999.0: np.nan}, inplace=True)
    
    # Prefer 'WVHT' column, otherwise pick the first numeric column available
    if 'WVHT' in data.columns:
        wave = data[['WVHT']].interpolate(method='linear').ffill().bfill()
    else:
        # find first numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise RuntimeError(f"no numeric columns found in CSV; columns: {list(data.columns)}")
        chosen = numeric_cols[0]
        print(f"Warning: 'WVHT' not found, using column '{chosen}' instead")
        wave = data[[chosen]].interpolate(method='linear').ffill().bfill()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(wave)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    global model
    # Lazy-load the model if it wasn't available at startup
    if model is None:
        try:
            model = load_model("model/wave_height_lstm_model.keras")
        except Exception as e:
            return jsonify({"error": "model not available", "details": str(e)}), 500
    try:
        # Ensure CSV and scaler are loaded/prepared
        try:
            ensure_data_loaded()
        except Exception as e:
            return jsonify({"error": "data not available or invalid", "details": str(e)}), 500

        # Last 10 scaled values for the model input
        last_10_scaled = scaled[-10:].reshape(1, 10, 1)
        # Corresponding last 10 raw wave-height values (already cleaned/interpolated)
        recent_wave_heights = wave.tail(10)["WVHT"].astype(float).tolist()

        # Use the scaled window as input to the LSTM model
        pred = model.predict(last_10_scaled, verbose=0)
        pred_value = scaler.inverse_transform(pred)
        wave_height = float(pred_value[0][0])
        
        if np.isnan(wave_height) or np.isinf(wave_height):
            return (
                jsonify(
                    {
                        "error": "prediction failed",
                        "details": "Model returned NaN/Inf value",
                    }
                ),
                500,
            )

        return jsonify(
            {
                "wave_height": round(wave_height, 2),
                # Send the last 10 values the model looked at so the frontend
                # can show them in a list/graph.
                "recent_wave_heights": recent_wave_heights,
            }
        )
    except Exception as e:
        return jsonify({"error": "prediction failed", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
