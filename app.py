# ─────────────────────────────────────────────────────────────
# app.py  —  F1 Race Predictor (deployment version)
# Works both locally AND when hosted online on Render
# ─────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Load model ────────────────────────────────────────────────
print("Loading model...")
model          = pickle.load(open("f1_best_model.pkl", "rb"))
driver_encoder = pickle.load(open("driver_encoder.pkl", "rb"))
team_encoder   = pickle.load(open("team_encoder.pkl",   "rb"))
print("Model loaded! Server ready.")

FEATURE_COLS = [
    "GridPos", "QualiPos", "BestQualiTime", "QualiGapToPole",
    "DriverRaceCount", "DriverRecentForm", "TeamAvgFinish",
    "WetXGrid", "TempMax", "RainfallMM", "WindSpeedMax",
    "HumidityMax", "IsWet", "SeasonProgress",
    "DriverEncoded", "TeamEncoded",
]

TEAM_AVG = {
    "Red Bull Racing": 2.1,  "Mercedes": 3.5,
    "Ferrari": 3.8,          "McLaren": 5.2,
    "Aston Martin": 6.1,     "Alpine": 9.4,
    "Williams": 13.2,        "AlphaTauri": 11.0,
    "Alfa Romeo": 12.1,      "Haas": 13.8,
}

DRIVER_FORM = {
    "VER":1.8, "PER":4.2, "HAM":4.5, "RUS":5.1,
    "LEC":4.8, "SAI":5.3, "NOR":5.8, "PIA":8.1,
    "ALO":6.5, "STR":9.2, "GAS":9.8, "OCO":10.1,
    "TSU":10.5,"HUL":11.2,"MAG":12.1,"BOT":12.5,
    "ZHO":13.2,"ALB":13.8,"SAR":17.1,"DEV":15.0,
}

# ── Serve the HTML frontend ───────────────────────────────────
# This means visiting your URL shows the website directly!
@app.route("/")
def index():
    return send_from_directory(".", "f1_predictor_v2.html")

# ── Predict endpoint ──────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data      = request.get_json()
        driver    = data.get("driver",   "VER")
        team      = data.get("team",     "Red Bull Racing")
        grid_pos  = int(data.get("gridPos",   1))
        quali_pos = int(data.get("qualiPos",  1))
        quali_gap = float(data.get("qualiGap", 0.0))
        temp      = float(data.get("temp",     25.0))
        rain      = float(data.get("rain",     0.0))
        wind      = float(data.get("wind",     15.0))
        humid     = float(data.get("humidity", 55.0))
        is_wet    = int(data.get("isWet",      0))

        try: driver_enc = int(driver_encoder.transform([driver])[0])
        except: driver_enc = 0
        try: team_enc = int(team_encoder.transform([team])[0])
        except: team_enc = 0

        row = pd.DataFrame([{
            "GridPos":          grid_pos,
            "QualiPos":         quali_pos,
            "BestQualiTime":    88.0 + quali_gap,
            "QualiGapToPole":   quali_gap,
            "DriverRaceCount":  50,
            "DriverRecentForm": DRIVER_FORM.get(driver, 10.0),
            "TeamAvgFinish":    TEAM_AVG.get(team, 10.0),
            "WetXGrid":         is_wet * grid_pos,
            "TempMax":          temp,
            "RainfallMM":       rain,
            "WindSpeedMax":     wind,
            "HumidityMax":      humid,
            "IsWet":            is_wet,
            "SeasonProgress":   0.5,
            "DriverEncoded":    driver_enc,
            "TeamEncoded":      team_enc,
        }])[FEATURE_COLS]

        prediction = float(model.predict(row)[0])
        prediction = max(1.0, min(20.0, prediction))
        confidence = max(40.0, min(92.0, 92.0 - quali_gap*8 - (10 if is_wet else 0)))

        return jsonify({
            "success":    True,
            "position":   round(prediction, 1),
            "confidence": round(confidence, 1),
            "driver":     driver,
            "team":       team,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Full grid endpoint ────────────────────────────────────────
@app.route("/predict-grid", methods=["POST"])
def predict_grid():
    try:
        data   = request.get_json()
        temp   = float(data.get("temp",     25.0))
        rain   = float(data.get("rain",     0.0))
        wind   = float(data.get("wind",     15.0))
        humid  = float(data.get("humidity", 55.0))
        is_wet = int(data.get("isWet",      0))

        drivers = [
            ("VER","Red Bull Racing"), ("PER","Red Bull Racing"),
            ("HAM","Mercedes"),        ("RUS","Mercedes"),
            ("LEC","Ferrari"),         ("SAI","Ferrari"),
            ("NOR","McLaren"),         ("PIA","McLaren"),
            ("ALO","Aston Martin"),    ("STR","Aston Martin"),
            ("GAS","Alpine"),          ("OCO","Alpine"),
            ("TSU","AlphaTauri"),      ("HUL","Haas"),
            ("MAG","Haas"),            ("BOT","Alfa Romeo"),
            ("ZHO","Alfa Romeo"),      ("ALB","Williams"),
            ("SAR","Williams"),        ("DEV","AlphaTauri"),
        ]

        results = []
        for i, (driver, team) in enumerate(drivers):
            grid_pos  = i + 1
            quali_gap = i * 0.08
            try: driver_enc = int(driver_encoder.transform([driver])[0])
            except: driver_enc = 0
            try: team_enc = int(team_encoder.transform([team])[0])
            except: team_enc = 0

            row = pd.DataFrame([{
                "GridPos":          grid_pos,
                "QualiPos":         grid_pos,
                "BestQualiTime":    88.0 + quali_gap,
                "QualiGapToPole":   quali_gap,
                "DriverRaceCount":  50,
                "DriverRecentForm": DRIVER_FORM.get(driver, 10.0),
                "TeamAvgFinish":    TEAM_AVG.get(team, 10.0),
                "WetXGrid":         is_wet * grid_pos,
                "TempMax":          temp,
                "RainfallMM":       rain,
                "WindSpeedMax":     wind,
                "HumidityMax":      humid,
                "IsWet":            is_wet,
                "SeasonProgress":   0.5,
                "DriverEncoded":    driver_enc,
                "TeamEncoded":      team_enc,
            }])[FEATURE_COLS]

            pred = float(model.predict(row)[0])
            pred = max(1.0, min(20.0, pred))
            results.append({
                "driver": driver, "team": team,
                "gridPos": grid_pos, "predicted": round(pred, 1),
            })

        results.sort(key=lambda x: x["predicted"])
        return jsonify({"success": True, "grid": results})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*45}")
    print(f"  F1 Race Predictor API")
    print(f"  Running at http://localhost:{port}")
    print(f"{'='*45}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
