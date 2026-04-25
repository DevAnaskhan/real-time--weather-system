"""
Flask REST API: live OpenWeatherMap data + next-hour temperature prediction.
Set OWM_API_KEY in the environment (or a .env file via python-dotenv).
"""

import os
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

OWM_URL = "https://api.openweathermap.org/data/2.5/weather"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Missing {MODEL_PATH}. Run: python train_model.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def fetch_openweather(city: str, api_key: str) -> dict:
    params = {
        "q": city.strip(),
        "appid": api_key,
        "units": "metric",
    }
    resp = requests.get(OWM_URL, params=params, timeout=15)
    if resp.status_code != 200:
        try:
            err = resp.json().get("message", resp.text)
        except Exception:
            err = resp.text
        raise ValueError(err or f"OpenWeatherMap error {resp.status_code}")
    data = resp.json()
    main = data.get("main", {})
    weather = (data.get("weather") or [{}])[0]
    return {
        "city": data.get("name", city),
        "country": (data.get("sys") or {}).get("country", ""),
        "temp_c": float(main["temp"]),
        "feels_like_c": float(main.get("feels_like", main["temp"])),
        "humidity": float(main["humidity"]),
        "pressure_hpa": float(main["pressure"]),
        "description": weather.get("description", "").title(),
    }


def predict_next_hour_temp(temp_c: float, humidity: float, pressure_hpa: float) -> float:
    model = get_model()
    X = pd.DataFrame(
        [{"temp_c": temp_c, "humidity": humidity, "pressure_hpa": pressure_hpa}]
    )
    pred = float(model.predict(X)[0])
    return round(pred, 2)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.is_file()})


@app.get("/api/weather")
def weather_query():
    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({"error": "Missing query parameter: city"}), 400
    api_key = os.environ.get("OWM_API_KEY", "").strip()
    if not api_key:
        return jsonify(
            {"error": "Set environment variable OWM_API_KEY (OpenWeatherMap API key)."}
        ), 503
    try:
        current = fetch_openweather(city, api_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.RequestException as e:
        return jsonify({"error": f"Weather service unreachable: {e}"}), 502

    try:
        predicted = predict_next_hour_temp(
            current["temp_c"], current["humidity"], current["pressure_hpa"]
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    return jsonify(
        {
            "city": current["city"],
            "country": current["country"],
            "current": {
                "temp_c": current["temp_c"],
                "feels_like_c": current["feels_like_c"],
                "humidity_pct": current["humidity"],
                "pressure_hpa": current["pressure_hpa"],
                "description": current["description"],
            },
            "predicted_next_hour_temp_c": predicted,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
