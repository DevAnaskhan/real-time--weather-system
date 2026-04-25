"""
Streamlit UI: enter a city and compare current weather with ML-predicted next-hour temperature.

Start the API first:  python api.py
Then run:             streamlit run streamlit_app.py
"""

import os

import requests
import streamlit as st

DEFAULT_API = os.environ.get("WEATHER_API_URL", "http://127.0.0.1:5000").rstrip("/")

st.set_page_config(page_title="Weather & forecast", page_icon="🌤️", layout="centered")
st.title("Live weather & next-hour temperature")
st.caption("Current conditions from OpenWeatherMap; next-hour estimate from a trained model.")

api_base = st.sidebar.text_input("API base URL", value=DEFAULT_API)

city = st.text_input("City name", placeholder="e.g. Delhi", value="")

if st.button("Get weather & prediction", type="primary"):
    if not city.strip():
        st.warning("Please enter a city name.")
    else:
        url = f"{api_base}/api/weather"
        try:
            r = requests.get(url, params={"city": city.strip()}, timeout=30)
            data = r.json()
        except requests.RequestException as e:
            st.error(f"Could not reach API: {e}")
            st.stop()

        if r.status_code != 200:
            st.error(data.get("error", r.text))
            st.stop()

        cur = data["current"]
        pred = data["predicted_next_hour_temp_c"]
        loc = f"{data['city']}"
        if data.get("country"):
            loc += f", {data['country']}"

        st.subheader(loc)
        st.write(cur["description"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature (°C)", f"{cur['temp_c']:.1f}")
        c2.metric("Feels like (°C)", f"{cur['feels_like_c']:.1f}")
        c3.metric("Humidity (%)", f"{cur['humidity_pct']:.0f}")
        c4.metric("Pressure (hPa)", f"{cur['pressure_hpa']:.0f}")

        delta = pred - cur["temp_c"]
        st.metric(
            "Predicted temp next hour (°C)",
            f"{pred:.2f}",
            delta=f"{delta:+.2f} vs now",
            help="Model trained on synthetic patterns from temp, humidity, and pressure — not an official forecast.",
        )

st.sidebar.markdown(
   """
**Setup**
1. `export OWM_API_KEY=...` (or use `.env`)
2. `python train_model.py`
3. `python api.py`
4. `streamlit run streamlit_app.py`
"""
)
