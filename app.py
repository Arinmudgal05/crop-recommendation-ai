import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.express as px
from streamlit_lottie import st_lottie
import json

st.set_page_config(page_title="AgriIntel AI", layout="wide")

# -------------------------
# Load Models (Cached)
# -------------------------
@st.cache_resource
def load_models():
    crop_model = joblib.load("models/crop_model.pkl")
    crop_encoder = joblib.load("models/crop_encoder.pkl")
    fert_model = joblib.load("models/fertilizer_model.pkl")
    fert_encoder = joblib.load("models/fertilizer_encoder.pkl")
    fert_crop_encoder = joblib.load("models/fert_crop_encoder.pkl")
    metadata = joblib.load("models/metadata.pkl")

    return crop_model, crop_encoder, fert_model, fert_encoder, fert_crop_encoder, metadata

crop_model, crop_encoder, fert_model, fert_encoder, fert_crop_encoder, metadata = load_models()

# -------------------------
# Weather Fetch Function
# -------------------------
def fetch_weather(city):
    api_key = st.secrets["weather_api"]
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()

    temperature = response["main"]["temp"]
    humidity = response["main"]["humidity"]
    rainfall = response.get("rain", {}).get("1h", 0)

    return temperature, humidity, rainfall

# -------------------------
# UI Styling
# -------------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stButton>button {
    background-color: #00C9A7;
    color: black;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🌾 AgriIntel AI — Smart Crop & Fertilizer System")

# -------------------------
# Input Section
# -------------------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200)
    P = st.number_input("Phosphorus (P)", 0, 200)
    K = st.number_input("Potassium (K)", 0, 200)
    ph = st.number_input("pH", 0.0, 14.0)

with col2:
    location = st.text_input("Enter Location (e.g., New Delhi)")

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    try:
        temp, humidity, rainfall = fetch_weather(location)

        st.success(f"Weather in {location}: Temp={temp}°C | Humidity={humidity}% | Rainfall={rainfall}mm")

        # Crop Prediction
        crop_input = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        crop_probs = crop_model.predict_proba(crop_input)[0]
        top3_idx = np.argsort(crop_probs)[-3:][::-1]
        top3_crops = crop_encoder.inverse_transform(top3_idx)
        top3_conf = crop_probs[top3_idx]

        st.subheader("🌾 Top 3 Crop Recommendations")
        for crop, conf in zip(top3_crops, top3_conf):
            st.progress(float(conf))
            st.write(f"{crop} — {conf*100:.2f}%")

        # Fertilizer Prediction
        crop_encoded = fert_crop_encoder.transform([top3_crops[0]])[0]
        fert_input = np.array([[N, P, K, temp, humidity, rainfall, crop_encoded]])
        fert_probs = fert_model.predict_proba(fert_input)[0]
        fert_idx = np.argmax(fert_probs)
        fert_label = fert_encoder.inverse_transform([fert_idx])[0]
        fert_conf = fert_probs[fert_idx]

        st.subheader("💊 Recommended Fertilizer")
        st.success(f"{fert_label} — {fert_conf*100:.2f}% confidence")

        # Feature Importance
        st.subheader("📊 Crop Model Feature Importance")
        importances = crop_model.calibrated_classifiers_[0].estimator.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": metadata["crop_features"],
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(importance_df, x="Feature", y="Importance", color="Importance")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("Error fetching weather. Please check location or API key.")