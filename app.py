import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.express as px
from streamlit_lottie import st_lottie
import json
import time

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
# Weather Fetch Function (UNCHANGED)
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
# PREMIUM UI CSS
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(10,20,40,0.92), rgba(0,0,0,0.95)),
                url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}

.main-title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #00ffcc, #00ccff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.glass {
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.2);
}

div.stButton > button {
    background: linear-gradient(90deg, #00ffcc, #00ccff);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown('<div class="main-title">🌾 AgriIntel AI</div>', unsafe_allow_html=True)
st.markdown("### Smart Crop & Fertilizer Recommendation System")
st.markdown("---")

# -------------------------
# INPUT SECTION
# -------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌱 Soil Nutrients")
    N = st.number_input("Nitrogen (N)", 0, 200)
    P = st.number_input("Phosphorus (P)", 0, 200)
    K = st.number_input("Potassium (K)", 0, 200)
    ph = st.number_input("pH", 0.0, 14.0)

with col2:
    st.subheader("🌍 Location & Weather")
    location = st.text_input("Enter Location (e.g., New Delhi)")

predict = st.button("🚀 Predict Optimal Crop")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# PREDICTION
# -------------------------
if predict:

    try:
        with st.spinner("Fetching real-time weather data... 🌦️"):
            temp, humidity, rainfall = fetch_weather(location)
            time.sleep(1)

        # Weather Card
        st.markdown(f"""
        <div class="glass">
        🌡 Temperature: {temp} °C &nbsp;&nbsp;&nbsp;
        💧 Humidity: {humidity}% &nbsp;&nbsp;&nbsp;
        🌧 Rainfall: {rainfall} mm
        </div>
        """, unsafe_allow_html=True)

        # ---------------- Crop Prediction ----------------
        crop_input = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        crop_probs = crop_model.predict_proba(crop_input)[0]
        top3_idx = np.argsort(crop_probs)[-3:][::-1]
        top3_crops = crop_encoder.inverse_transform(top3_idx)
        top3_conf = crop_probs[top3_idx]

        st.markdown("## 🌾 Top 3 Crop Recommendations")

        for crop, conf in zip(top3_crops, top3_conf):
            st.progress(float(conf))
            st.write(f"**{crop}** — {conf*100:.2f}%")

        # ---------------- Fertilizer ----------------
        crop_encoded = fert_crop_encoder.transform([top3_crops[0]])[0]
        fert_input = np.array([[N, P, K, temp, humidity, rainfall, crop_encoded]])
        fert_probs = fert_model.predict_proba(fert_input)[0]
        fert_idx = np.argmax(fert_probs)
        fert_label = fert_encoder.inverse_transform([fert_idx])[0]
        fert_conf = fert_probs[fert_idx]

        st.markdown("## 💊 Recommended Fertilizer")
        st.success(f"{fert_label} — {fert_conf*100:.2f}% confidence")

        # ---------------- Feature Importance ----------------
        st.markdown("## 📊 Crop Model Feature Importance")
        importances = crop_model.calibrated_classifiers_[0].estimator.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": metadata["crop_features"],
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            color="Importance",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.error("Error fetching weather. Check location or API key in Streamlit secrets.")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("© 2026 AgriIntel AI | Built with ❤️ using Machine Learning & Streamlit")
