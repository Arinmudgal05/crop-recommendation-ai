import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import time

st.set_page_config(page_title="AgriIntel AI", layout="wide")

# -------------------------
# Load Models
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
# Weather Function (Safe)
# -------------------------
def fetch_weather(city):

    api_key = st.secrets.get("weather_api")

    if not api_key:
        st.error("Weather API key missing. Add it in Streamlit Secrets.")
        st.stop()

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)

    if response.status_code != 200:
        st.error("Weather API error. Check city name or API key.")
        st.stop()

    data = response.json()

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    rainfall = data.get("rain", {}).get("1h", 0)

    return temperature, humidity, rainfall

# -------------------------
# UI CSS
# -------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.9)),
                url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #2d6a4f;
}

.glass {
    background: rgba(255,255,255,0.75);
    padding: 30px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}

div.stButton > button {
    background: linear-gradient(90deg,#52b788,#40916c);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

div.stButton > button:hover {
    transform: scale(1.05);
}

.rain {
  position: fixed;
  width: 100%;
  height: 100%;
  pointer-events: none;
  background-image: url("https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy.gif");
  background-repeat: repeat;
  opacity: 0.2;
  top: 0;
  left: 0;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown('<div class="main-title">🌾 AgriIntel AI</div>', unsafe_allow_html=True)
st.markdown("### Smart Crop & Fertilizer Recommendation System")
st.markdown("---")

# -------------------------
# Input Section
# -------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌱 Soil Nutrients")

    N = st.number_input("Nitrogen (N)", 0, 200)
    P = st.number_input("Phosphorus (P)", 0, 200)
    K = st.number_input("Potassium (K)", 0, 200)
    ph = st.number_input("Soil pH", 0.0, 14.0)

with col2:
    st.subheader("🌍 Location & Weather")

    location = st.text_input("Enter City (Example: New Delhi)")

predict = st.button("🚀 Predict Optimal Crop")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Prediction
# -------------------------
if predict:

    if location.strip() == "":
        st.warning("Please enter a location.")
        st.stop()

    st.markdown('<div class="rain"></div>', unsafe_allow_html=True)

    with st.spinner("Fetching weather data..."):
        temp, humidity, rainfall = fetch_weather(location)
        time.sleep(1)

    st.markdown(f"""
    <div class="glass">
    🌡 Temperature: {temp} °C &nbsp;&nbsp;&nbsp;
    💧 Humidity: {humidity}% &nbsp;&nbsp;&nbsp;
    🌧 Rainfall: {rainfall} mm
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # Crop Prediction
    # -------------------------
    crop_input = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    crop_probs = crop_model.predict_proba(crop_input)[0]

    top3_idx = np.argsort(crop_probs)[-3:][::-1]

    top3_crops = crop_encoder.inverse_transform(top3_idx)

    top3_conf = crop_probs[top3_idx]

    st.markdown("## 🌾 Top 3 Crop Recommendations")

    for crop, conf in zip(top3_crops, top3_conf):
        st.progress(float(conf))
        st.write(f"**{crop} — {conf*100:.2f}%**")

    # -------------------------
    # Fertilizer Recommendation
    # -------------------------
    crop_encoded = fert_crop_encoder.transform([top3_crops[0]])[0]

    fert_input = np.array([[N, P, K, temp, humidity, rainfall, crop_encoded]])

    fert_probs = fert_model.predict_proba(fert_input)[0]

    fert_idx = np.argmax(fert_probs)

    fert_label = fert_encoder.inverse_transform([fert_idx])[0]

    fert_conf = fert_probs[fert_idx]

    st.markdown("## 💊 Recommended Fertilizer")

    st.success(f"{fert_label} — {fert_conf*100:.2f}% confidence")

    # -------------------------
    # Feature Importance
    # -------------------------
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
        template="plotly_white"
    )

    st.plotly_chart(fig, width="stretch")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("© 2026 AgriIntel AI | Built with Machine Learning & Streamlit")
