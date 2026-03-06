import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.express as px

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
# Weather API
# -------------------------
def fetch_weather(city):
    try:
        api_key = st.secrets["weather_api"]

        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"

        response = requests.get(url).json()

        temp = response["current"]["temp_c"]
        humidity = response["current"]["humidity"]
        rainfall = response["current"]["precip_mm"]

        return temp, humidity, rainfall

    except:
        return None


# -------------------------
# UI CSS
# -------------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)),
url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854");
background-size: cover;
}

.main-title{
font-size:48px;
font-weight:800;
text-align:center;
color:#1b4332;
}

.weather-card{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0 10px 30px rgba(0,0,0,0.1);
margin-top:20px;
}

.weather-grid{
display:flex;
justify-content:space-around;
text-align:center;
}

.weather-box{
padding:10px;
}

.weather-value{
font-size:28px;
font-weight:bold;
}

.weather-label{
font-size:14px;
color:gray;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown('<div class="main-title">🌾 AgriIntel AI</div>', unsafe_allow_html=True)

st.markdown("### Smart Crop & Fertilizer Recommendation System")

st.divider()

# -------------------------
# Input Section
# -------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("🌱 Soil Nutrients")

    N = st.number_input("Nitrogen (N)", 0, 200, 40)
    P = st.number_input("Phosphorus (P)", 0, 200, 40)
    K = st.number_input("Potassium (K)", 0, 200, 40)

    ph = st.number_input("Soil pH", 0.0, 14.0, 7.0)

with col2:

    st.subheader("🌍 Location")

    location = st.text_input("Enter city (example: Delhi, Indore, Jaipur)", "Delhi")

predict = st.button("🚀 Predict Optimal Crop")

# -------------------------
# Prediction
# -------------------------

if predict:

    weather = fetch_weather(location)

    if weather:

        temp, humidity, rainfall = weather

        # Weather Card
        st.markdown(f"""
        <div class="weather-card">

        <h3 style="text-align:center">🌤 Live Weather</h3>

        <div class="weather-grid">

        <div class="weather-box">
        <div class="weather-value">🌡 {temp}°C</div>
        <div class="weather-label">Temperature</div>
        </div>

        <div class="weather-box">
        <div class="weather-value">💧 {humidity}%</div>
        <div class="weather-label">Humidity</div>
        </div>

        <div class="weather-box">
        <div class="weather-value">🌧 {rainfall} mm</div>
        <div class="weather-label">Rainfall</div>
        </div>

        </div>

        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("Weather fetch failed. Check city name or API key.")


    # -------------------------
    # Crop Prediction
    # -------------------------

    crop_input = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    crop_probs = crop_model.predict_proba(crop_input)[0]

    top3_idx = np.argsort(crop_probs)[-3:][::-1]

    top3_crops = crop_encoder.inverse_transform(top3_idx)

    top3_conf = crop_probs[top3_idx]

    st.subheader("🌾 Top Crop Recommendations")

    for crop, conf in zip(top3_crops, top3_conf):

        st.progress(float(conf))

        st.write(f"**{crop} — {conf*100:.2f}% confidence**")


    # -------------------------
    # Fertilizer Prediction
    # -------------------------

    crop_encoded = fert_crop_encoder.transform([top3_crops[0]])[0]

    fert_input = np.array([[N, P, K, temp, humidity, rainfall, crop_encoded]])

    fert_probs = fert_model.predict_proba(fert_input)[0]

    fert_idx = np.argmax(fert_probs)

    fert_label = fert_encoder.inverse_transform([fert_idx])[0]

    fert_conf = fert_probs[fert_idx]

    st.subheader("💊 Recommended Fertilizer")

    st.success(f"{fert_label} — {fert_conf*100:.2f}% confidence")


    # -------------------------
    # Feature Importance
    # -------------------------

    st.subheader("📊 Feature Importance")

    importances = crop_model.calibrated_classifiers_[0].estimator.feature_importances_

    importance_df = pd.DataFrame({

        "Feature": metadata["crop_features"],
        "Importance": importances

    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        color="Importance"
    )

    st.plotly_chart(fig, width="stretch")


# -------------------------
# Footer
# -------------------------

st.divider()

st.caption("AgriIntel AI • Crop & Fertilizer Recommendation System • Powered by Machine Learning")
