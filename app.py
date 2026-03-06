import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import time

st.set_page_config(page_title="AgriIntel AI", layout="wide")

# -------------------------
# LOAD MODELS
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
# WEATHER FUNCTION
# -------------------------
def fetch_weather(city):
    api_key = st.secrets.get("weather_api")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url).json()

    temperature = response["main"]["temp"]
    humidity = response["main"]["humidity"]
    rainfall = response.get("rain", {}).get("1h", 0)

    return temperature, humidity, rainfall


# -------------------------
# PREMIUM UI
# -------------------------

st.markdown("""
<style>

/* Background */
.stApp{
background-image:url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854");
background-size:cover;
background-attachment:fixed;
}

/* Title */
.title{
text-align:center;
font-size:55px;
font-weight:800;
color:#1b4332;
}

/* Subtitle */
.subtitle{
text-align:center;
font-size:22px;
color:#344e41;
margin-bottom:30px;
}

/* Card */
.card{
background:rgba(255,255,255,0.9);
padding:35px;
border-radius:18px;
box-shadow:0 8px 30px rgba(0,0,0,0.15);
}

/* INPUT LABELS */
label{
color:#1b4332 !important;
font-weight:600;
}

/* NUMBER INPUT TEXT */
input{
color:#000000 !important;
font-weight:600 !important;
}

/* NUMBER INPUT BOX */
div[data-baseweb="input"]{
background:white !important;
border-radius:8px !important;
}

/* TEXT INPUT */
textarea, input[type="text"]{
color:#000000 !important;
background:white !important;
}

/* BUTTON */
button[kind="primary"]{
background:#2d6a4f !important;
color:white !important;
font-weight:bold !important;
border-radius:10px;
}

/* BUTTON HOVER */
button[kind="primary"]:hover{
background:#1b4332 !important;
}

/* RAIN ANIMATION */
.rain{
position:fixed;
width:100%;
height:100%;
background:url("https://i.imgur.com/MK3eW3As.gif");
opacity:0.15;
pointer-events:none;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------

st.markdown('<div class="title">🌾 AgriIntel AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Crop & Fertilizer Recommendation System</div>', unsafe_allow_html=True)


# -------------------------
# INPUT CARD
# -------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    st.subheader("🌱 Soil Nutrients")

    N = st.number_input("Nitrogen (N)",0,200)
    P = st.number_input("Phosphorus (P)",0,200)
    K = st.number_input("Potassium (K)",0,200)
    ph = st.number_input("Soil pH",0.0,14.0)

with col2:

    st.subheader("🌍 Location")

    location = st.text_input("Enter city (example: Delhi)")


predict = st.button("🚀 Predict Optimal Crop")

st.markdown('</div>', unsafe_allow_html=True)


# -------------------------
# PREDICTION
# -------------------------

if predict:

    st.markdown('<div class="rain"></div>', unsafe_allow_html=True)

    try:

        with st.spinner("Fetching live weather..."):
            temp, humidity, rainfall = fetch_weather(location)
            time.sleep(1)

        st.success(f"Temperature: {temp}°C | Humidity: {humidity}% | Rainfall: {rainfall} mm")


        # Crop prediction

        crop_input = np.array([[N,P,K,temp,humidity,ph,rainfall]])

        crop_probs = crop_model.predict_proba(crop_input)[0]

        top3_idx = np.argsort(crop_probs)[-3:][::-1]

        top3_crops = crop_encoder.inverse_transform(top3_idx)

        top3_conf = crop_probs[top3_idx]


        st.subheader("🌾 Top Crop Recommendations")

        for crop,conf in zip(top3_crops,top3_conf):

            st.progress(float(conf))
            st.write(f"**{crop} — {conf*100:.2f}%**")



        # Fertilizer prediction

        crop_encoded = fert_crop_encoder.transform([top3_crops[0]])[0]

        fert_input = np.array([[N,P,K,temp,humidity,rainfall,crop_encoded]])

        fert_probs = fert_model.predict_proba(fert_input)[0]

        fert_idx = np.argmax(fert_probs)

        fert_label = fert_encoder.inverse_transform([fert_idx])[0]

        fert_conf = fert_probs[fert_idx]


        st.subheader("💊 Recommended Fertilizer")

        st.success(f"{fert_label} — {fert_conf*100:.2f}% confidence")



        # Feature importance

        st.subheader("📊 Feature Importance")

        importances = crop_model.calibrated_classifiers_[0].estimator.feature_importances_

        importance_df = pd.DataFrame({
            "Feature":metadata["crop_features"],
            "Importance":importances
        }).sort_values(by="Importance",ascending=False)

        fig = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            color="Importance"
        )

        st.plotly_chart(fig,width="stretch")

    except Exception:

        st.error("Weather fetch failed. Try 'Delhi' or 'Mumbai'.")

