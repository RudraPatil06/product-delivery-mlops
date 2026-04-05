import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# === CITY RESTAURANT DATA ===
bangalore_restaurants = {
    "Domino's - MG Road": (12.9716, 77.5946),
    "KFC - Indiranagar": (12.9784, 77.6408),
    "McDonald's - Whitefield": (12.9698, 77.7500)
}

mumbai_restaurants = {
    "Domino's - Andheri": (19.1197, 72.8468),
    "McDonald's - Bandra": (19.0596, 72.8295),
    "KFC - Powai": (19.1176, 72.9060),
    "Pizza Hut - Dadar": (19.0176, 72.8562)
}

# === PAGE CONFIG ===
st.set_page_config(
    page_title="🚚 Delivery Time Predictor",
    page_icon="🚚",
    layout="wide"
)

st.title("🚀 Delivery Time Prediction Dashboard")
st.markdown("***Production MLOps Demo - GPS → Auto-Features → ML Prediction***")

# === SIDEBAR ===
st.sidebar.header("🎯 Quick Demo")
if st.sidebar.button("🔥 LIVE GPS Prediction"):
    st.sidebar.success("Demo running!")

# === API ===
API_URL = "https://delivery-mlops.onrender.com"

# === DEFAULT LOCATION BUTTON (OUTSIDE FORM) ===
st.markdown("### 📍 Quick Location")
if st.button("📍 Use Default Mumbai Location"):
    st.session_state["customer_lat"] = 19.0760
    st.session_state["customer_lng"] = 72.8777

# === MAIN LAYOUT ===
col1, col2 = st.columns([1, 1])

# ================= LEFT SIDE =================
with col1:
    st.header("📱 Real-World Order Input")

    with st.form("order_form"):
        city = st.selectbox("🌆 Select City", ["Bangalore (Accurate)", "Mumbai (Demo)"])

        st.subheader("🏪 Restaurant Location")
        if city == "Bangalore (Accurate)":
            restaurants = bangalore_restaurants
        else:
            restaurants = mumbai_restaurants
            st.warning("⚠️ Model trained on Bangalore data — Mumbai predictions are approximate")

        selected_restaurant = st.selectbox("🏪 Select Restaurant", list(restaurants.keys()))
        pickup_lat, pickup_lng = restaurants[selected_restaurant]

        st.write(f"📍 Restaurant Location: {pickup_lat}, {pickup_lng}")

        st.subheader("👤 Customer Location")
        st.info("💡 Tip: Copy coordinates from Google Maps")

        # ✅ SESSION STATE USED HERE
        customer_lat = st.number_input(
            "Customer Latitude",
            value=st.session_state.get("customer_lat", 19.0760),
            step=0.0001
        )

        customer_lng = st.number_input(
            "Customer Longitude",
            value=st.session_state.get("customer_lng", 72.8777),
            step=0.0001
        )

        order_hour = st.slider("Order Hour (24h)", 0, 23, 18)
        prep_time = st.slider("Prep Time (mins)", 5, 45, 15)

        predict_btn = st.form_submit_button("🚀 PREDICT DELIVERY TIME")

# ================= RIGHT SIDE =================
with col2:
    if predict_btn:
        with st.spinner("🔮 Calling ML API..."):
            try:
                order_time_str = f"2024-01-15T{order_hour:02d}:30:00"

                payload = {
                    "pickup_lat": pickup_lat,
                    "pickup_lng": pickup_lng,
                    "customer_lat": customer_lat,
                    "customer_lng": customer_lng,
                    "order_time": order_time_str,
                    "prep_time": prep_time
                }

                response = requests.post(f"{API_URL}/predict_real", json=payload)
                result = response.json()

                if response.status_code == 200:
                    st.success("✅ Prediction Successful!")

                    data = result['data']
                    auto = data.get('auto_features', {})

                    c1, c2, c3 = st.columns(3)
                    c1.metric("⏱ Delivery Time", f"{data['predicted_delivery_time_min']} min")
                    c2.metric("📏 Distance", f"{auto.get('Distance_km', 0):.1f} km")
                    c3.metric("🚦 Traffic", auto.get('Traffic_Level', '?'))

                    # === MAP ===
                    st.subheader("🗺️ Delivery Route")
                    map_df = pd.DataFrame([
                        {"Location": "Restaurant", "lat": pickup_lat, "lon": pickup_lng},
                        {"Location": "Customer", "lat": customer_lat, "lon": customer_lng}
                    ])

                    fig = px.scatter_mapbox(
                        map_df,
                        lat="lat",
                        lon="lon",
                        hover_name="Location",
                        mapbox_style="open-street-map",
                        zoom=12
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # === FEATURES ===
                    st.subheader("🪄 ML Features")
                    st.json(auto)

                else:
                    st.error("❌ API Error")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Run API first → uvicorn api.app:app --reload")

# === SYSTEM STATUS ===
st.markdown("---")
st.header("🖥️ System Status")

try:
    health = requests.get(f"{API_URL}/health").json()
    st.success("🟢 API Running")
except:
    st.error("🔴 API Not Running")

# === FOOTER ===
st.markdown("---")
st.markdown("🎓 Final Year MLOps Project | Streamlit + FastAPI + ML")