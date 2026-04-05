import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import random

# === HELPER FUNCTION ===
def get_nearby_location(lat, lng):
    return lat + random.uniform(-0.02, 0.02), lng + random.uniform(-0.02, 0.02)

# === RESTAURANTS ===
bangalore_restaurants = {
    "Domino's - MG Road": (12.9716, 77.5946),
    "KFC - Indiranagar": (12.9784, 77.6408),
    "McDonald's - Whitefield": (12.9698, 77.7500)
}

# === PAGE CONFIG ===
st.set_page_config(page_title="🚚 Delivery Predictor", layout="wide")

st.title("🚀 Delivery Time Prediction Dashboard")

# === API ===
API_URL = "https://delivery-mlops.onrender.com"

# === AUTO CUSTOMER BUTTON ===
st.markdown("### 📍 Smart Location")

if st.button("📍 Auto Nearby Customer Location"):
    lat, lng = get_nearby_location(
        st.session_state.get("pickup_lat", 12.9716),
        st.session_state.get("pickup_lng", 77.5946)
    )
    st.session_state["customer_lat"] = lat
    st.session_state["customer_lng"] = lng
    st.success("✅ Customer location generated near restaurant")

# === LAYOUT ===
col1, col2 = st.columns([1, 1])

# ================= INPUT =================
with col1:
    st.header("📱 Order Input")

    with st.form("order_form"):
        city = st.selectbox("🌆 Select City", ["Bangalore (Recommended)"])

        restaurants = bangalore_restaurants

        selected_restaurant = st.selectbox("🏪 Select Restaurant", list(restaurants.keys()))
        pickup_lat, pickup_lng = restaurants[selected_restaurant]

        # ✅ STORE PICKUP
        st.session_state["pickup_lat"] = pickup_lat
        st.session_state["pickup_lng"] = pickup_lng

        st.write(f"📍 Restaurant Location: {pickup_lat}, {pickup_lng}")

        st.subheader("👤 Customer Location")

        customer_lat = st.number_input(
            "Customer Latitude",
            value=st.session_state.get("customer_lat", pickup_lat),
            step=0.0001
        )

        customer_lng = st.number_input(
            "Customer Longitude",
            value=st.session_state.get("customer_lng", pickup_lng),
            step=0.0001
        )

        order_hour = st.slider("Order Hour", 0, 23, 18)
        prep_time = st.slider("Prep Time", 5, 45, 15)

        predict_btn = st.form_submit_button("🚀 Predict Delivery Time")

# ================= OUTPUT =================
with col2:
    if predict_btn:
        with st.spinner("🔮 Predicting..."):
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
                    data = result["data"]
                    auto = data.get("auto_features", {})

                    st.success("✅ Prediction Successful")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("⏱ Delivery Time", f"{data['predicted_delivery_time_min']} min")
                    c2.metric("📏 Distance", f"{auto.get('Distance_km', 0):.1f} km")
                    c3.metric("🚦 Traffic", auto.get("Traffic_Level", "?"))

                    # === MAP ===
                    st.subheader("🗺️ Route")
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

                else:
                    st.error("❌ API Error")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# === STATUS ===
st.markdown("---")
st.header("🖥️ System Status")

try:
    requests.get(f"{API_URL}/health")
    st.success("🟢 API Running")
except:
    st.error("🔴 API Down")

st.markdown("---")
st.markdown("🎓 Final Year MLOps Project")
