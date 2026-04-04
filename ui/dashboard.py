import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, time  # ✅ FIXED: Added imports
import time


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
# Page config
st.set_page_config(
    page_title="🚚 Delivery Time Predictor", 
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Delivery Time Prediction Dashboard")
st.markdown("***Production MLOps Demo - GPS → Auto-Features → ML Prediction***")

# Sidebar
st.sidebar.header("🎯 Quick Demo")
if st.sidebar.button("🔥 LIVE GPS Prediction", use_container_width=True):
    st.sidebar.success("Demo running!")

# API Base URL
API_URL = "http://localhost:8000"

# === MAIN DASHBOARD ===
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📱 Real-World Order Input")
    
    with st.form("order_form", clear_on_submit=True):
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

        st.info("💡 Tip: Get coordinates from Google Maps (Right click → copy lat/lng)")

        customer_lat = st.number_input("Customer Latitude", value=19.0760, step=0.0001, key="customer_lat")
        customer_lng = st.number_input("Customer Longitude", value=72.8777, step=0.0001, key="customer_lng")
        if st.button("📍 Use Default Mumbai Location"):
            customer_lat = 19.0760
            customer_lng = 72.8777
        order_hour = st.slider("Order Hour (24h)", 0, 23, 18, key="hour")
        prep_time = st.slider("Prep Time (mins)", 5, 45, 15)
        
        predict_btn = st.form_submit_button("🚀 PREDICT DELIVERY TIME", use_container_width=True)

with col2:
    if predict_btn:
        with st.spinner("🔮 Calling ML API..."):
            try:
                # Format time
                order_time_str = f"2024-01-15T{order_hour:02d}:30:00"
                
                # API Payload
                payload = {
                    "pickup_lat": pickup_lat,
                    "pickup_lng": pickup_lng,
                    "customer_lat": customer_lat,
                    "customer_lng": customer_lng,
                    "order_time": order_time_str,
                    "prep_time": prep_time
                }
                
                # Call YOUR API
                response = requests.post(f"{API_URL}/predict_real", json=payload, timeout=10)
                result = response.json()
                
                if response.status_code == 200:
                    st.success(f"✅ Predicted in {response.elapsed.total_seconds()*1000:.0f}ms!")
                    
                    prediction_data = result['data']
                    auto_features = prediction_data.get('auto_features', {})
                    
                    # === KEY METRICS ===
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Delivery Time", 
                                f"{prediction_data['predicted_delivery_time_min']} min",
                                f"{prediction_data['confidence_interval'][1]:.0f} max")
                    with col2:
                        st.metric("📏 Distance", f"{auto_features.get('Distance_km', 0):.1f} km")
                    with col3:
                        st.metric("🚦 Traffic", auto_features.get('Traffic_Level', '?'))
                    
                    # === LIVE MAP ===
                    st.subheader("🗺️ Delivery Route")
                    map_df = pd.DataFrame([
                        {"Location": "Restaurant", "lat": pickup_lat, "lon": pickup_lng},
                        {"Location": "Customer", "lat": customer_lat, "lon": customer_lng}
                    ])
                    
                    fig_map = px.scatter_mapbox(
                        map_df, lat="lat", lon="lon", hover_name="Location",
                        mapbox_style="open-street-map", zoom=12, height=350
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # === AUTO FEATURES TABLE ===
                    st.subheader("🪄 Auto-Generated ML Features")
                    features_df = pd.DataFrame([
                        ["Distance_km", f"{auto_features.get('Distance_km', 0):.1f} km"],
                        ["Weather", auto_features.get('Weather', '?')],
                        ["Traffic", auto_features.get('Traffic_Level', '?')],
                        ["Time", auto_features.get('Time_of_Day', '?')],
                        ["Vehicle", auto_features.get('Vehicle_Type', '?')]
                    ], columns=["Feature", "Value"])
                    st.dataframe(features_df, use_container_width=True)
                    
                    # === RECOMMENDATIONS ===
                    st.subheader("💡 Actionable Insights")
                    for i, suggestion in enumerate(prediction_data['suggestions']):
                        st.info(f"**{i+1}.** {suggestion}")
                    
                    # === SCENARIOS ===
                    st.subheader("🎯 Optimization Scenarios")
                    scenarios_df = pd.DataFrame(prediction_data['scenario_comparison'])
                    st.dataframe(scenarios_df.style.highlight_max('savings'), use_container_width=True)
                    
                    # === VISUALIZATIONS ===
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_scenarios = px.bar(scenarios_df, x='name', y='time',
                                             title="Time Savings Comparison",
                                             color='savings', height=300)
                        st.plotly_chart(fig_scenarios, use_container_width=True)
                    
                    with col2:
                        conf_low, conf_high = prediction_data['confidence_interval']
                        st.metric("Confidence Range", f"{conf_low:.0f} - {conf_high:.0f} mins")
                
                else:
                    st.error(f"API Error {response.status_code}")
                    st.json(result)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Ensure API running: `uvicorn api.app:app --reload`")

# === STATUS ===
st.markdown("---")
st.header("🖥️ System Status")

col1, col2, col3 = st.columns(3)
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=3)
    health = health_resp.json()
    col1.metric("API Health", "🟢 Healthy" if health_resp.status_code == 200 else "🔴 Down")
    col2.metric("Model", health.get('model', 'Loaded'))
    col3.metric("Response", f"{health_resp.elapsed.total_seconds()*1000:.0f}ms")
except:
    col1.metric("API Health", "🔴 Unreachable")
    col2.metric("Model", "❌ Offline")
    col3.metric("Response", "N/A")

# Footer
st.markdown("---")
st.markdown("""
<center>
<h4>🎓 Final Year MLOps Project</h4>
<p><b>GPS → Auto-Features → ML Ensemble → Business Insights</b></p>
<p>FastAPI + MLflow + Streamlit + scikit-learn | Production Ready ⚡</p>
</center>
""")
