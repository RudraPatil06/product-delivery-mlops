import joblib
import pandas as pd
from datetime import datetime

# Load trained model
model = joblib.load("delivery_time_model.pkl")

def auto_generate_features():
    """
    Rule-based automated feature generation
    (Simulates backend systems)
    """

    now = datetime.now()
    hour = now.hour

    # Time of day logic
    if hour < 10:
        time_of_day = "Morning"
        traffic = "Low"
    elif hour < 17:
        time_of_day = "Afternoon"
        traffic = "High"
    elif hour < 21:
        time_of_day = "Evening"
        traffic = "High"
    else:
        time_of_day = "Night"
        traffic = "Medium"

    # Weather rule (static for now, API later)
    weather = "Clear"

    # Vehicle logic
    vehicle = "Bike" if traffic == "High" else "Car"

    # Order processing time (warehouse rule)
    order_processing_time = 15 if time_of_day in ["Afternoon", "Evening"] else 10

    # Courier experience (assigned by system)
    courier_experience = 3

    # Distance (from warehouse → customer, fixed here)
    distance_km = 12.0

    return {
    "Distance_km": distance_km,
    "Weather": weather,
    "Traffic_Level": traffic,
    "Time_of_Day": time_of_day,
    "Vehicle_Type": vehicle,
    "Preparation_Time_min": order_processing_time,  
    "Courier_Experience_yrs": courier_experience
}


# Generate automated features
features = auto_generate_features()
input_df = pd.DataFrame([features])

# Predict
prediction = model.predict(input_df)

print("\n--- Automated Delivery Prediction (Rule-Based) ---")
for k, v in features.items():
    print(f"{k}: {v}")

print(f"\nEstimated Delivery Time: {prediction[0]:.2f} minutes\n")
