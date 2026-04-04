from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import uvicorn
import sys
import os
import pandas as pd

# Fix import path
sys.path.append("src")
from predict import predict_delivery_time, predict_real_world, AutoFeatureEngineer

app = FastAPI(
    title="🚚 Delivery Time Predictor API",
    description="Production MLOps: GPS → Auto-Features → ML Prediction",
    version="2.1.0"
)

# Input models
class ManualFeatures(BaseModel):
    Distance_km: float = Field(..., ge=0, description="Distance in KM")
    Weather: str = Field(..., description="Clear/Cloudy/Rainy/Snowy")
    Traffic_Level: str = Field(..., description="Low/Medium/High")
    Time_of_Day: str = Field(..., description="Morning/Afternoon/Evening/Night")
    Vehicle_Type: str = Field(..., description="Bike/Car/Truck")
    Preparation_Time_min: float = Field(..., ge=1, le=60)
    Courier_Experience_yrs: float = Field(..., ge=0, le=20)

class GPSOrder(BaseModel):
    pickup_lat: float = Field(..., ge=-90, le=90, description="Restaurant latitude")
    pickup_lng: float = Field(..., ge=-180, le=180, description="Restaurant longitude")
    customer_lat: float = Field(..., ge=-90, le=90, description="Customer latitude")
    customer_lng: float = Field(..., ge=-180, le=180, description="Customer longitude")
    order_time: str = Field(..., description="ISO format: 2024-01-15T18:30:00")
    prep_time: float = Field(15, ge=5, le=45, description="Kitchen prep minutes")
    courier_exp: float = Field(2, ge=0, le=20, description="Courier experience years")

@app.get("/")
async def home():
    return {
        "🚀": "Delivery Time ML API - Production Ready",
        "endpoints": {
            "POST /predict": "Manual features (legacy)",
            "POST /predict_real": "GPS auto-features (production)",
            "GET /health": "Health check"
        },
        "docs": "/docs",
        "model": "RandomForest + GradientBoosting Ensemble"
    }

@app.post("/predict")
async def predict_manual(features: ManualFeatures):
    """📊 Legacy: Manual ML features input"""
    try:
        result = predict_delivery_time(features.dict())
        return {
            "success": True,
            "method": "manual",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual prediction failed: {str(e)}")

@app.post("/predict_real")
async def predict_gps(raw_order: GPSOrder):
    """🚀 Production: GPS coordinates → Auto-generate features → Predict"""
    try:
        # Convert to dict for your predict_real_world function
        raw_dict = raw_order.dict()
        result = predict_real_world(raw_dict)
        return {
            "success": True,
            "method": "gps_auto",
            "distance_calculated": f"{result.get('auto_features', {}).get('Distance_km', 'N/A')} km",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPS prediction failed: {str(e)}")

@app.get("/health")
async def health():
    """❤️ Production health check"""
    try:
        # Quick model test
        test_features = {
            "Distance_km": 10, "Weather": "Clear", "Traffic_Level": "Medium",
            "Time_of_Day": "Evening", "Vehicle_Type": "Car",
            "Preparation_Time_min": 15, "Courier_Experience_yrs": 2
        }
        _ = predict_delivery_time(test_features)
        return {
            "status": "🟢 HEALTHY",
            "model": "loaded",
            "test_prediction": "success",
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model unhealthy: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Delivery Predictor API...")
    print("📱 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)