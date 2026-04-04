from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys
sys.path.append('src')
from predict import predict_delivery_time, predict_real_world

app = FastAPI(title="Delivery Time API")

class GPSOrder(BaseModel):
    pickup_lat: float = 12.97
    pickup_lng: float = 77.59
    customer_lat: float = 12.93
    customer_lng: float = 77.62
    order_time: str = "2024-01-15T18:30:00"
    prep_time: float = 15

@app.get("/")
def home():
    return {"status": "🚀 Delivery API Ready!", "docs": "/docs"}

@app.post("/predict_real")
def predict_gps(order: GPSOrder):
    result = predict_real_world(order.dict())
    return {"success": True, "data": result}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    