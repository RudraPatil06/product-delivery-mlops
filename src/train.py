import mlflow
import mlflow.sklearn
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib
import torch
from datetime import datetime
from pathlib import Path

print("🚀 Starting Delivery Time Prediction Training...")

# Device check
def check_device():
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Running training on: {device}")
    return device

check_device()

# Load data
print("📥 Loading data...")
df = pd.read_csv("data/delivery_data.csv")
df = df.drop(columns=["Order_ID"])

# Feature Engineering
print("🔧 Feature Engineering...")
df["is_peak_hour"] = df["Time_of_Day"].apply(lambda x: 1 if x in ["Evening", "Night"] else 0)
df["distance_per_prep"] = df["Distance_km"] / (df["Preparation_Time_min"] + 1)
df["total_load"] = df["Distance_km"] * df["Preparation_Time_min"]

# Prepare data
X = df.drop(columns=["Delivery_Time_min"])
y = df["Delivery_Time_min"]

categorical_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
numerical_cols = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs", 
                 "is_peak_hour", "distance_per_prep", "total_load"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing Pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("cat", cat_pipeline, categorical_cols),
    ("num", num_pipeline, numerical_cols)
])

# Models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

# MLflow Experiment
mlflow.set_experiment("Delivery_Optimization_Project")
best_model = None
best_mae = float("inf")
best_run_id = None

print("\n🤖 Training Models...")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name) as run:
        print(f"Training {model_name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"  {model_name} MAE: {mae:.2f}")
        
        # Log to MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(
    pipeline,
    "model",
    pip_requirements=[
        "scikit-learn",
        "pandas",
        "numpy"
    ]
)
        
        # Track best model
        if mae < best_mae:
            best_mae = mae
            best_model = pipeline
            best_run_id = mlflow.active_run().info.run_id

print(f"\n🏆 Best Model MAE: {best_mae:.2f}")

# Quality Gate
MAX_MAE = 10.0
if best_mae > MAX_MAE:
    raise ValueError(f"Model failed quality gate! MAE={best_mae:.2f} > {MAX_MAE}")

# Save Artifacts
print("💾 Saving model...")
Path("models").mkdir(exist_ok=True)
joblib.dump(best_model, "models/delivery_time_model.pkl")

# Register Model
mlflow.register_model(f"runs:/{best_run_id}/model", "DeliveryTimeModel")

# Save Metrics
metrics = {
    "best_mae": round(best_mae, 4),
    "best_model": "GradientBoosting" if best_mae < 8 else "RandomForest",
    "timestamp": datetime.now().isoformat(),
    "mlflow_run_id": best_run_id
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Training COMPLETE!")
print("📊 View experiments: mlflow ui")
print("🎯 Model saved: models/delivery_time_model.pkl")
