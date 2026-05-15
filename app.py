import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

# Initialize FastAPI app
app = FastAPI(title="Telecom Customer Churn Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and features
MODEL_PATH = "models/xgb_churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/features.pkl"

def load_assets():
    """Load machine learning model, scaler, and feature list from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        sc = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        print("🚀 Models and features loaded successfully.")
        return model, sc, features
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None

xgb_model, scaler, model_features = load_assets()

# Input schema
class PredictionInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """
    Predict the churn probability for a given customer profile.
    
    Args:
        input_data (PredictionInput): Customer features including tenure and charges.
        
    Returns:
        dict: Churn probability rounded to 1 decimal place.
    """
    # 1. Convert input to dict and then DataFrame
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    
    # 2. Apply get_dummies
    df_encoded = pd.get_dummies(df)
    
    # 3. Reindex to match training features
    df_final = df_encoded.reindex(columns=model_features, fill_value=0)
    
    # 4. Transform using the loaded scaler
    scaled_data = scaler.transform(df_final)
    
    # 5. Predict probability
    prediction_prob = xgb_model.predict_proba(scaled_data)[0][1]
    
    # Round to 1 decimal point for percentage representation
    churn_probability = round(float(prediction_prob) * 100, 1)
    
    return {"churn_probability": churn_probability}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serve the dashboard frontend."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "index.html not found in static folder."

if __name__ == "__main__":
    import uvicorn
    # Defaulting to 8080 as per user terminal logs
    uvicorn.run(app, host="0.0.0.0", port=8080)
