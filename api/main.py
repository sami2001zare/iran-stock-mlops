from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
import joblib
from schemas import StockFeatures, PredictionResponse

app = FastAPI(title="Iran Stock Prediction API")

# Load model once at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    client = mlflow.tracking.MlflowClient()
    # Get latest production model
    try:
        latest_version = client.get_latest_versions("StockDirectionModel", stages=["Production"])[-1].version
        model_uri = f"models:/StockDirectionModel/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded from registry")
    except:
        # Fallback to local model
        model = joblib.load("models/latest_model.pkl")
        print("Loaded local model")

@app.post("/predict", response_model=PredictionResponse)
def predict(features: StockFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    # Convert input to DataFrame
    df = pd.DataFrame([features.dict()])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]  # probability of class 1
    return PredictionResponse(direction=int(pred), probability=float(proba))

@app.get("/health")
def health():
    return {"status": "ok"}