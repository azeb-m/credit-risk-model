import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk Prediction API")

# Load model from MLflow Registry
MODEL_NAME = "CreditRisk_Predictor"
MODEL_STAGE = "Production"

try:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert request to DataFrame with column names
        X = pd.DataFrame([request.dict()])

        # Predict probability
        prob = model.predict(X)[0]

        return PredictionResponse(
            risk_probability=float(prob),
            is_high_risk=int(prob >= 0.5)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
