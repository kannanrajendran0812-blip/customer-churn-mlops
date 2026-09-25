from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Best model path
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


# Load model once when API starts
model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Prediction service for customer churn",
    version="1.0.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "best_model.joblib",
    }


@app.post("/predict")
def predict(customer: CustomerData):

    # Convert request to DataFrame
    data = pd.DataFrame(
        [customer.model_dump()]
    )

    # Prediction
    prediction = model.predict(data)[0]

    # Churn probability
    probability = model.predict_proba(data)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(
            float(probability),
            4
        ),
    }