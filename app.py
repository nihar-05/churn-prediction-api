"""
Telco Customer Churn Prediction — FastAPI Backend
Model: sklearn Pipeline with SMOTE (imbalanced-learn) + classifier
Dataset: IBM Telco Customer Churn (7043 rows, 20 features)
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal
import uvicorn
import os

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts whether a telecom customer will churn based on account and service features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ───────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "churn_prediction_model.pkl")

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


# ── Request / Response schemas ─────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., example="Male")
    SeniorCitizen: Literal[0, 1] = Field(..., example=0)
    Partner: Literal["Yes", "No"] = Field(..., example="Yes")
    Dependents: Literal["Yes", "No"] = Field(..., example="No")
    tenure: int = Field(..., ge=0, le=72, example=12)
    PhoneService: Literal["Yes", "No"] = Field(..., example="Yes")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., example="No")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., example="DSL")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., example="Month-to-month")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., example="Yes")
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=29.85)
    TotalCharges: float = Field(..., ge=0, example=358.20)


class PredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float = Field(..., description="Probability of churn (0–1)")
    risk_level: Literal["Low", "Medium", "High"]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Helper ─────────────────────────────────────────────────────────────────────
def risk_label(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {"message": "Telco Churn Prediction API is running. Visit /docs for the Swagger UI."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict customer churn.

    - **churn_prediction**: `"Yes"` or `"No"`
    - **churn_probability**: Float between 0 and 1
    - **risk_level**: `"Low"` / `"Medium"` / `"High"`
    """
    try:
        input_df = pd.DataFrame([customer.dict()])

        # Some pipelines expect TotalCharges as string (raw dataset quirk)
        input_df["TotalCharges"] = input_df["TotalCharges"].astype(str)

        proba = model.predict_proba(input_df)[0]
        # classes_ order: [No, Yes]  →  index 1 = churn probability
        churn_prob = float(proba[1])
        prediction = "Yes" if churn_prob >= 0.5 else "No"

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(churn_prob, 4),
            risk_level=risk_label(churn_prob),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerFeatures]):
    """Predict churn for a list of customers (max 100)."""
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100.")
    results = []
    for c in customers:
        results.append(predict(c))
    return results


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
