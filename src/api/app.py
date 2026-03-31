#!/usr/bin/env python3
"""
FastAPI application for serving the flight delay prediction model.

This service provides:
- /predict: Accepts feature data and returns delay probability
- /health: Simple health check endpoint
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flight-prediction-api")

# Load the trained model (adjust path if needed)
MODEL_PATH = Path("../models/lgbm_flight_delay.pkl")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

logger.info(f"Loading model from {MODEL_PATH}")
MODEL = joblib.load(MODEL_PATH)

# FastAPI app
app = FastAPI(title="Flight Delay Prediction API")

# Pydantic model for request validation
class FlightData(BaseModel):
    """Feature schema for flight prediction requests."""
    # These are example features; adjust to match your actual feature set
    DEPARTURE_DELAY: float
    DISTANCE: float
    DURATION: float
    ORIGIN_AIRPORT: str
    DESTINATION_AIRPORT: str
    MONTH: int
    DAY_OF_WEEK: int

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict")
async def predict(data: FlightData):
    """
    Predict the probability of a flight delay (>15 minutes).

    The input should match the feature schema defined in FlightData.
    """
    try:
        # Convert Pydantic model to DataFrame
        df = pd.DataFrame([data.dict()])

        # Ensure categorical columns are properly typed
        categorical_cols = ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Generate prediction
        delay_probabilities = MODEL.predict_proba(df)[:, 1]  # Probability of class 1 (delay)
        probability = float(delay_probabilities[0])

        logger.info(f"Prediction request: probability={probability:.3f}")
        return {"delay_probability": probability}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)