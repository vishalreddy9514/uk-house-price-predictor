"""
FastAPI backend for the UK house price model.

Run locally:
    uvicorn api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI.
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

app = FastAPI(title="UK House Price Predictor", version="1.0.0")

# allow the Streamlit / React frontend (running on a different port/domain)
# to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = joblib.load(MODEL_PATH)
pipeline = _bundle["pipeline"]
MODEL_NAME = _bundle["model_name"]
TEST_MAE = _bundle["test_mae"]
TEST_R2 = _bundle["test_r2"]


class HouseFeatures(BaseModel):
    region: Literal[
        "London", "South East", "South West", "North West",
        "Yorkshire", "West Midlands", "East of England", "Scotland",
    ]
    property_type: Literal["Flat", "Terraced", "Semi-Detached", "Detached"]
    bedrooms: int = Field(ge=1, le=10)
    bathrooms: int = Field(ge=1, le=6)
    sqft: int = Field(ge=150, le=10000)
    year_built: int = Field(ge=1800, le=2026)
    dist_to_station_km: float = Field(ge=0, le=50)
    has_garden: bool
    crime_score: float = Field(ge=1, le=10, description="1 = low crime, 10 = high crime")


class PredictionResponse(BaseModel):
    predicted_price: float
    model_used: str
    model_test_mae: float
    model_test_r2: float


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME, "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    row = features.model_dump()
    row["property_age"] = 2026 - row.pop("year_built")
    row["sqft_per_bedroom"] = row["sqft"] / row["bedrooms"]
    row["has_garden"] = int(row["has_garden"])

    X = pd.DataFrame([row])
    pred = float(pipeline.predict(X)[0])

    return PredictionResponse(
        predicted_price=round(pred, 2),
        model_used=MODEL_NAME,
        model_test_mae=round(TEST_MAE, 2),
        model_test_r2=round(TEST_R2, 4),
    )
