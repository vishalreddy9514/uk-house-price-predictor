"""
Tests for the FastAPI backend.

Run with: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_input_returns_price():
    payload = {
        "region": "London",
        "property_type": "Flat",
        "bedrooms": 2,
        "bathrooms": 1,
        "sqft": 650,
        "year_built": 1990,
        "dist_to_station_km": 0.5,
        "has_garden": False,
        "crime_score": 4.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price" in body
    assert body["predicted_price"] > 0
    assert body["model_used"] in ("XGBoost", "LinearRegression")


def test_predict_larger_property_costs_more():
    small = {
        "region": "London", "property_type": "Flat", "bedrooms": 1,
        "bathrooms": 1, "sqft": 400, "year_built": 1990,
        "dist_to_station_km": 1.0, "has_garden": False, "crime_score": 5.0,
    }
    large = {**small, "bedrooms": 4, "sqft": 1800, "property_type": "Detached"}

    small_price = client.post("/predict", json=small).json()["predicted_price"]
    large_price = client.post("/predict", json=large).json()["predicted_price"]

    assert large_price > small_price


def test_predict_rejects_invalid_region():
    payload = {
        "region": "Atlantis",
        "property_type": "Flat",
        "bedrooms": 2,
        "bathrooms": 1,
        "sqft": 650,
        "year_built": 1990,
        "dist_to_station_km": 0.5,
        "has_garden": False,
        "crime_score": 4.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    payload = {
        "region": "London",
        "property_type": "Flat",
        "bathrooms": 1,
        "sqft": 650,
        "year_built": 1990,
        "dist_to_station_km": 0.5,
        "has_garden": False,
        "crime_score": 4.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_crime_score():
    payload = {
        "region": "London",
        "property_type": "Flat",
        "bedrooms": 2,
        "bathrooms": 1,
        "sqft": 650,
        "year_built": 1990,
        "dist_to_station_km": 0.5,
        "has_garden": False,
        "crime_score": 99,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422