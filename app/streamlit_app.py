"""
Simple Streamlit UI for the UK house price predictor.

Run locally (with the API already running on port 8000):
    streamlit run app/streamlit_app.py
"""
import os

import requests
import streamlit as st

# When deployed, set API_URL as an environment variable to your live API's URL.
# Locally it defaults to the FastAPI server running on port 8000.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="UK House Price Predictor", page_icon="🏠")
st.title("🏠 UK House Price Predictor")
st.caption("Trained on housing data with region, property type, size and local features.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Region", [
            "London", "South East", "South West", "North West",
            "Yorkshire", "West Midlands", "East of England", "Scotland",
        ])
        property_type = st.selectbox("Property type", ["Flat", "Terraced", "Semi-Detached", "Detached"])
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=1)
        sqft = st.number_input("Size (sq ft)", min_value=150, max_value=10000, value=650)

    with col2:
        year_built = st.number_input("Year built", min_value=1800, max_value=2026, value=1990)
        dist_to_station = st.slider("Distance to nearest station (km)", 0.0, 20.0, 0.5)
        has_garden = st.checkbox("Has garden", value=False)
        crime_score = st.slider("Local crime score (1=low, 10=high)", 1.0, 10.0, 4.5)

    submitted = st.form_submit_button("Predict price")

if submitted:
    payload = {
        "region": region,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "year_built": year_built,
        "dist_to_station_km": dist_to_station,
        "has_garden": has_garden,
        "crime_score": crime_score,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        st.success(f"### Estimated price: £{result['predicted_price']:,.0f}")
        st.caption(
            f"Model: {result['model_used']}  |  "
            f"Test MAE: £{result['model_test_mae']:,.0f}  |  "
            f"Test R²: {result['model_test_r2']}"
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach the prediction API at {API_URL}. Is it running? ({e})")
