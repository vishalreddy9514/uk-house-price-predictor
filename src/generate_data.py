"""
Generates a synthetic but realistically-structured UK housing dataset.

Why synthetic: this lets the whole pipeline run anywhere with zero downloads.
To use REAL data instead, download the HM Land Registry Price Paid Data
(https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
and reshape it to match the columns below -- the rest of the pipeline
(train.py, api, app) doesn't need to change.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000

regions = ["London", "South East", "South West", "North West",
           "Yorkshire", "West Midlands", "East of England", "Scotland"]
# rough relative price multiplier per region (London most expensive)
region_multiplier = {
    "London": 2.6, "South East": 1.5, "South West": 1.2, "North West": 0.85,
    "Yorkshire": 0.75, "West Midlands": 0.8, "East of England": 1.3, "Scotland": 0.9,
}

property_types = ["Flat", "Terraced", "Semi-Detached", "Detached"]
type_base_price = {"Flat": 180000, "Terraced": 210000, "Semi-Detached": 260000, "Detached": 380000}

rows = []
for _ in range(N):
    region = np.random.choice(regions)
    ptype = np.random.choice(property_types, p=[0.3, 0.3, 0.25, 0.15])
    bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.12, 0.28, 0.32, 0.2, 0.08])
    bathrooms = min(bedrooms, np.random.choice([1, 2, 3], p=[0.55, 0.35, 0.1]))
    sqft = int(np.random.normal(400 + bedrooms * 220, 120))
    sqft = max(sqft, 250)
    year_built = int(np.random.choice(range(1880, 2024)))
    dist_to_station_km = round(np.random.exponential(1.8), 2)
    has_garden = np.random.choice([0, 1], p=[0.35, 0.65])
    crime_score = round(np.random.uniform(1, 10), 1)  # 1 = low crime, 10 = high

    base = type_base_price[ptype] * region_multiplier[region]
    price = (
        base
        + bedrooms * 18000
        + bathrooms * 9000
        + sqft * 140
        + has_garden * 12000
        - dist_to_station_km * 4000
        - crime_score * 3500
        + (2024 - year_built) * -80  # older houses slightly cheaper, all else equal
        + np.random.normal(0, 25000)  # noise
    )
    price = max(int(price), 60000)

    rows.append({
        "region": region,
        "property_type": ptype,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "year_built": year_built,
        "dist_to_station_km": dist_to_station_km,
        "has_garden": has_garden,
        "crime_score": crime_score,
        "price": price,
    })

df = pd.DataFrame(rows)
df.to_csv("data/uk_housing.csv", index=False)
print(f"Wrote {len(df)} rows to data/uk_housing.csv")
print(df.head())
