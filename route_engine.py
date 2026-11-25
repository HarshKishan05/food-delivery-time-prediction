
```python
import requests
import joblib
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path


# =========================================
# STEP 1: LOAD EXISTING TRAINED MODEL
# =========================================

MODEL_PATH = Path(__file__).resolve().parent / "food_delivery_time_model.pkl"

MODEL = joblib.load(MODEL_PATH)


# =========================================
# STEP 2: MODEL FEATURES
# =========================================

FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Distance_km",
    "Order_Hour",
    "Pickup_Delay_min",
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
    "Order_DayofWeek",
]


# =========================================
# STEP 3: STRAIGHT-LINE DISTANCE
# =========================================

def haversine_km(src, dst):
    """
    Calculate straight-line distance between
    source and destination coordinates.

    Coordinates format: (latitude, longitude)
    """

    lat1, lon1 = src
    lat2, lon2 = dst

    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return 6371 * c


# =========================================
# STEP 4: FETCH ALTERNATIVE ROUTES
# =========================================

def get_routes(src, dst):

    # OSRM expects longitude,latitude
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{src[1]},{src[0]};{dst[1]},{dst[0]}"
    )

    params = {
        "alternatives": "true",
        "overview": "full",
        "geometries": "geojson"
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError("No driving routes found.")

    return data["routes"]


# =========================================
# STEP 5: PREDICT ETA FOR ONE DISTANCE
# =========================================

def predict_eta(order, distance_km):

    # Add route-specific distance to order features
    row = {
        **order,
        "Distance_km": distance_km
    }

    # Maintain the required feature order
    X = pd.DataFrame([row])[FEATURES]

    # Predict delivery time using trained model
    prediction = float(MODEL.predict(X)[0])

    return max(0.0, prediction)


# =========================================
# STEP 6: NORMAL ETA WITHOUT ROUTING
# =========================================

def predict_distance_only(order, distance_km):

    eta = predict_eta(order, distance_km)

    return {
        "eta_min": round(eta, 1),
        "best": None,
        "routes": [],
        "minutes_saved": 0.0,
        "fallback": True,
        "fallback_reason": (
            "Routing API unavailable. ETA is based "
            "on the supplied distance."
        )
    }


# =========================================
# STEP 7: ETA + BEST ROUTE
# =========================================

def predict_with_path(order, src, dst):

    straight_distance = haversine_km(src, dst)

    # Fetch routes from OSRM
    try:
        routes = get_routes(src, dst)

    except Exception as exc:

        result = predict_distance_only(
            order,
            straight_distance
        )

        result["fallback_reason"] = (
            f"Routing API failed: {exc}. "
            "ETA uses straight-line distance, "
            "so it may be less accurate."
        )

        return result

    # Calculate ETA for each route
    results = []

    for i, route in enumerate(routes):

        distance_km = route["distance"] / 1000

        eta = predict_eta(
            order,
            distance_km
        )

        results.append({
            "route_id": i,
            "distance_km": round(distance_km, 2),
            "eta_min": round(eta, 1),
            "road_duration_min": round(
                route["duration"] / 60, 1
            ),
            "geometry": route["geometry"]
        })

    # Select the route with minimum model-predicted ETA
    best = min(
        results,
        key=lambda route: route["eta_min"]
    )

    # Compare with the first route returned by OSRM
    default = results[0]

    return {
        "eta_min": best["eta_min"],
        "best": best,
        "routes": results,
        "minutes_saved": round(
            default["eta_min"] - best["eta_min"], 1
        ),
        "fallback": False,
        "fallback_reason": None
    }
```
