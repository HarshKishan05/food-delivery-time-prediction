```python
import streamlit as st
import pandas as pd
import joblib
import folium

from pathlib import Path
from streamlit_folium import st_folium

from route_engine import predict_with_path


# =========================================
# STEP 1: PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Food Delivery ETA",
    page_icon="🍔",
    layout="wide"
)


# =========================================
# STEP 2: LOAD EXISTING MODEL
# =========================================

@st.cache_resource
def load_model():

    model_path = (
        Path(__file__).resolve().parent
        / "food_delivery_time_model.pkl"
    )

    return joblib.load(model_path)


model = load_model()


# =========================================
# STEP 3: APP TITLE
# =========================================

st.title("🍔 Food Delivery Time Prediction")

st.write(
    "Predict delivery time and compare alternative "
    "driving routes."
)


# =========================================
# STEP 4: DELIVERY PERSON AND ORDER INPUTS
# =========================================

st.subheader("Delivery and Order Details")

col1, col2 = st.columns(2)


# -----------------------------------------
# COLUMN 1
# -----------------------------------------

with col1:

    age = st.number_input(
        "Delivery Person Age",
        min_value=15,
        max_value=50,
        value=30
    )

    rating = st.number_input(
        "Delivery Person Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

    vehicle_condition = st.selectbox(
        "Vehicle Condition",
        [0, 1, 2]
    )

    multiple_deliveries = st.selectbox(
        "Multiple Deliveries",
        [0, 1, 2, 3]
    )

    weather = st.selectbox(
        "Weather Conditions",
        [
            "Fog",
            "Stormy",
            "Sandstorms",
            "Windy",
            "Cloudy",
            "Sunny"
        ]
    )

    traffic = st.selectbox(
        "Road Traffic Density",
        [
            "Jam",
            "High",
            "Medium",
            "Low"
        ]
    )

    type_of_order = st.selectbox(
        "Type of Order",
        [
            "Snack",
            "Drinks",
            "Buffet",
            "Meal"
        ]
    )


# -----------------------------------------
# COLUMN 2
# -----------------------------------------

with col2:

    type_of_vehicle = st.selectbox(
        "Type of Vehicle",
        [
            "motorcycle",
            "scooter",
            "electric_scooter",
            "bicycle"
        ]
    )

    festival = st.selectbox(
        "Festival",
        [
            "No",
            "Yes"
        ]
    )

    if festival == "Yes":

        festival_name = st.selectbox(
            "Festival Name",
            [
                "Diwali",
                "Holi",
                "Eid",
                "Christmas",
                "Other"
            ]
        )

    city = st.selectbox(
        "City",
        [
            "Metropolitian",
            "Urban",
            "Semi-Urban"
        ]
    )

    order_day = st.selectbox(
        "Order Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    order_hour = st.number_input(
        "Order Hour",
        min_value=0,
        max_value=23,
        value=18
    )

    pickup_delay = st.number_input(
        "Pickup Delay (minutes)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )


# =========================================
# STEP 5: LOCATION COORDINATES
# =========================================

st.subheader("📍 Restaurant and Delivery Locations")

st.info(
    "Enter the latitude and longitude of both locations. "
    "The default coordinates are example points near Patna."
)

col3, col4 = st.columns(2)


with col3:

    st.markdown("### Restaurant")

    restaurant_lat = st.number_input(
        "Restaurant Latitude",
        value=25.5941,
        format="%.6f"
    )

    restaurant_lon = st.number_input(
        "Restaurant Longitude",
        value=85.1376,
        format="%.6f"
    )


with col4:

    st.markdown("### Delivery Location")

    delivery_lat = st.number_input(
        "Delivery Latitude",
        value=25.6120,
        format="%.6f"
    )

    delivery_lon = st.number_input(
        "Delivery Longitude",
        value=85.1580,
        format="%.6f"
    )


# =========================================
# STEP 6: PREDICT ETA + BEST ROUTE
# =========================================

if st.button(
    "🚀 Predict ETA + Best Path",
    type="primary",
    use_container_width=True
):

    # Validate coordinates
    coordinates_valid = (
        -90 <= restaurant_lat <= 90
        and -90 <= delivery_lat <= 90
        and -180 <= restaurant_lon <= 180
        and -180 <= delivery_lon <= 180
    )

    if not coordinates_valid:

        st.error("Please enter valid coordinates.")

    elif (
        restaurant_lat == delivery_lat
        and restaurant_lon == delivery_lon
    ):

        st.error(
            "Restaurant and delivery locations "
            "must be different."
        )

    else:

        # ---------------------------------
        # STEP 6A: PREPARE ORDER FEATURES
        # ---------------------------------

        order = {
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "Vehicle_condition": vehicle_condition,
            "multiple_deliveries": multiple_deliveries,
            "Weather_conditions": weather,
            "Road_traffic_density": traffic,
            "Type_of_order": type_of_order,
            "Type_of_vehicle": type_of_vehicle,
            "Festival": festival,
            "City": city,
            "Order_DayofWeek": order_day,
            "Order_Hour": order_hour,
            "Pickup_Delay_min": pickup_delay
        }

        src = (
            restaurant_lat,
            restaurant_lon
        )

        dst = (
            delivery_lat,
            delivery_lon
        )

        # ---------------------------------
        # STEP 6B: CALL ROUTE ENGINE
        # ---------------------------------

        try:

            with st.spinner(
                "Fetching routes and predicting ETA..."
            ):

                out = predict_with_path(
                    order,
                    src,
                    dst
                )

            # ---------------------------------
            # STEP 6C: SHOW ETA
            # ---------------------------------

            st.divider()
            st.subheader("⏱️ Delivery ETA")

            st.metric(
                "Estimated Delivery Time",
                f"{out['eta_min']:.1f} minutes"
            )

            if out["fallback"]:

                st.warning(
                    "Routing API was unavailable. "
                    "The displayed ETA uses straight-line "
                    "distance rather than road distance."
                )

                st.caption(
                    out["fallback_reason"]
                )

            # ---------------------------------
            # STEP 6D: DRAW MAP IF ROUTES EXIST
            # ---------------------------------

            if out["best"] is not None:

                best = out["best"]

                st.subheader("🛣️ Best Route")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Best Route",
                    f"Route {best['route_id'] + 1}"
                )

                c2.metric(
                    "Road Distance",
                    f"{best['distance_km']} km"
                )

                c3.metric(
                    "Model-Predicted ETA",
                    f"{best['eta_min']} min"
                )

                st.caption(
                    "OSRM road duration: "
                    f"{best['road_duration_min']} min. "
                    "This is a routing estimate, not necessarily "
                    "live-traffic-aware."
                )

                if out["minutes_saved"] > 0:

                    st.success(
                        "Estimated time saved versus the first "
                        f"route: {out['minutes_saved']} minutes."
                    )

                elif out["minutes_saved"] == 0:

                    st.info(
                        "The selected route has the same "
                        "predicted ETA as the first route."
                    )

                else:

                    st.info(
                        "The selected route is the fastest "
                        "according to the model, but the first "
                        "route has a lower predicted ETA only "
                        "if model outputs are consistent. "
                        "Please check model feature handling."
                    )

                # ---------------------------------
                # STEP 6E: CREATE MAP
                # ---------------------------------

                route_map = folium.Map(
                    location=src,
                    zoom_start=13
                )

                folium.Marker(
                    location=src,
                    tooltip="Restaurant",
                    popup="Restaurant",
                    icon=folium.Icon(color="blue")
                ).add_to(route_map)

                folium.Marker(
                    location=dst,
                    tooltip="Delivery Location",
                    popup="Delivery Location",
                    icon=folium.Icon(color="red")
                ).add_to(route_map)

                # ---------------------------------
                # STEP 6F: DISPLAY ALL ROUTES
                # ---------------------------------

                for route in out["routes"]:

                    is_best = (
                        route["route_id"]
                        == best["route_id"]
                    )

                    coordinates = [
                        (lat, lon)
                        for lon, lat
                        in route["geometry"]["coordinates"]
                    ]

                    folium.PolyLine(
                        locations=coordinates,
                        color=(
                            "green" if is_best else "gray"
                        ),
                        weight=(
                            6 if is_best else 3
                        ),
                        opacity=0.9,
                        tooltip=(
                            f"Route {route['route_id'] + 1} | "
                            f"{route['distance_km']} km | "
                            f"{route['eta_min']} min"
                        )
                    ).add_to(route_map)

                st.subheader("🗺️ Alternative Routes")

                st_folium(
                    route_map,
                    width=None,
                    height=500,
                    use_container_width=True
                )

                # ---------------------------------
                # STEP 6G: ROUTE COMPARISON TABLE
                # ---------------------------------

                st.subheader("📊 Route Comparison")

                route_table = pd.DataFrame([
                    {
                        "Route": (
                            f"Route {r['route_id'] + 1}"
                        ),
                        "Distance (km)": r["distance_km"],
                        "Predicted ETA (min)": r["eta_min"],
                        "OSRM Duration (min)": (
                            r["road_duration_min"]
                        ),
                        "Selected Best": (
                            "Yes"
                            if r["route_id"]
                            == best["route_id"]
                            else "No"
                        )
                    }
                    for r in out["routes"]
                ])

                st.dataframe(
                    route_table,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.error(
                "Prediction failed. Check the model's "
                "feature names, data types, and preprocessing."
            )

            st.exception(e)
```
