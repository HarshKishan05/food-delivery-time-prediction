
import streamlit as st
import pandas as pd
import joblib


# =========================================
# LOAD TRAINED MODEL
# =========================================

model = joblib.load("food_delivery_time_model.pkl")


# =========================================
# APP TITLE
# =========================================

st.title("🍔 Food Delivery Time Prediction")

st.write(
    "Enter the delivery details to predict the estimated delivery time."
)


# =========================================
# INPUT FIELDS
# =========================================

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
            "Motorcycle",
            "Scooter",
            "Electric_scooter",
            "Bicycle"
        ]
    )

    festival = st.selectbox(
        "Festival",
        [
            "No",
            "Yes"
        ]
    )

    # Festival name is only for user interface.
    # It is NOT sent to the model because
    # the trained model does not contain this feature.

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
# DELIVERY DISTANCE
# =========================================

st.subheader("Delivery Location Details")

distance = st.number_input(
    "Delivery Distance (km)",
    min_value=0.0,
    max_value=21.0,
    value=9.0,
    step=0.1
)


# =========================================
# PREDICTION
# =========================================

if st.button("Predict Delivery Time"):

    # Create input DataFrame
    # Column names MUST match training data

    input_data = pd.DataFrame({
        "Delivery_person_Age": [age],
        "Delivery_person_Ratings": [rating],
        "Vehicle_condition": [vehicle_condition],
        "multiple_deliveries": [multiple_deliveries],
        "Weather_conditions": [weather],
        "Road_traffic_density": [traffic],
        "Type_of_order": [type_of_order],
        "Type_of_vehicle": [type_of_vehicle],
        "Festival": [festival],
        "City": [city],
        "Order_DayofWeek": [order_day],
        "Order_Hour": [order_hour],
        "Distance_km": [distance],
        "Pickup_Delay_min": [pickup_delay]
    })


    # Make prediction

    prediction = model.predict(input_data)[0]


    # Display result

    st.success(
        f"Estimated Delivery Time: {prediction:.1f} minutes"
    )

