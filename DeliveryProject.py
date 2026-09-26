
# =========================================
# LIBRARIES
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import GridSearchCV


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("Zomato_Dataset.csv")

print(df.head())
print("Shape:", df.shape)

print("\nMissing Values (%):")
print(df.isnull().sum() / len(df) * 100)

print("\nDuplicated:", df.duplicated().sum())

print("\nDataset Information:")
print(df.info())

print("\nTarget Description:")
print(df["Time_taken (min)"].describe())


# =========================================
# EDA
# =========================================

# Boxplot for Target Outlier Detection

plt.figure(figsize=(8, 4))

sns.boxplot(x=df["Time_taken (min)"])

plt.title("Boxplot of Delivery Time")
plt.show()


# Delivery Rating vs Delivery Time

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=df["Delivery_person_Ratings"],
    y=df["Time_taken (min)"]
)

plt.title("Delivery Rating vs Delivery Time")
plt.xlabel("Delivery Rating")
plt.ylabel("Time Taken in Minutes")

plt.show()


# Correlation Heatmap

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap of Numeric Variables")

plt.show()


# =========================================
# DATA CLEANING
# =========================================

# Numerical columns that need checking

numerical_cols = [
    "Delivery_person_Age",
    "Delivery_person_Ratings"
]

# Categorical / discrete columns

categorical_cols = [
    "Weather_conditions",
    "Road_traffic_density",
    "multiple_deliveries",
    "Festival",
    "City"
]


# -----------------------------------------
# Fill Numerical Missing Values
# -----------------------------------------

for col in numerical_cols:

    df[col] = df[col].fillna(df[col].median())

    print(
        col,
        "remaining missing:",
        df[col].isnull().sum()
    )


# -----------------------------------------
# Fill Categorical Missing Values
# -----------------------------------------

for col in categorical_cols:

    df[col] = df[col].fillna(df[col].mode()[0])

    print(
        col,
        "remaining missing:",
        df[col].isnull().sum()
    )


# =========================================
# CHECK INCONSISTENT VALUES
# =========================================

print("\nAge:")
print(df["Delivery_person_Age"].describe())

print("\nRating:")
print(df["Delivery_person_Ratings"].describe())


# Rating should be between 1 and 5

print(
    "\nNumber of Rating = 6:",
    (df["Delivery_person_Ratings"] == 6).sum()
)


# Convert invalid rating 6 into NaN

df.loc[
    df["Delivery_person_Ratings"] == 6,
    "Delivery_person_Ratings"
] = np.nan


# Fill invalid ratings using median

df["Delivery_person_Ratings"] = df[
    "Delivery_person_Ratings"
].fillna(
    df["Delivery_person_Ratings"].median()
)


print("\nFinal Delivery Rating:")
print(df["Delivery_person_Ratings"].describe())


# Check categorical values

for col in [
    "Weather_conditions",
    "Road_traffic_density",
    "multiple_deliveries",
    "Festival",
    "City"
]:

    print("\n", col)
    print(df[col].unique())


# =========================================
# TARGET OUTLIER DETECTION
# =========================================

Q1 = df["Time_taken (min)"].quantile(0.25)

Q3 = df["Time_taken (min)"].quantile(0.75)

IQR = Q3 - Q1

lower_boundary = Q1 - 1.5 * IQR

upper_boundary = Q3 + 1.5 * IQR


print("\nTarget Q1:", Q1)
print("Target Q3:", Q3)
print("Target IQR:", IQR)
print("Lower Boundary:", lower_boundary)
print("Upper Boundary:", upper_boundary)


outliers = df[
    (df["Time_taken (min)"] < lower_boundary) |
    (df["Time_taken (min)"] > upper_boundary)
]

print("Target Outliers:", len(outliers))

# These target outliers are plausible delivery times,
# so we KEEP them.


# =========================================
# FEATURE ENGINEERING
# =========================================

# -----------------------------------------
# Distance_km using Haversine Formula
# -----------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


df["Distance_km"] = haversine_distance(
    df["Restaurant_latitude"],
    df["Restaurant_longitude"],
    df["Delivery_location_latitude"],
    df["Delivery_location_longitude"]
)


print("\nDistance Description:")
print(df["Distance_km"].describe())


# -----------------------------------------
# Distance Outlier Detection
# -----------------------------------------

plt.figure(figsize=(8, 4))

sns.boxplot(x=df["Distance_km"])

plt.title("Boxplot of Distance_km")
plt.xlabel("Distance (km)")

plt.show()


Q1 = df["Distance_km"].quantile(0.25)

Q3 = df["Distance_km"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


print("\nDistance Q1:", Q1)
print("Distance Q3:", Q3)
print("Distance IQR:", IQR)
print("Distance Lower Bound:", lower_bound)
print("Distance Upper Bound:", upper_bound)


distance_outliers = df[
    (df["Distance_km"] < lower_bound) |
    (df["Distance_km"] > upper_bound)
]

print(
    "Number of Distance Outliers:",
    len(distance_outliers)
)


# Keep only practical delivery distances

df = df[
    df["Distance_km"] <= upper_bound
].copy()


print("\nDistance After Cleaning:")
print(df["Distance_km"].describe())


# =========================================
# ORDER HOUR
# =========================================

df["Order_Hour"] = pd.to_datetime(
    df["Time_Orderd"],
    format="%H:%M",
    errors="coerce"
).dt.hour


print("\nOrder Hour:")
print(df["Order_Hour"].head())


# =========================================
# ORDER DAY OF WEEK
# =========================================

df["Order_DayofWeek"] = pd.to_datetime(
    df["Order_Date"],
    dayfirst=True,
    errors="coerce"
).dt.day_name()


print("\nOrder Day of Week:")
print(df["Order_DayofWeek"].head())


# =========================================
# PICKUP DELAY
# =========================================

# Convert order times into datetime

order_time = pd.to_datetime(
    df["Time_Orderd"],
    format="%H:%M",
    errors="coerce"
)

pickup_time = pd.to_datetime(
    df["Time_Order_picked"],
    format="%H:%M",
    errors="coerce"
)


# Calculate pickup delay in minutes

df["Pickup_Delay_min"] = (
    pickup_time - order_time
).dt.total_seconds() / 60


# Handle midnight crossing

df.loc[
    df["Pickup_Delay_min"] < 0,
    "Pickup_Delay_min"
] += 24 * 60


print("\nPickup Delay:")
print(df["Pickup_Delay_min"].describe())


# =========================================
# STEP 8 - TARGET AND INPUT
# =========================================

X = df.drop(
    "Time_taken (min)",
    axis=1
)

y = df["Time_taken (min)"]


# =========================================
# REMOVE UNNECESSARY / RAW COLUMNS
# =========================================

drop_cols = [

    # ID columns
    "ID",
    "Delivery_person_ID",

    # Raw time columns
    "Time_Orderd",
    "Time_Order_picked",
    "Order_Date",

    # Raw coordinates
    # Distance_km already represents
    # the useful location information
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude"
]


X = X.drop(
    columns=drop_cols
)


print("\nFinal X Shape:")
print(X.shape)

print("\nFinal X Columns:")
print(X.columns)


# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTrain Test Split:")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# =========================================
# PREPROCESSING
# =========================================

numerical_cols = [

    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Distance_km",
    "Order_Hour",
    "Pickup_Delay_min"
]


categorical_cols = [

    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
    "Order_DayofWeek"
]


# Numerical Pipeline

num_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    )
])


# Categorical Pipeline

cat_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),

    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# Column Transformer

transformer = [

    (
        "num",
        num_pipeline,
        numerical_cols
    ),

    (
        "cat",
        cat_pipeline,
        categorical_cols
    )
]


preprocessor = ColumnTransformer(
    transformers=transformer
)


# =========================================
# STEP 10 - BASELINE MODEL
# =========================================

baseline_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        LinearRegression()
    )
])


baseline_model.fit(
    X_train,
    y_train
)


baseline_pred = baseline_model.predict(
    X_test
)


# =========================================
# STEP 11.1 - DECISION TREE
# =========================================

dt_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "regressor",
        DecisionTreeRegressor(
            random_state=42
        )
    )
])


dt_model.fit(
    X_train,
    y_train
)


dt_pred = dt_model.predict(
    X_test
)


# =========================================
# STEP 11.2 - RANDOM FOREST
# =========================================

rf_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )
])


rf_model.fit(
    X_train,
    y_train
)


rf_pred = rf_model.predict(
    X_test
)


# =========================================
# STEP 11.3 - GRADIENT BOOSTING
# =========================================

gb_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "regressor",
        GradientBoostingRegressor(
            random_state=42
        )
    )
])


gb_model.fit(
    X_train,
    y_train
)


gb_pred = gb_model.predict(
    X_test
)


# =========================================
# STEP 12 & 13 - EVALUATION & COMPARISON
# =========================================

models_predictions = {

    "Linear Regression": baseline_pred,

    "Decision Tree": dt_pred,

    "Random Forest": rf_pred,

    "Gradient Boosting": gb_pred
}


for name, pred in models_predictions.items():

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )


    print("\n", name)

    print("MAE :", mae)

    print("RMSE:", rmse)

    print("R²  :", r2)


# =========================================
# STEP 14 - HYPERPARAMETER TUNING
# =========================================

# Best parameters obtained from tuning

best_params = {

    "n_estimators": 300,

    "max_depth": 20,

    "min_samples_split": 10,

    "min_samples_leaf": 4
}


print("\nBest Parameters:")
print(best_params)


# =========================================
# STEP 15 - FINAL MODEL
# =========================================

final_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "regressor",

        RandomForestRegressor(

            n_estimators=300,

            max_depth=20,

            min_samples_split=10,

            min_samples_leaf=4,

            random_state=42,

            n_jobs=-1
        )
    )
])


# -----------------------------------------
# FIT FINAL MODEL
# -----------------------------------------

final_model.fit(
    X_train,
    y_train
)


# -----------------------------------------
# FINAL PREDICTION
# -----------------------------------------

final_pred = final_model.predict(
    X_test
)


# -----------------------------------------
# FINAL EVALUATION
# -----------------------------------------

mae = mean_absolute_error(
    y_test,
    final_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_pred
    )
)

r2 = r2_score(
    y_test,
    final_pred
)


print("\n=================================")
print("FINAL MODEL RESULTS")
print("=================================")

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


# =========================================
# STEP 16 - SAVE MODEL
# =========================================

joblib.dump(
    final_model,
    "food_delivery_time_model.pkl"
)


print("\nModel Saved Successfully!")
print("File: food_delivery_time_model.pkl")

