# 🍔 Food Delivery Time Prediction

A Machine Learning regression project that predicts the estimated food delivery time based on order, delivery, traffic, weather, vehicle, distance, and pickup-delay related information.

🔗 **Live Demo:** https://food-delivery-time-prediction-sfqpht8rnetp33fjsqju35.streamlit.app/

---

## 📌 Project Overview

Food delivery time depends on several factors such as delivery distance, traffic conditions, weather, vehicle type, delivery partner information, order timing, and pickup delay.

The objective of this project is to build a regression model that predicts the expected delivery time in minutes.

The final model is deployed as an interactive **Streamlit web application**.

---

## 🎯 Problem Statement

Given information about a food delivery order and its delivery conditions, predict:

**`Time_taken (min)`**

This is a **supervised machine learning regression problem** because the target variable is continuous.

---

## 📊 Dataset

The project uses a food delivery operations dataset containing information about:

* Delivery partner details
* Ratings
* Vehicle condition
* Weather conditions
* Road traffic density
* Type of order
* Type of vehicle
* Festival information
* City
* Order date and time
* Restaurant and delivery locations
* Delivery time

The original dataset is not included in this repository.

---

## 🔄 Machine Learning Workflow

```text
Problem Understanding
        ↓
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Preprocessing
        ↓
Baseline Model
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Final Random Forest Model
        ↓
Model Serialization
        ↓
Streamlit Deployment
```

---

## 🔎 Exploratory Data Analysis

The target variable `Time_taken (min)` was analyzed using descriptive statistics and visualizations.

Some important observations included:

* Multiple deliveries showed a positive relationship with delivery time.
* Delivery partner ratings showed a negative relationship with delivery time.
* Delivery partner age and vehicle condition also showed relationships with delivery time.
* Raw latitude and longitude values individually had limited direct usefulness for prediction.

Outliers were investigated using the IQR method and domain plausibility was considered before deciding whether to retain or remove them.

---

## 🛠️ Feature Engineering

Several useful features were created from the original dataset.

### 📍 Delivery Distance

Restaurant and delivery coordinates were used to calculate geographical distance using the **Haversine formula**.

The resulting feature:

```text
Distance_km
```

Extreme distance values were filtered using an IQR-based threshold.

### ⏰ Order Hour

The order time was converted into:

```text
Order_Hour
```

This captures the effect of different times of the day on delivery time.

### 📅 Order Day of Week

The order date was converted into:

```text
Order_DayofWeek
```

### 🛵 Pickup Delay

The difference between order time and pickup time was calculated as:

```text
Pickup_Delay_min
```

This captures the time spent waiting between order placement and pickup.

---

## ⚙️ Data Preprocessing

### Numerical Features

The numerical pipeline uses:

* Median imputation
* StandardScaler

Numerical features include:

```text
Delivery_person_Age
Delivery_person_Ratings
Vehicle_condition
multiple_deliveries
Distance_km
Order_Hour
Pickup_Delay_min
```

### Categorical Features

The categorical pipeline uses:

* Most-frequent imputation
* One-Hot Encoding
* `handle_unknown="ignore"`

Categorical features include:

```text
Weather_conditions
Road_traffic_density
Type_of_order
Type_of_vehicle
Festival
City
Order_DayofWeek
```

A `ColumnTransformer` combines both preprocessing pipelines.

---

## 🤖 Models Used

The following regression models were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Random Forest was selected as the final model after model comparison and hyperparameter tuning.

### Final Random Forest Parameters

```text
n_estimators = 300
max_depth = 20
min_samples_split = 10
min_samples_leaf = 4
random_state = 42
```

---

## 📈 Model Evaluation

The models were evaluated using:

* **MAE — Mean Absolute Error**
* **RMSE — Root Mean Squared Error**
* **R² Score**

The final model was trained using a complete preprocessing + Random Forest pipeline and saved using Joblib.

---

## 🚀 Deployment

The trained model was serialized as:

```text
food_delivery_time_model.pkl
```

The application was built using **Streamlit** and deployed using **Streamlit Community Cloud**.

### Live Application

🔗 https://food-delivery-time-prediction-sfqpht8rnetp33fjsqju35.streamlit.app/

The application allows users to enter delivery-related information and receive an estimated delivery time.

---

## 🖥️ Project Structure

```text
food-delivery-time-prediction/
│
├── app.py
├── DeliveryProject.py
├── food_delivery_time_model.pkl
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

The dataset is excluded from the repository using `.gitignore`.

The model file is stored using **Git LFS** because of its size.

---

## 🧰 Tech Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest
* Joblib

### Deployment

* Streamlit
* Streamlit Community Cloud
* GitHub
* Git LFS

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/HarshKishan05/food-delivery-time-prediction.git
```

Move into the project directory:

```bash
cd food-delivery-time-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

Possible improvements include:

* Additional feature engineering
* Model comparison with advanced boosting algorithms
* Better hyperparameter optimization
* Prediction confidence or uncertainty estimation
* Improved UI and visualization
* Model monitoring after deployment
* Automated retraining pipeline

---

## 👨‍💻 Student Details

**Harsh Kishan**

B.Tech — Artificial Intelligence & Machine Learning
Roll No.BTECH/15145/24
AIML Branch,3rd Year(5th Semester)
Birla Institute of Technology, Patna
