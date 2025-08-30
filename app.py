import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your master dataset
master_df = pd.read_csv("master_df.csv")

# -------------------------------
# Train models inside app (or you can load from pickle if you saved them)
# -------------------------------
# Features
features = ["appointment_month", "age", "is_weekend", "cost"]

# Regression target
reg_df = master_df.groupby(features).size().reset_index(name="appointment_count")

X = reg_df[features]
y_reg = reg_df["appointment_count"]

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X_train, y_train)

# Classification target
reg_df["high_demand"] = (reg_df["appointment_count"] > reg_df["appointment_count"].median()).astype(int)
y_class = reg_df["high_demand"]

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

class_model = RandomForestClassifier(random_state=42)
class_model.fit(X_train, y_train)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🩺 NHI Healthcare Demand Prediction Dashboard")

st.header("Manual Input Prediction")

# Create input fields
month = st.slider("Appointment Month", 1, 12, 6)
age = st.slider("Patient Age", 0, 100, 30)
is_weekend = st.selectbox("Is Weekend?", [0, 1])  # 0 = No, 1 = Yes
cost = st.number_input("Treatment Cost", min_value=0, max_value=5000, value=500)

# Predict button
if st.button("Predict Demand"):
    input_data = pd.DataFrame([[month, age, is_weekend, cost]], columns=features)

    # Regression prediction
    reg_pred = reg_model.predict(input_data)[0]

    # Classification prediction
    class_pred = class_model.predict(input_data)[0]
    demand_label = "High Demand" if class_pred == 1 else "Low Demand"

    # Show results
    st.success(f"📊 Predicted Appointment Count: {int(reg_pred)}")
    st.info(f"🔍 Demand Classification: {demand_label}")

