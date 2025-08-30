import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# -------------------------
# Load data
# -------------------------
demand_df = pd.read_csv("demand_df.csv")
master_df = pd.read_csv("master_df.csv")
reg_results = pd.read_csv("reg_results.csv")
class_results = pd.read_csv("class_results.csv")

# -------------------------
# Train (or load) model for manual prediction
# -------------------------
# Use RandomForestClassifier for high vs low demand
X = master_df[["appointment_month", "age", "is_weekend", "cost"]]
y = (master_df["appointment_id"].notnull()).astype(int)  # Example target, replace with actual demand flag

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="NHI Healthcare Demand Prediction Dashboard", layout="wide")

st.title("NHI Healthcare Demand Prediction Dashboard")

tab1, tab2 = st.tabs(["📊 Live Dashboard", "🤖 Manual Prediction"])

# -------------------------
# TAB 1: Live Dashboard
# -------------------------
with tab1:
    st.subheader("Healthcare Demand Trends")

    st.write("### Monthly Demand")
    st.bar_chart(demand_df.groupby("appointment_month")["appointment_count"].sum())

    st.write("### Model Results (Regression)")
    st.dataframe(reg_results)

    st.write("### Model Results (Classification)")
    st.dataframe(class_results)

# -------------------------
# TAB 2: Manual Prediction
# -------------------------
with tab2:
    st.subheader("Manual Demand Prediction")

    month = st.selectbox("Appointment Month", sorted(master_df["appointment_month"].unique()))
    age = st.slider("Age", int(master_df["age"].min()), int(master_df["age"].max()))
    weekend = st.radio("Is Weekend?", [0, 1])
    cost = st.number_input("Treatment Cost", min_value=0, max_value=int(master_df["cost"].max()), step=50)

    if st.button("Predict Demand"):
        # Prepare input for model
        user_input = np.array([[month, age, weekend, cost]])
        prediction = clf.predict(user_input)[0]
        proba = clf.predict_proba(user_input)[0]

        result = "High Demand" if prediction == 1 else "Low Demand"
        confidence = round(max(proba) * 100, 2)

        st.success(f"✅ Predicted: {result} (Confidence: {confidence}%)")

        # Show probability breakdown
        st.write("Prediction probabilities:")
        st.write({
            "Low Demand": f"{proba[0]*100:.2f}%",
            "High Demand": f"{proba[1]*100:.2f}%"
        })




