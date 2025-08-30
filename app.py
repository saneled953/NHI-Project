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


