import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("models/traffic_model.pkl")

st.title("🚦 TrafficSense")
st.subheader("Traffic Congestion Prediction System")

vehicle_count = st.number_input("Enter Vehicle Count", min_value=0, step=1)

if st.button("Predict"):
    prediction = model.predict([[vehicle_count]])
    st.success(f"Predicted Congestion Level: {prediction[0]:.2f}")