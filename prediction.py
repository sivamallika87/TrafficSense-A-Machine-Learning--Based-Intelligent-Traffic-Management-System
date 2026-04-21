import joblib

# Load trained model
model = joblib.load("models/traffic_model.pkl")

# Example input (vehicle count)
vehicle_count = [[75]]

# Predict congestion level
prediction = model.predict(vehicle_count)

print(f"🚗 Vehicle Count: {vehicle_count[0][0]}")
print(f"📊 Predicted Congestion Level: {prediction[0]:.2f}")