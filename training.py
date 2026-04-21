import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Step 1: Load dataset
data = pd.read_csv("data/traffic_data.csv")

# Step 2: Features & target
X = data[['vehicle_count']]
y = data['congestion_level']

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)

# Step 4: Save model
joblib.dump(model, "models/traffic_model.pkl")

print("✅ Model trained successfully")
print("💾 Model saved at models/traffic_model.pkl")