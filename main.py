import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# Create results folder if not exists
os.makedirs("results", exist_ok=True)

# Step 1: Load dataset
data = pd.read_csv("data/traffic_data.csv")

# Step 2: Select features and target
# Make sure your dataset has these column names
X = data[['vehicle_count']]
y = data['congestion_level']

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict
predictions = model.predict(X)

# Step 5: Plot results
plt.scatter(X, y, label="Actual Data")
plt.plot(X, predictions, linestyle='dashed', label="Predicted")
plt.xlabel("Vehicle Count")
plt.ylabel("Congestion Level")
plt.title("Traffic Prediction")
plt.legend()

# Step 6: Save output
plt.savefig("results/output.png")
plt.show()

print("✅ Model trained successfully")
print("📊 Output saved in results/output.png")