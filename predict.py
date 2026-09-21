import joblib
import pandas as pd

# Load saved model

model_data = joblib.load("models/mobile_price_model.pkl")

model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]


# Take user input

print("\nMobile Price Prediction")
print("-----------------------")
print("Enter the following mobile specifications:\n")


user_data = {}

for feature in feature_names:
    value = float(input(f"Enter {feature}: "))
    user_data[feature] = value


# Create DataFrame

input_data = pd.DataFrame([user_data])

# Make absolutely sure the order is the same
input_data = input_data[feature_names]


# Scale input

input_scaled = scaler.transform(input_data)


# Make prediction

prediction = model.predict(input_scaled)[0]


# Display result

price_labels = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}

print("Prediction:", prediction)
print("Price Range:", price_labels[prediction])
