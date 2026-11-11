
# ============================================================
# PORTGUARD AI
# PORT CONGESTION PREDICTION MODULE
# ============================================================

import joblib
import pandas as pd

from pathlib import Path


# ------------------------------------------------------------
# 1. LOAD SAVED MACHINE LEARNING MODEL
# ------------------------------------------------------------

project_directory = Path(__file__).resolve().parent

model_path = project_directory / "models" / "congestion_model.pkl"

model = joblib.load(model_path)

print("\n==========================================")
print(" PORTGUARD AI - CONGESTION PREDICTION")
print("==========================================")

print("\nTrained model loaded successfully!")


# ------------------------------------------------------------
# 2. ACCEPT NEW PORT ACTIVITY INFORMATION
# ------------------------------------------------------------

port_name = input("\nEnter port name: ").strip()

vessel_arrivals = int(input("Enter vessel arrivals: "))

port_capacity = int(input("Enter port capacity: "))

queue_length = int(input("Enter queue length: "))

weather_severity = int(input("Enter weather severity (0-10): "))

historical_avg_delay = float(
    input("Enter historical average delay (hours): ")
)

hour_of_day = int(input("Enter hour of day (0-23): "))

month = int(input("Enter month (1-12): "))


# ------------------------------------------------------------
# 3. VALIDATE INPUT INFORMATION
# ------------------------------------------------------------

valid_ports = [
    "Chennai",
    "Mumbai",
    "Visakhapatnam",
    "Kolkata",
    "Cochin"
]

if port_name not in valid_ports:
    raise ValueError("Invalid port name.")

if not 5 <= vessel_arrivals <= 50:
    raise ValueError("Vessel arrivals must be between 5 and 50.")

if not 1 <= port_capacity <= 50:
    raise ValueError("Port capacity must be between 1 and 50.")

if not 0 <= queue_length <= 30:
    raise ValueError("Queue length must be between 0 and 30.")

if not 0 <= weather_severity <= 10:
    raise ValueError("Weather severity must be between 0 and 10.")

if not 0 <= historical_avg_delay <= 8:
    raise ValueError("Historical delay must be between 0 and 8 hours.")

if not 0 <= hour_of_day <= 23:
    raise ValueError("Hour must be between 0 and 23.")

if not 1 <= month <= 12:
    raise ValueError("Month must be between 1 and 12.")


# ------------------------------------------------------------
# 4. CREATE INPUT DATAFRAME
# ------------------------------------------------------------

input_data = pd.DataFrame([{
    "port_name": port_name,
    "vessel_arrivals": vessel_arrivals,
    "port_capacity": port_capacity,
    "queue_length": queue_length,
    "weather_severity": weather_severity,
    "historical_avg_delay": historical_avg_delay,
    "hour_of_day": hour_of_day,
    "month": month
}])


# ------------------------------------------------------------
# 5. MAKE CONGESTION PREDICTION
# ------------------------------------------------------------

prediction = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]

congestion_probability = probabilities[
    list(model.classes_).index(1)
]


# ------------------------------------------------------------
# 6. DISPLAY PREDICTION RESULTS
# ------------------------------------------------------------

print("\n==========================================")
print(" PORTGUARD AI - PREDICTION RESULT")
print("==========================================")

print(f"\nPort Name: {port_name}")

print(f"Vessel Arrivals: {vessel_arrivals}")

print(f"Port Capacity: {port_capacity}")

print(f"Queue Length: {queue_length}")

print(f"Weather Severity: {weather_severity}")

print(f"Historical Average Delay: {historical_avg_delay} hours")


if prediction == 1:

    print("\nCONGESTION STATUS: CONGESTION DETECTED")

else:

    print("\nCONGESTION STATUS: NO CONGESTION DETECTED")


print(
    f"\nModel-estimated congestion probability: "
    f"{congestion_probability * 100:.2f}%"
)

print("\n==========================================")

print("Prediction completed successfully!")

print("==========================================")
