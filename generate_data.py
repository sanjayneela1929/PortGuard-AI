import numpy as np
import pandas as pd
from pathlib import Path


# Using a fixed random seed ensures that we can
# generate the same dataset again whenever required.

np.random.seed(42)



ports = {
    "Chennai": 30,
    "Mumbai": 40,
    "Visakhapatnam": 25,
    "Kolkata": 28,
    "Cochin": 22
}



number_of_records = 500

data = []

for i in range(number_of_records):

    # Select a random port
    port_name = np.random.choice(list(ports.keys()))

    # Get the handling capacity of the selected port
    port_capacity = ports[port_name]

    # Generate incoming vessel arrivals
    vessel_arrivals = np.random.randint(5, 51)

    # Generate queue length
    queue_length = np.random.randint(0, 31)

    # Generate weather severity between 0 and 10
    weather_severity = np.random.randint(0, 11)

    # Generate historical average delay
    historical_avg_delay = round(
        np.random.uniform(0, 8), 2
    )

    # Generate arrival hour
    hour_of_day = np.random.randint(0, 24)

    # Generate month
    month = np.random.randint(1, 13)



    # Calculate excess vessel arrivals
    excess_arrivals = max(
        0, vessel_arrivals - port_capacity
    )

    # Identify peak arrival hours
    peak_hour_factor = (
        0.8 if hour_of_day in [7, 8, 9, 17, 18, 19]
        else 0
    )

    # Identify monsoon-season months
    seasonal_factor = (
        0.5 if month in [6, 7, 8, 9]
        else 0
    )

    # Random variation to simulate unpredictable conditions
    random_noise = np.random.normal(0, 1.2)

    # Combine the congestion-related factors
    delay_hours = (
        1.0
        + (excess_arrivals * 0.22)
        + (queue_length * 0.16)
        + (weather_severity * 0.28)
        + (historical_avg_delay * 0.30)
        + peak_hour_factor
        + seasonal_factor
        + random_noise
    )

    # Delay cannot be negative
    delay_hours = max(0, delay_hours)

    delay_hours = round(delay_hours, 2)



    congestion = 1 if delay_hours >= 6 else 0




    data.append({
        "port_name": port_name,
        "vessel_arrivals": vessel_arrivals,
        "port_capacity": port_capacity,
        "queue_length": queue_length,
        "weather_severity": weather_severity,
        "historical_avg_delay": historical_avg_delay,
        "hour_of_day": hour_of_day,
        "month": month,
        "delay_hours": delay_hours,
        "congestion": congestion
    })


df = pd.DataFrame(data)



# Get the project directory
project_directory = Path(__file__).resolve().parent

# Create data directory if it does not exist
data_directory = project_directory / "data"
data_directory.mkdir(exist_ok=True)

# Save the dataset
output_path = data_directory / "port_congestion_data.csv"

df.to_csv(output_path, index=False)


print(" PORTGUARD AI - DATA GENERATION COMPLETE")


print(f"\nTotal records generated: {len(df)}")

print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nCongestion distribution:")
print(df["congestion"].value_counts())

print("\nCongestion percentage:")
print(df["congestion"].value_counts(normalize=True) * 100)

print("\nDataset saved successfully!")
print(f"File location: {output_path}")

