
# ============================================================
# PORTGUARD AI
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

project_directory = Path(__file__).resolve().parent

data_path = project_directory / "data" / "port_congestion_data.csv"

reports_directory = project_directory / "reports"
reports_directory.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(data_path)

print("\n==========================================")
print(" PORTGUARD AI - DATA ANALYSIS")
print("==========================================")

print("\nDataset loaded successfully!")


# ------------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 5. CHECK DUPLICATE RECORDS
# ------------------------------------------------------------

print("\nDuplicate Records:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 6. CONGESTION DISTRIBUTION
# ------------------------------------------------------------

print("\nCongestion Distribution:")
print(df["congestion"].value_counts())

print("\nCongestion Percentage:")
print(
    (df["congestion"].value_counts(normalize=True) * 100).round(2)
)


# ------------------------------------------------------------
# 7. AVERAGE DELAY BY PORT
# ------------------------------------------------------------

print("\nAverage Delay by Port:")

print(
    df.groupby("port_name")["delay_hours"]
    .mean()
    .round(2)
)


# ------------------------------------------------------------
# 8. VISUALIZATION SETTINGS
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# 9. CONGESTION DISTRIBUTION CHART
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="congestion"
)

plt.title("Congestion Distribution")
plt.xlabel("Congestion (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    reports_directory / "congestion_distribution.png"
)

plt.show()


# ------------------------------------------------------------
# 10. DELAY BY PORT
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="port_name",
    y="delay_hours"
)

plt.title("Shipment Delay Distribution by Port")
plt.xlabel("Port Name")
plt.ylabel("Delay (Hours)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    reports_directory / "delay_by_port.png"
)

plt.show()


# ------------------------------------------------------------
# 11. VESSEL ARRIVALS VS DELAY
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="vessel_arrivals",
    y="delay_hours",
    hue="congestion",
    palette="Set1"
)

plt.title("Vessel Arrivals vs Shipment Delay")
plt.xlabel("Vessel Arrivals")
plt.ylabel("Delay (Hours)")

plt.tight_layout()

plt.savefig(
    reports_directory / "arrivals_vs_delay.png"
)

plt.show()


# ------------------------------------------------------------
# 12. CORRELATION HEATMAP
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    reports_directory / "correlation_heatmap.png"
)

plt.show()


# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

print(" DATA ANALYSIS COMPLETED")

print(f"Total records analysed: {len(df)}")

print(f"Total columns: {len(df.columns)}")

print(f"Congested records: {(df['congestion'] == 1).sum()}")

print(f"Non-congested records: {(df['congestion'] == 0).sum()}")

print("\nCharts saved inside the reports folder.")


