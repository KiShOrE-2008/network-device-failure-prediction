import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/network_devices.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nFailure Distribution:")
print(df["Failed"].value_counts())

# Create output folder
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 1. Failure Distribution
# -----------------------------
plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Failed"
)

plt.title("Healthy vs Failed Devices")
plt.xlabel("Failure Status")
plt.ylabel("Number of Devices")

plt.savefig(
    "outputs/failure_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# 2. CPU vs Failure
# -----------------------------
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Failed",
    y="CPU_Usage"
)

plt.title("CPU Usage vs Device Failure")
plt.xlabel("Failure Status")
plt.ylabel("CPU Usage (%)")

plt.savefig(
    "outputs/cpu_vs_failure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# 3. Memory vs Failure
# -----------------------------
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Failed",
    y="Memory_Usage"
)

plt.title("Memory Usage vs Device Failure")
plt.xlabel("Failure Status")
plt.ylabel("Memory Usage (%)")

plt.savefig(
    "outputs/memory_vs_failure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# 4. Temperature vs Failure
# -----------------------------
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Failed",
    y="Temperature"
)

plt.title("Temperature vs Device Failure")
plt.xlabel("Failure Status")
plt.ylabel("Temperature (°C)")

plt.savefig(
    "outputs/temperature_vs_failure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# 5. Correlation Heatmap
# -----------------------------
numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

plt.figure(
    figsize=(12, 8)
)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap"
)

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nEDA completed successfully!")
print("Graphs saved in outputs/ folder.")