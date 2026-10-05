import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(BACKEND_ROOT, "data", "network_devices.csv")
OUTPUTS_DIR = os.path.join(BACKEND_ROOT, "outputs")

# Load dataset
df = pd.read_csv(DATA_PATH)

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
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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
    os.path.join(OUTPUTS_DIR, "failure_distribution.png"),
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
    os.path.join(OUTPUTS_DIR, "cpu_vs_failure.png"),
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
    os.path.join(OUTPUTS_DIR, "memory_vs_failure.png"),
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
    os.path.join(OUTPUTS_DIR, "temperature_vs_failure.png"),
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
    os.path.join(OUTPUTS_DIR, "correlation_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nEDA completed successfully!")
print(f"Graphs saved in {OUTPUTS_DIR} folder.")