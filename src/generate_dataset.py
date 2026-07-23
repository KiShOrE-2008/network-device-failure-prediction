import pandas as pd
import numpy as np
import os

# Reproducibility
np.random.seed(42)

# Number of records
NUM_RECORDS = 10000

# Create data directory
os.makedirs("data", exist_ok=True)

# Device IDs
device_ids = [
    f"DEV-{i:05d}"
    for i in range(1, NUM_RECORDS + 1)
]

# Device types
device_types = np.random.choice(
    ["Router", "Switch"],
    size=NUM_RECORDS
)

# CPU usage
cpu_usage = np.random.uniform(
    10, 100, NUM_RECORDS
)

# Memory usage
memory_usage = np.random.uniform(
    20, 100, NUM_RECORDS
)

# Temperature in Celsius
temperature = np.random.uniform(
    25, 90, NUM_RECORDS
)

# Uptime in days
uptime = np.random.uniform(
    1, 1000, NUM_RECORDS
)

# Interface errors
interface_errors = np.random.poisson(
    lam=20,
    size=NUM_RECORDS
)

# Packet loss percentage
packet_loss = np.random.uniform(
    0, 10, NUM_RECORDS
)

# Bandwidth usage percentage
bandwidth_usage = np.random.uniform(
    10, 100, NUM_RECORDS
)

# System log errors
log_errors = np.random.poisson(
    lam=5,
    size=NUM_RECORDS
)

# ------------------------------------------------
# Create failure probability
# ------------------------------------------------

failure_score = (
    0.25 * (cpu_usage / 100)
    + 0.20 * (memory_usage / 100)
    + 0.20 * (temperature / 90)
    + 0.10 * (interface_errors / 100)
    + 0.10 * (packet_loss / 10)
    + 0.10 * (bandwidth_usage / 100)
    + 0.05 * (log_errors / 30)
)

# Add random noise
failure_score += np.random.normal(
    0,
    0.05,
    NUM_RECORDS
)

# Convert score into failure label
failed = (
    failure_score > 0.65
).astype(int)

# ------------------------------------------------
# Create DataFrame
# ------------------------------------------------

df = pd.DataFrame({
    "Device_ID": device_ids,
    "Device_Type": device_types,
    "CPU_Usage": np.round(cpu_usage, 2),
    "Memory_Usage": np.round(memory_usage, 2),
    "Temperature": np.round(temperature, 2),
    "Uptime": np.round(uptime, 2),
    "Interface_Errors": interface_errors,
    "Packet_Loss": np.round(packet_loss, 2),
    "Bandwidth_Usage": np.round(bandwidth_usage, 2),
    "Log_Errors": log_errors,
    "Failed": failed
})

# Save dataset
file_path = "data/network_devices.csv"

df.to_csv(
    file_path,
    index=False
)

# ------------------------------------------------
# Display information
# ------------------------------------------------

print("=" * 50)
print("Network Device Dataset Generated")
print("=" * 50)

print(f"Total Records: {len(df)}")

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFailure Distribution:")
print(df["Failed"].value_counts())

print("\nFirst 5 Records:")
print(df.head())

print(f"\nDataset saved to: {file_path}")