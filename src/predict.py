import pandas as pd
import joblib

# Load trained model
model = joblib.load(
    "models/failure_model.pkl"
)

# New device data
device = pd.DataFrame([
    {
        "Device_Type": "Router",
        "CPU_Usage": 92,
        "Memory_Usage": 94,
        "Temperature": 78,
        "Uptime": 20,
        "Interface_Errors": 156,
        "Packet_Loss": 8.2,
        "Bandwidth_Usage": 95,
        "Log_Errors": 20
    }
])

# Prediction
prediction = model.predict(
    device
)[0]

# Probability
probability = model.predict_proba(
    device
)[0][1]

# Risk classification
if probability < 0.30:

    risk = "LOW"

elif probability < 0.70:

    risk = "MEDIUM"

else:

    risk = "HIGH"


print("\n" + "=" * 50)

print("NETWORK DEVICE FAILURE PREDICTION")

print("=" * 50)

print(
    f"Device Type: {device['Device_Type'].iloc[0]}"
)

print(
    f"CPU Usage: {device['CPU_Usage'].iloc[0]}%"
)

print(
    f"Memory Usage: {device['Memory_Usage'].iloc[0]}%"
)

print(
    f"Temperature: {device['Temperature'].iloc[0]}°C"
)

print(
    f"Interface Errors: {device['Interface_Errors'].iloc[0]}"
)

print(
    f"Packet Loss: {device['Packet_Loss'].iloc[0]}%"
)

print(
    f"\nFailure Probability: {probability * 100:.2f}%"
)

print(
    f"Risk Level: {risk}"
)

if risk == "HIGH":

    print(
        "\n⚠️ WARNING: Preventive maintenance recommended."
    )

else:

    print(
        "\nDevice condition is currently acceptable."
    )