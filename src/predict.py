import pandas as pd
import joblib
import sys

def get_input(prompt, default, cast_func=float, validate_func=None):
    while True:
        try:
            val_str = input(f"{prompt} [Default: {default}]: ").strip()
            if not val_str:
                return default
            val = cast_func(val_str)
            if validate_func and not validate_func(val):
                print("  ⚠️ Input out of valid range/format. Please try again.")
                continue
            return val
        except ValueError:
            print("  ⚠️ Invalid input format. Please enter a valid value.")

def main():
    # Load trained model
    try:
        model = joblib.load("models/failure_model.pkl")
    except FileNotFoundError:
        print("Error: Trained model 'models/failure_model.pkl' not found.")
        print("Please train the model first using: python src/train_model.py")
        sys.exit(1)

    # If --non-interactive is passed, use the hardcoded demo values directly
    if len(sys.argv) > 1 and sys.argv[1] == "--non-interactive":
        device_type = "Router"
        cpu_usage = 92.0
        memory_usage = 94.0
        temperature = 78.0
        uptime = 20.0
        interface_errors = 156
        packet_loss = 8.2
        bandwidth_usage = 95.0
        log_errors = 20
    else:
        print("=" * 50)
        print("NETWORK DEVICE FAILURE PREDICTION - INTERACTIVE MODE")
        print("=" * 50)
        print("Please enter telemetry values or press [Enter] to use defaults.")
        print("-" * 50)

        device_type = get_input(
            "Device Type (Router/Switch)",
            "Router",
            cast_func=str,
            validate_func=lambda x: x.lower() in ["router", "switch"]
        )
        # Standardize casing
        device_type = "Router" if device_type.lower() == "router" else "Switch"

        cpu_usage = get_input(
            "CPU Usage (%) (0-100)",
            92.0,
            validate_func=lambda x: 0 <= x <= 100
        )

        memory_usage = get_input(
            "Memory Usage (%) (0-100)",
            94.0,
            validate_func=lambda x: 0 <= x <= 100
        )

        temperature = get_input(
            "Temperature (°C) (0-150)",
            78.0,
            validate_func=lambda x: 0 <= x <= 150
        )

        uptime = get_input(
            "Uptime (days) (>=0)",
            20.0,
            validate_func=lambda x: x >= 0
        )

        interface_errors = get_input(
            "Interface Errors (count >=0)",
            156,
            cast_func=int,
            validate_func=lambda x: x >= 0
        )

        packet_loss = get_input(
            "Packet Loss (%) (0-100)",
            8.2,
            validate_func=lambda x: 0 <= x <= 100
        )

        bandwidth_usage = get_input(
            "Bandwidth Usage (%) (0-100)",
            95.0,
            validate_func=lambda x: 0 <= x <= 100
        )

        log_errors = get_input(
            "Log Errors (count >=0)",
            20,
            cast_func=int,
            validate_func=lambda x: x >= 0
        )

    # New device data DataFrame
    device = pd.DataFrame([
        {
            "Device_Type": device_type,
            "CPU_Usage": cpu_usage,
            "Memory_Usage": memory_usage,
            "Temperature": temperature,
            "Uptime": uptime,
            "Interface_Errors": interface_errors,
            "Packet_Loss": packet_loss,
            "Bandwidth_Usage": bandwidth_usage,
            "Log_Errors": log_errors
        }
    ])

    # Prediction
    prediction = model.predict(device)[0]

    # Probability
    probability = model.predict_proba(device)[0][1]

    # Risk classification
    if probability < 0.30:
        risk = "LOW"
    elif probability < 0.70:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Device Type:        {device_type}")
    print(f"CPU Usage:          {cpu_usage}%")
    print(f"Memory Usage:       {memory_usage}%")
    print(f"Temperature:        {temperature}°C")
    print(f"Uptime:             {uptime} days")
    print(f"Interface Errors:   {interface_errors}")
    print(f"Packet Loss:        {packet_loss}%")
    print(f"Bandwidth Usage:    {bandwidth_usage}%")
    print(f"Log Errors:         {log_errors}")
    print("-" * 50)
    print(f"Failure Probability: {probability * 100:.2f}%")
    print(f"Risk Level:          {risk}")
    print("-" * 50)

    if risk == "HIGH":
        print("⚠️ WARNING: Preventive maintenance recommended.")
    elif risk == "MEDIUM":
        print("💡 NOTE: Monitor device closely for further degradation.")
    else:
        print("✅ Device condition is currently healthy/acceptable.")
    print("=" * 50)

if __name__ == "__main__":
    main()