import sys
import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import health_engine
import shap_explainer
import anomaly_detection
from intelligence import diagnostic_engine

def get_input(prompt, default, cast_func=float, validate_func=None):
    while True:
        try:
            val_str = input(f"{prompt} [Default: {default}]: ").strip()
            if not val_str:
                return default
            val = cast_func(val_str)
            if validate_func and not validate_func(val):
                print("  ⚠️ Input out of valid range. Please try again.")
                continue
            return val
        except ValueError:
            print("  ⚠️ Invalid input format. Please enter a valid value.")

def main():
    model_path = os.path.join(BASE_DIR, "..", "models", "failure_model.pkl")
    diag_path = os.path.join(BASE_DIR, "..", "models", "diagnostic_model.pkl")
    anom_path = os.path.join(BASE_DIR, "..", "models", "anomaly_model.pkl")

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Trained model '{model_path}' not found.")
        print("Please train models first using: python src/train_model.py")
        sys.exit(1)

    diag_model = joblib.load(diag_path) if os.path.exists(diag_path) else None
    anom_model = joblib.load(anom_path) if os.path.exists(anom_path) else None

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
        cpu_trend = 22.0
        temp_trend = 12.0
    else:
        print("=" * 60)
        print("NETGUARD NOC — CLI TELEMETRY DIAGNOSTICS & ANOMALY DETECTOR")
        print("=" * 60)
        print("Enter device telemetry values or press [Enter] for defaults.")
        print("-" * 60)

        device_type = get_input("Device Type (Router/Switch)", "Router", cast_func=str, validate_func=lambda x: x.lower() in ["router", "switch"])
        device_type = "Router" if device_type.lower() == "router" else "Switch"
        cpu_usage = get_input("CPU Usage (%) (0-100)", 92.0, validate_func=lambda x: 0 <= x <= 100)
        memory_usage = get_input("Memory Usage (%) (0-100)", 94.0, validate_func=lambda x: 0 <= x <= 100)
        temperature = get_input("Temperature (°C) (0-150)", 78.0, validate_func=lambda x: 0 <= x <= 150)
        uptime = get_input("Uptime (days) (>=0)", 20.0, validate_func=lambda x: x >= 0)
        interface_errors = get_input("Interface Errors (count >=0)", 156, cast_func=int, validate_func=lambda x: x >= 0)
        packet_loss = get_input("Packet Loss (%) (0-100)", 8.2, validate_func=lambda x: 0 <= x <= 100)
        bandwidth_usage = get_input("Bandwidth Usage (%) (0-100)", 95.0, validate_func=lambda x: 0 <= x <= 100)
        log_errors = get_input("Log Errors (count >=0)", 20, cast_func=int, validate_func=lambda x: x >= 0)
        cpu_trend = 20.0
        temp_trend = 10.0

    telemetry = {
        "Device_Type": device_type,
        "CPU_Usage": cpu_usage,
        "Memory_Usage": memory_usage,
        "Temperature": temperature,
        "Uptime": uptime,
        "Interface_Errors": interface_errors,
        "Packet_Loss": packet_loss,
        "Bandwidth_Usage": bandwidth_usage,
        "Log_Errors": log_errors,
        "CPU_Trend": cpu_trend,
        "Memory_Trend": 5.0,
        "Temperature_Trend": temp_trend,
        "Error_Trend": 20.0,
        "PacketLoss_Trend": 2.0
    }

    df_input = pd.DataFrame([telemetry])
    probability = float(model.predict_proba(df_input)[0][1])

    ml_type = "NONE"
    if diag_model is not None:
        try:
            pred_idx = int(diag_model.predict(df_input)[0])
            if hasattr(diag_model, 'label_classes_') and pred_idx < len(diag_model.label_classes_):
                ml_type = str(diag_model.label_classes_[pred_idx])
        except Exception:
            pass

    diag_res = diagnostic_engine.diagnose_failure_mode(telemetry, ml_type, probability)
    anom_res = anomaly_detection.predict_anomaly(telemetry, anom_model, anom_path)
    health_score = health_engine.compute_health_score(telemetry)
    shap_causes = shap_explainer.get_shap_causes(telemetry, top_n=5)

    risk = "HIGH" if probability >= 0.65 else "MEDIUM" if probability >= 0.30 else "LOW"

    print("\n" + "=" * 60)
    print("NETGUARD NOC DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Device Type:           {device_type}")
    print(f"CPU Utilization:       {cpu_usage}% (Trend: +{cpu_trend}%)")
    print(f"Memory Utilization:    {memory_usage}%")
    print(f"Temperature:           {temperature}°C (Trend: +{temp_trend}°C)")
    print(f"Interface Errors:      {interface_errors} errors")
    print(f"Packet Loss:           {packet_loss}%")
    print("-" * 60)
    print(f"Failure Probability:   {probability * 100:.2f}%")
    print(f"Health Score:          {health_score} / 100")
    print(f"Risk Level:            {risk}")
    print(f"Diagnosed Mode:        {diag_res['diagnosed_failure_type']}")
    print(f"Anomaly Score:         {anom_res['anomaly_score']}% ({'ANOMALY DETECTED' if anom_res['is_anomaly'] else 'NORMAL'})")
    print("-" * 60)
    print("Diagnosis Narrative:")
    print(f"  {diag_res['description']}")
    print("\nRecommended Maintenance Actions:")
    for act in diag_res['recommended_actions']:
        print(f"  • {act}")
    
    if shap_causes:
        print("\nPrimary Risk Drivers (SHAP Attribution):")
        for c in shap_causes[:3]:
            print(f"  • {c['feature']}: {c['contribution']:+.4f} ({c['direction']})")

    print("=" * 60)

if __name__ == "__main__":
    main()