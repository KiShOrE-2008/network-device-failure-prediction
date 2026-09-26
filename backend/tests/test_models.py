import os
import sys
import joblib
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

MODEL_PATH = "models/failure_model.pkl"
DIAGNOSTIC_PATH = "models/diagnostic_model.pkl"


def test_model_files_exist():
    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(DIAGNOSTIC_PATH)


def test_binary_model_inference():
    model = joblib.load(MODEL_PATH)
    test_row = pd.DataFrame([{
        "Device_Type": "Router",
        "CPU_Usage": 90.0,
        "Memory_Usage": 85.0,
        "Temperature": 80.0,
        "Uptime": 100.0,
        "Interface_Errors": 50,
        "Packet_Loss": 4.0,
        "Bandwidth_Usage": 80.0,
        "Log_Errors": 10,
        "Syslog_Critical_Count": 2,
        "CPU_5step_avg": 85.0,
        "Memory_5step_avg": 80.0,
        "Temperature_5step_avg": 75.0,
        "Error_5step_avg": 40.0,
        "PacketLoss_5step_avg": 3.0,
        "CPU_Trend": 15.0,
        "Memory_Trend": 5.0,
        "Temperature_Trend": 8.0,
        "Error_Trend": 10.0,
        "PacketLoss_Trend": 1.0,
        "CPU_Spike": 1,
        "Temperature_Spike": 1,
        "Error_Spike": 1
    }])
    prob = float(model.predict_proba(test_row)[0][1])
    assert 0.0 <= prob <= 1.0
