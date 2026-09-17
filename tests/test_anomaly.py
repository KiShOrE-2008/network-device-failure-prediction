import pytest
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import anomaly_detection

def test_predict_anomaly_bounds():
    telemetry = {
        "CPU_Usage": 95.0,
        "Memory_Usage": 90.0,
        "Temperature": 85.0,
        "CPU_Trend": 20.0,
        "Temperature_Trend": 12.0
    }
    res = anomaly_detection.predict_anomaly(telemetry)
    assert "anomaly_score" in res
    assert 0.0 <= res["anomaly_score"] <= 100.0
    assert isinstance(res["is_anomaly"], bool)
