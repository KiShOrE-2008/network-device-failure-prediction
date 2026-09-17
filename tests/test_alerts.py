import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import history_store


def test_alert_deduplication():
    device_id = "TEST-DEV-9999"
    telemetry = {"CPU_Usage": 92.0, "Temperature": 85.0}

    # Initial high-risk prediction
    res1 = {"failure_probability": 0.88, "risk": "CRITICAL", "predicted_failure": "THERMAL", "health_score": 15.0}
    history_store.log_prediction(device_id, telemetry, res1)

    active_alerts1 = [a for a in history_store.get_active_alerts() if a["device_id"] == device_id]
    assert len(active_alerts1) == 1
    alert_id = active_alerts1[0]["id"]

    # Subsequent prediction for same device should update existing active alert instead of creating duplicate
    res2 = {"failure_probability": 0.92, "risk": "CRITICAL", "predicted_failure": "THERMAL", "health_score": 12.0}
    history_store.log_prediction(device_id, telemetry, res2)

    active_alerts2 = [a for a in history_store.get_active_alerts() if a["device_id"] == device_id]
    assert len(active_alerts2) == 1
    assert active_alerts2[0]["id"] == alert_id


def test_alert_lifecycle():
    device_id = "TEST-DEV-8888"
    telemetry = {"CPU_Usage": 95.0, "Temperature": 88.0}
    res = {"failure_probability": 0.90, "risk": "CRITICAL", "predicted_failure": "HARDWARE", "health_score": 10.0}
    history_store.log_prediction(device_id, telemetry, res)

    alerts = [a for a in history_store.get_active_alerts() if a["device_id"] == device_id]
    assert len(alerts) == 1
    alert_id = alerts[0]["id"]

    # Test acknowledge
    ack_res = history_store.acknowledge_alert(alert_id)
    assert ack_res is True

    # Test resolve
    res_res = history_store.resolve_alert(alert_id)
    assert res_res is True
