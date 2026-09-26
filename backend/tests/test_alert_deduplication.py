import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import history_store


def test_repeated_prediction_does_not_create_duplicate_alert():
    """Verify multiple predictions for a high-risk device update single active incident instead of creating duplicates."""
    device_id = "DEV-TEST-DEDUP-001"
    telemetry = {"cpu": 95.0, "temperature": 85.0}
    
    high_risk_result = {
        "failure_probability": 0.92,
        "risk": "CRITICAL",
        "predicted_failure": "THERMAL",
        "health_score": 15.0,
        "anomaly_score": 85.0
    }

    # Log prediction 3 times consecutively for same device
    history_store.log_prediction(device_id, telemetry, high_risk_result)
    history_store.log_prediction(device_id, telemetry, high_risk_result)
    history_store.log_prediction(device_id, telemetry, high_risk_result)

    alerts = history_store.get_active_alerts(limit=100)
    dev_alerts = [a for a in alerts if a["device_id"] == device_id]

    assert len(dev_alerts) == 1, f"Expected exactly 1 active alert for {device_id}, got {len(dev_alerts)}"
    assert dev_alerts[0]["status"] == "ACTIVE"


def test_acknowledged_incident_is_updated():
    """Verify repeated prediction on ACKNOWLEDGED incident updates it without creating a new ACTIVE incident."""
    device_id = "DEV-TEST-DEDUP-002"
    telemetry = {"cpu": 90.0, "temperature": 80.0}
    result = {"failure_probability": 0.88, "risk": "CRITICAL", "predicted_failure": "MEMORY", "health_score": 20.0}

    # Create incident
    history_store.log_prediction(device_id, telemetry, result)
    alerts = history_store.get_active_alerts(limit=100)
    target_alert = next(a for a in alerts if a["device_id"] == device_id)
    alert_id = target_alert["id"]

    # Acknowledge incident
    ack_res = history_store.acknowledge_alert(alert_id)
    assert ack_res is True

    # Log prediction again
    history_store.log_prediction(device_id, telemetry, result)

    alerts_after = history_store.get_active_alerts(limit=100)
    dev_alerts_after = [a for a in alerts_after if a["device_id"] == device_id]

    assert len(dev_alerts_after) == 1
    assert dev_alerts_after[0]["id"] == alert_id
    assert dev_alerts_after[0]["status"] == "ACKNOWLEDGED"


def test_resolved_incident_can_reopen_if_defined():
    """Verify that once an incident is RESOLVED, a new critical prediction opens a fresh ACTIVE incident."""
    device_id = "DEV-TEST-DEDUP-003"
    telemetry = {"cpu": 90.0, "temperature": 80.0}
    result = {"failure_probability": 0.88, "risk": "CRITICAL", "predicted_failure": "INTERFACE", "health_score": 20.0}

    # Log initial prediction
    history_store.log_prediction(device_id, telemetry, result)
    alerts = history_store.get_active_alerts(limit=100)
    target_alert = next(a for a in alerts if a["device_id"] == device_id)
    alert_id = target_alert["id"]

    # Resolve incident
    history_store.resolve_alert(alert_id)

    # Log new critical prediction
    history_store.log_prediction(device_id, telemetry, result)

    alerts_after = history_store.get_active_alerts(limit=100)
    dev_alerts_after = [a for a in alerts_after if a["device_id"] == device_id]

    # Should have 1 active alert (the new one)
    assert len(dev_alerts_after) == 1
    assert dev_alerts_after[0]["id"] != alert_id
    assert dev_alerts_after[0]["status"] == "ACTIVE"
