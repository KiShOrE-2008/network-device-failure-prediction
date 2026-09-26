"""
tests/test_incident_lifecycle.py
---------------------------------
Incident Lifecycle & Deduplication Test Suite.
Verifies that 100 repeated predictions produce 1 updated incident (NOT 100 duplicate incidents),
and tests ACTIVE -> ACKNOWLEDGED -> RESOLVED transitions.
"""

import os
import sys
import tempfile
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(BACKEND_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import history_store
from services.incident_service import IncidentService


def test_repeated_predictions_deduplicate_to_single_incident():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_noc.db")
        history_store.DB_PATH = test_db
        history_store.init_db()

        service = IncidentService()

        telemetry = {"CPU_Usage": 95.0, "Temperature": 88.0, "Interface_Errors": 40}
        result = {
            "failure_probability": 0.88,
            "risk": "CRITICAL",
            "predicted_failure": "THERMAL",
            "anomaly_score": 75.0,
            "is_anomaly": True,
            "health_score": 15.0
        }

        # Fire 100 repeated predictions for DEV-001
        for _ in range(100):
            service.process_prediction("DEV-001", telemetry, result)

        incidents = service.get_incidents()
        dev1_incidents = [inc for inc in incidents if inc["device_id"] == "DEV-001"]

        # Crucial assertion: exactly 1 incident created and updated in place!
        assert len(dev1_incidents) == 1, f"Expected 1 deduplicated incident, found {len(dev1_incidents)}"
        assert dev1_incidents[0]["status"] == "ACTIVE"


def test_incident_lifecycle_transitions():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_noc_lifecycle.db")
        history_store.DB_PATH = test_db
        history_store.init_db()

        service = IncidentService()

        telemetry = {"CPU_Usage": 92.0, "Temperature": 85.0}
        result = {
            "failure_probability": 0.82,
            "risk": "HIGH",
            "predicted_failure": "HARDWARE",
            "anomaly_score": 72.0,
            "is_anomaly": True
        }

        service.process_prediction("DEV-002", telemetry, result)
        incidents = service.get_incidents()
        dev2_inc = [inc for inc in incidents if inc["device_id"] == "DEV-002"][0]
        inc_id = dev2_inc["id"]

        assert dev2_inc["status"] == "ACTIVE"

        # 1. ACTIVE -> ACKNOWLEDGED
        ack_success = service.acknowledge_incident(inc_id)
        assert ack_success is True

        incidents = service.get_incidents()
        ack_inc = [inc for inc in incidents if inc["id"] == inc_id][0]
        assert ack_inc["status"] == "ACKNOWLEDGED"

        # 2. ACKNOWLEDGED -> RESOLVED
        res_success = service.resolve_incident(inc_id)
        assert res_success is True

        # Active incidents query should no longer include RESOLVED incident
        active_incidents = service.get_incidents()
        assert not any(inc["id"] == inc_id for inc in active_incidents)
