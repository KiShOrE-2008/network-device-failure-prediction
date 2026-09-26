"""
src/services/incident_service.py
----------------------------------
Incident Lifecycle Management Service.

Enforces strict incident lifecycle:
Prediction -> Risk HIGH/CRITICAL? -> YES -> Existing active/acknowledged incident? -> YES: UPDATE / NO: CREATE
Transitions: ACTIVE -> ACKNOWLEDGED -> RESOLVED.
"""

import os
import sys
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import history_store


class IncidentService:
    def __init__(self):
        history_store.init_db()

    def process_prediction(self, device_id: str, telemetry: dict, result: dict) -> None:
        """Processes prediction and creates/updates incident if risk is HIGH or CRITICAL."""
        history_store.log_prediction(device_id, telemetry, result)

    def process_fleet_alerts(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes batch predictions and updates active incident alerts."""
        active_count = 0
        for p in predictions:
            if p.get("risk") in ["HIGH", "CRITICAL"] or p.get("is_anomaly"):
                dev_id = str(p.get("device_id", "UNKNOWN"))
                telemetry = p.get("telemetry", {})
                self.process_prediction(dev_id, telemetry, p)
                active_count += 1
        return {"total_active_alerts": active_count}

    def get_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:

        """Returns all active/acknowledged incidents."""
        return history_store.get_active_alerts(limit=limit)

    def acknowledge_incident(self, incident_id: int) -> bool:
        """Transitions incident status to ACKNOWLEDGED."""
        return history_store.acknowledge_alert(incident_id)

    def resolve_incident(self, incident_id: int) -> bool:
        """Transitions incident status to RESOLVED."""
        return history_store.resolve_alert(incident_id)
