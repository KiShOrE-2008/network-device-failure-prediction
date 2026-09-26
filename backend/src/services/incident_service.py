"""
src/services/incident_service.py
---------------------------------
Incident Lifecycle Service for NetGuard NOC.
Manages incident deduplication, DB persistence, status transitions,
and active NOC alerts.
"""

from typing import Dict, Any, List
from history_store import HistoryStore


class IncidentService:
    def __init__(self, db_store: HistoryStore = None):
        self.db_store = db_store or HistoryStore()

    def process_fleet_alerts(self, fleet_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deduplicates fleet alerts and records incidents into database history store.
        Returns newly generated and active alerts.
        """
        active_incidents = []
        new_alerts = 0

        for pred in fleet_predictions:
            prob = pred.get("failure_probability")
            prob_val = prob if prob is not None else (pred.get("heuristic_risk_score", 0) / 100.0)
            risk = pred.get("risk", "LOW")

            if risk in ["HIGH", "CRITICAL"] or prob_val >= 0.70:
                is_new = self.db_store.record_incident(
                    device_id=pred["device_id"],
                    failure_type=pred.get("predicted_failure", "THERMAL"),
                    failure_probability=prob_val,
                    severity=risk,
                    recommended_actions=pred.get("recommended_actions", []),
                    telemetry_snapshot=pred.get("telemetry", {})
                )
                if is_new:
                    new_alerts += 1

                active_incidents.append(pred)

        return {
            "total_active_alerts": len(active_incidents),
            "newly_created_alerts": new_alerts,
            "active_incidents": active_incidents
        }

    def get_incidents(self, status: str = None) -> List[Dict[str, Any]]:
        return self.db_store.get_incidents(status=status)

    def update_incident_status(self, incident_id: int, new_status: str, notes: str = None) -> bool:
        return self.db_store.update_incident_status(incident_id=incident_id, status=new_status, notes=notes)
