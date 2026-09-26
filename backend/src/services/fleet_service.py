"""
src/services/fleet_service.py
------------------------------
Fleet Intelligence Service for NetGuard NOC.
Wraps FleetPredictor to manage fleet-wide inference, summary metrics,
and risk ranking.
"""

from typing import Dict, Any, List
from fleet_predictor import FleetPredictor


class FleetService:
    def __init__(self,
                 failure_model_path: str = "models/failure_model.pkl",
                 diagnostic_model_path: str = "models/diagnostic_model.pkl",
                 anomaly_model_path: str = "models/anomaly_model.pkl"):
        self.predictor = FleetPredictor(
            failure_model_path=failure_model_path,
            diagnostic_model_path=diagnostic_model_path,
            anomaly_model_path=anomaly_model_path
        )

    def get_fleet_predictions(self) -> List[Dict[str, Any]]:
        return self.predictor.predict_all()

    def get_fleet_summary(self) -> Dict[str, Any]:
        preds = self.get_fleet_predictions()
        return self.predictor.get_fleet_summary(preds)
