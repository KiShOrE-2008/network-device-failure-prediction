"""
src/services/fleet_service.py
------------------------------
Fleet Intelligence & Batch Predictor Service.
"""

import os
import sys
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fleet_predictor import FleetPredictor


class FleetService:
    def __init__(self):
        self.predictor = FleetPredictor()

    def get_fleet_predictions(self) -> List[Dict[str, Any]]:
        return self.predictor.predict_all()

    def get_fleet_summary(self) -> Dict[str, Any]:
        predictions = self.get_fleet_predictions()
        return self.predictor.get_fleet_summary(predictions)
