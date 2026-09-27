"""
src/services/anomaly_service.py
--------------------------------
Anomaly Detection Service wrapping Isolation Forest detector and baseline statistics.
"""

import os
import sys
import pandas as pd
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import anomaly_detection


class AnomalyService:
    def __init__(self, model_path: str = "models/anomaly_model.pkl"):
        self.model_path = os.path.join(WORKSPACE_ROOT, model_path) if not os.path.isabs(model_path) else model_path

    def predict(self, telemetry: dict) -> Dict[str, Any]:
        """Runs anomaly evaluation on single telemetry observation."""
        return anomaly_detection.predict_anomaly(telemetry, model_path=self.model_path)

    def predict_batch(self, df: pd.DataFrame) -> List[float]:
        """Runs batch anomaly evaluation on DataFrame."""
        return anomaly_detection.predict_anomaly_batch(df, model_path=self.model_path)
