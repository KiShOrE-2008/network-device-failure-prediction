"""
src/services/anomaly_service.py
--------------------------------
Anomaly Detection Service for NetGuard NOC.
Delegates to single-source-of-truth IsolationForest anomaly detector.
"""

import os
from typing import Dict, Any, List
import pandas as pd

from anomaly_detection import predict_anomaly, predict_anomaly_batch


class AnomalyService:
    def __init__(self, model_path: str = "models/anomaly_model.pkl"):
        self.model_path = model_path

    def analyze_device(self, telemetry: dict) -> Dict[str, Any]:
        return predict_anomaly(telemetry, model_path=self.model_path)

    def analyze_batch(self, df: pd.DataFrame) -> List[float]:
        return predict_anomaly_batch(df, model_path=self.model_path)
