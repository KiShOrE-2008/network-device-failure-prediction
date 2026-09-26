"""
src/services/inference_service.py
----------------------------------
Inference Service for NetGuard NOC.
Provides single-device failure prediction, feature matrix preparation,
and model metadata validation.
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, Optional

from feature_engineering import FEATURE_COLUMNS, prepare_feature_matrix
from model_registry import ModelRegistry, ModelSchemaMismatchError


class InferenceService:
    def __init__(self, failure_model_path: str = "models/failure_model.pkl", metadata_path: str = "models/model_metadata.json"):
        self.failure_model_path = failure_model_path
        self.metadata_path = metadata_path
        self.registry = ModelRegistry(metadata_path=metadata_path)
        self.failure_model = None
        self.model_status = "UNAVAILABLE"
        self.load_model()

    def load_model(self):
        """Loads and validates failure model contract."""
        if os.path.exists(self.failure_model_path):
            try:
                self.failure_model, _ = self.registry.validate_model_contract(self.failure_model_path, model_type="failure")
                self.model_status = "AVAILABLE"
            except Exception as e:
                print(f"[InferenceService] Failure model load warning: {e}")
                self.failure_model = None
                self.model_status = "UNAVAILABLE"
        else:
            self.model_status = "UNAVAILABLE"

    def predict_device(self, telemetry: dict) -> Dict[str, Any]:
        """
        Runs single-device failure probability prediction using authoritative FEATURE_COLUMNS.
        If model is unavailable, returns prediction_available: False without fabricating fake probabilities.
        """
        if self.failure_model is None:
            return {
                "prediction_available": False,
                "model_status": "UNAVAILABLE",
                "failure_probability": None,
                "heuristic_risk_score": self._compute_heuristic_risk(telemetry)
            }

        try:
            df_input = prepare_feature_matrix(pd.DataFrame([telemetry]), FEATURE_COLUMNS)
            prob = float(self.failure_model.predict_proba(df_input)[0, 1])
            return {
                "prediction_available": True,
                "model_status": "AVAILABLE",
                "failure_probability": round(prob, 4),
                "failure_probability_pct": round(prob * 100, 1),
                "heuristic_risk_score": self._compute_heuristic_risk(telemetry)
            }
        except Exception as e:
            return {
                "prediction_available": False,
                "model_status": f"INFERENCE_ERROR: {e}",
                "failure_probability": None,
                "heuristic_risk_score": self._compute_heuristic_risk(telemetry)
            }

    def _compute_heuristic_risk(self, telemetry: dict) -> float:
        cpu = float(telemetry.get('CPU_Usage', 0))
        temp = float(telemetry.get('Temperature', 0))
        errs = float(telemetry.get('Interface_Errors', 0))
        return round(min(100.0, (cpu / 200.0) * 100 + (temp / 180.0) * 100 + (errs / 50.0) * 100), 1)
