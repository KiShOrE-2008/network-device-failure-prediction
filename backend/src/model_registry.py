"""
src/model_registry.py
----------------------
Model Registry & Schema Contract Validation Engine for NetGuard NOC.

Provides formal model contracts and schema verification for:
1. Failure Model (Input schema, Target, Version, Validation strategy)
2. Anomaly Model (Input schema, Baseline, Version, Detector type)
3. Diagnostic Model (Input schema, Classes, Version, Training filter)
"""

import os
import json
import joblib
from typing import Dict, Any, Tuple, Optional
from feature_engineering import FEATURE_COLUMNS, ANOMALY_FEATURE_COLUMNS, DIAGNOSTIC_FEATURE_COLUMNS


class ModelSchemaMismatchError(Exception):
    """Raised when model metadata schema or target does not match expectations."""
    pass


class FailureModelContract:
    def __init__(self, input_schema=None, target="Failure_Next_12h", version="3.0.0", validation_strategy="temporal"):
        self.input_schema = input_schema or FEATURE_COLUMNS.copy()
        self.target = target
        self.version = version
        self.validation_strategy = validation_strategy


class AnomalyModelContract:
    def __init__(self, input_schema=None, baseline="Failed == 0", version="3.0.0", detector_type="IsolationForest"):
        self.input_schema = input_schema or ANOMALY_FEATURE_COLUMNS.copy()
        self.baseline = baseline
        self.version = version
        self.detector_type = detector_type


class DiagnosticModelContract:
    def __init__(self, input_schema=None, classes=None, version="3.0.0", training_filter="Failed == 1"):
        self.input_schema = input_schema or DIAGNOSTIC_FEATURE_COLUMNS.copy()
        self.classes = classes or ["THERMAL", "HARDWARE", "MEMORY", "INTERFACE", "CONGESTION"]
        self.version = version
        self.training_filter = training_filter


class ModelRegistry:
    """
    Manages loading, schema validation, and metadata registry for failure, anomaly, and diagnostic models.
    """

    def __init__(self, metadata_path: str = "models/model_metadata.json"):
        self.metadata_path = metadata_path
        self.metadata: Optional[Dict[str, Any]] = None

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Loads model registry metadata JSON if present."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                return self.metadata
            except Exception as e:
                print(f"[ModelRegistry] Error loading metadata from {self.metadata_path}: {e}")
                self.metadata = None
        return None

    def load_and_validate_metadata(self, expected_features: list = None, expected_target: str = "Failure_Next_12h") -> Dict[str, Any]:
        """
        Loads metadata and validates target and feature schema compliance.
        Raises ModelSchemaMismatchError if target or feature schema fails validation.
        """
        meta = self.load_metadata()
        if meta is None:
            raise ModelSchemaMismatchError(f"Model metadata not found at {self.metadata_path}")

        target = meta.get("target")
        if expected_target and target != expected_target:
            raise ModelSchemaMismatchError(f"Invalid model target: expected {expected_target}, got {target}")

        if expected_features is None:
            expected_features = FEATURE_COLUMNS.copy()

        all_features = meta.get("feature_schema", {}).get("all_features", [])
        missing = set(expected_features) - set(all_features)
        if missing:
            raise ModelSchemaMismatchError(f"Missing required feature columns: {missing}")

        return meta

    def validate_schema(self, expected_features: list, target_name: str = "Failure_Next_12h", expected_version: str = "3.0.0") -> Tuple[bool, str]:
        """
        Validates model schema against expected feature set, target variable, and version contract.
        Returns (is_valid: bool, reason: str).
        """
        try:
            self.load_and_validate_metadata(expected_features=expected_features, expected_target=target_name)
            return True, "MATCH"
        except ModelSchemaMismatchError as e:
            return False, str(e)

    def load_and_verify_model(self, model_path: str, expected_features: list = None, target_name: str = "Failure_Next_12h") -> Tuple[Any, Dict[str, Any]]:
        """
        Loads model file after validating registry metadata schema.
        Raises ModelSchemaMismatchError if schema verification fails.
        """
        meta = self.load_and_validate_metadata(expected_features=expected_features, expected_target=target_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        return model, meta
