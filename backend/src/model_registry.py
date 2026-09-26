"""
src/model_registry.py
----------------------
Model Registry & Contract Schema Validation for NetGuard NOC.
Validates model versioning, feature schemas, and temporal target alignment
prior to running inference to prevent silent schema mismatches.
"""

import os
import json
import joblib
from typing import Dict, Any, Tuple
from feature_engineering import FEATURE_COLUMNS, ANOMALY_FEATURE_COLUMNS, DIAGNOSTIC_FEATURE_COLUMNS


class ModelSchemaMismatchError(ValueError):
    """Raised when an active model artifact violates expected feature schema or contract metadata."""
    pass


class ModelRegistry:
    def __init__(self, metadata_path: str = "models/model_metadata.json"):
        self.metadata_path = metadata_path

    def load_and_validate_metadata(self) -> Dict[str, Any]:
        """Loads and audits model metadata against runtime schema contracts."""
        if not os.path.exists(self.metadata_path):
            return {
                "status": "UNAVAILABLE",
                "error": f"Metadata file missing at {self.metadata_path}"
            }

        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)

            # Contract validation rules
            expected_target = "Failure_Next_12h"
            target = metadata.get("target")
            if target != expected_target:
                raise ModelSchemaMismatchError(
                    f"MODEL_SCHEMA_MISMATCH: Invalid model target '{target}'. Expected '{expected_target}'."
                )

            features = metadata.get("feature_schema", {}).get("all_features", [])
            missing_features = [c for c in FEATURE_COLUMNS if c not in features]
            if missing_features:
                raise ModelSchemaMismatchError(
                    f"MODEL_SCHEMA_MISMATCH: Missing required feature columns: {missing_features}"
                )

            return metadata
        except Exception as e:
            if isinstance(e, ModelSchemaMismatchError):
                raise
            return {
                "status": "UNAVAILABLE",
                "error": f"Metadata read error: {e}"
            }

    def validate_model_contract(self, model_path: str, model_type: str = "failure") -> Tuple[Any, Dict[str, Any]]:
        """
        Loads model artifact from disk and validates schema contracts.
        Returns (loaded_model, metadata) or raises ModelSchemaMismatchError.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at path: {model_path}")

        metadata = self.load_and_validate_metadata()
        model = joblib.load(model_path)

        # Audit expected features based on model_type
        if model_type == "failure":
            expected_schema = FEATURE_COLUMNS
        elif model_type == "anomaly":
            expected_schema = ANOMALY_FEATURE_COLUMNS
        elif model_type == "diagnostic":
            expected_schema = DIAGNOSTIC_FEATURE_COLUMNS
        else:
            expected_schema = FEATURE_COLUMNS

        # Check model pipeline steps if available
        if hasattr(model, "feature_names_in_"):
            model_feats = list(model.feature_names_in_)
            mismatch = [c for c in expected_schema if c not in model_feats]
            if mismatch:
                raise ModelSchemaMismatchError(
                    f"MODEL_SCHEMA_MISMATCH: Model {model_type} expected features {expected_schema}, but model artifact has {model_feats}"
                )

        return model, metadata
