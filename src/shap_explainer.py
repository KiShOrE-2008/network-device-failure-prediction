"""
shap_explainer.py
-----------------
SHAP-based feature attribution for the XGBoost failure-prediction pipeline.

Uses TreeExplainer which is fast and exact for gradient-boosted trees.
The pipeline step names expected are "preprocessor" and "model" — matching
the names used in src/train_model.py.

If shap is unavailable or throws for any reason, every public function
returns None / an empty list so callers can fall back gracefully.

Public API
----------
get_shap_causes(telemetry_df, top_n=5) -> list[dict] | None
"""

from __future__ import annotations
import os
import sys
import logging
from typing import Any

import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(WORKSPACE_ROOT, "models", "failure_model.pkl")

# Feature column order (must match training input)
FEATURE_COLUMNS = [
    "Device_Type",
    "CPU_Usage",
    "Memory_Usage",
    "Temperature",
    "Uptime",
    "Interface_Errors",
    "Packet_Loss",
    "Bandwidth_Usage",
    "Log_Errors",
]

# Human-readable labels for transformed numerical features
# (after preprocessing, Device_Type becomes Device_Type_Switch)
_LABEL_MAP = {
    "CPU_Usage":           "CPU Usage",
    "Memory_Usage":        "Memory Usage",
    "Temperature":         "Temperature",
    "Uptime":              "Uptime",
    "Interface_Errors":    "Interface Errors",
    "Packet_Loss":         "Packet Loss",
    "Bandwidth_Usage":     "Bandwidth Usage",
    "Log_Errors":          "Log Errors",
    "Device_Type_Switch":  "Device Type (Switch)",
}

# ---------------------------------------------------------------------------
# Module-level SHAP setup (loaded once at import time)
# ---------------------------------------------------------------------------
_pipeline = None
_explainer = None
_preprocessor = None
_feature_names: list[str] = []
_shap_available = False

def _try_load() -> None:
    """Attempt to load model and build TreeExplainer. Silently fails."""
    global _pipeline, _explainer, _preprocessor, _feature_names, _shap_available
    try:
        import shap

        if not os.path.exists(MODEL_PATH):
            logger.warning("SHAP explainer: model file not found at %s", MODEL_PATH)
            return

        _pipeline = joblib.load(MODEL_PATH)

        # Resolve step names — "preprocessor" and "model" are our naming convention
        preprocess_step = (
            _pipeline.named_steps.get("preprocessor")
            or _pipeline.named_steps.get("preprocess")
            or list(_pipeline.named_steps.values())[0]
        )
        model_step = (
            _pipeline.named_steps.get("model")
            or _pipeline.named_steps.get("classifier")
            or list(_pipeline.named_steps.values())[-1]
        )

        _preprocessor = preprocess_step

        # Build a representative background dataset from a dummy single row
        # TreeExplainer with tree_path_dependent doesn't need a background set
        _explainer = shap.TreeExplainer(
            model_step,
            feature_perturbation="tree_path_dependent",
        )

        # Derive output feature names after ColumnTransformer
        try:
            _feature_names = list(_preprocessor.get_feature_names_out())
        except Exception:
            # Fallback for older sklearn versions
            _feature_names = []

        _shap_available = True
        logger.info("✅ SHAP TreeExplainer loaded successfully.")

    except Exception as exc:
        logger.warning("SHAP setup failed (non-fatal): %s", exc)
        _shap_available = False


# Run at import time
_try_load()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_shap_causes(
    telemetry: dict[str, Any],
    top_n: int = 5,
) -> list[dict[str, Any]] | None:
    """
    Compute SHAP values for a single telemetry dict and return the
    top_n most influential features as a list of dicts:
        [{"feature": "CPU Usage", "contribution": 0.38, "direction": "increases_risk"}, ...]

    Returns None if SHAP is unavailable (caller should fall back to
    health_engine.main_causes()).
    """
    if not _shap_available or _explainer is None or _preprocessor is None:
        return None

    try:
        import shap  # noqa: F401 — confirms import still works at call time

        # Build input DataFrame
        device_type = str(telemetry.get("Device_Type", "Router")).strip()
        device_type = "Router" if device_type.lower() == "router" else "Switch"

        row = {
            "Device_Type":      device_type,
            "CPU_Usage":        float(telemetry.get("CPU_Usage", 0)),
            "Memory_Usage":     float(telemetry.get("Memory_Usage", 0)),
            "Temperature":      float(telemetry.get("Temperature", 0)),
            "Uptime":           float(telemetry.get("Uptime", 0)),
            "Interface_Errors": int(telemetry.get("Interface_Errors", 0)),
            "Packet_Loss":      float(telemetry.get("Packet_Loss", 0)),
            "Bandwidth_Usage":  float(telemetry.get("Bandwidth_Usage", 0)),
            "Log_Errors":       int(telemetry.get("Log_Errors", 0)),
        }
        df = pd.DataFrame([row])

        # Transform through preprocessor
        X_transformed = _preprocessor.transform(df)

        # Get feature names (may be empty on older sklearn)
        feat_names = _feature_names if _feature_names else [
            f"f{i}" for i in range(X_transformed.shape[1])
        ]

        # Compute SHAP values; shape depends on SHAP version:
        #   SHAP <0.46:  list [neg_cls, pos_cls] each (n_samples, n_features)
        #   SHAP >=0.46: ndarray (n_samples, n_features, n_classes)
        shap_vals = _explainer.shap_values(X_transformed)

        if isinstance(shap_vals, list):
            # Older API: [neg_class, pos_class]
            shap_row = np.array(shap_vals[1][0])
        elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
            # New API: (n_samples, n_features, n_classes) — take class 1
            shap_row = shap_vals[0, :, 1]
        elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 2:
            # (n_samples, n_features) — single output or already collapsed
            shap_row = shap_vals[0]
        else:
            shap_row = np.array(shap_vals).flatten()


        # Build sorted result
        contributions = []
        for i, sv in enumerate(shap_row):
            raw_name = feat_names[i] if i < len(feat_names) else f"feature_{i}"
            # Strip sklearn prefixes like "num__", "cat__"
            clean_name = raw_name.split("__")[-1] if "__" in raw_name else raw_name
            label = _LABEL_MAP.get(clean_name, clean_name.replace("_", " ").title())

            contributions.append({
                "feature": label,
                "contribution": round(float(sv), 4),
                "direction": "increases_risk" if sv > 0 else "decreases_risk",
            })

        # Sort by absolute contribution, descending
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions[:top_n]

    except Exception as exc:
        logger.warning("SHAP inference failed (non-fatal): %s", exc)
        return None
