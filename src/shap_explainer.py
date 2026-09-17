"""
src/shap_explainer.py
---------------------
SHAP-based feature attribution for NetGuard NOC machine learning pipelines.
Supports TreeExplainer (XGBoost / RandomForest) and LinearExplainer / Explainer fallback.
"""

from __future__ import annotations
import os
import sys
import logging
from typing import Any, List, Dict

import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(WORKSPACE_ROOT, "models", "failure_model.pkl")

_LABEL_MAP = {
    "CPU_Usage":           "CPU Utilization",
    "Memory_Usage":        "Memory Utilization",
    "Temperature":         "Chassis Temperature",
    "Uptime":              "System Uptime",
    "Interface_Errors":    "Interface CRC Errors",
    "Packet_Loss":         "Packet Loss Rate",
    "Bandwidth_Usage":     "Bandwidth Utilization",
    "Log_Errors":          "Log Errors Count",
    "CPU_5step_avg":       "CPU 5-Step Avg",
    "Memory_5step_avg":    "Memory 5-Step Avg",
    "Temperature_5step_avg":"Temp 5-Step Avg",
    "Error_5step_avg":     "Error 5-Step Avg",
    "PacketLoss_5step_avg":"Packet Loss 5-Step Avg",
    "CPU_Trend":           "CPU 5-Step Trend",
    "Memory_Trend":        "Memory 5-Step Trend",
    "Temperature_Trend":   "Temperature 5-Step Trend",
    "Error_Trend":         "Interface Error Velocity",
    "PacketLoss_Trend":    "Packet Loss Velocity",
    "CPU_Spike":           "CPU Spike Indicator",
    "Temperature_Spike":   "Temperature Spike Indicator",
    "Error_Spike":         "Error Spike Indicator",
    "Device_Type_Switch":  "Device Architecture (Switch)"
}

_pipeline = None
_explainer = None
_preprocessor = None
_feature_names: List[str] = []
_shap_available = False

def _try_load() -> None:
    global _pipeline, _explainer, _preprocessor, _feature_names, _shap_available
    try:
        import shap

        if not os.path.exists(MODEL_PATH):
            logger.warning("SHAP explainer: model file not found at %s", MODEL_PATH)
            return

        _pipeline = joblib.load(MODEL_PATH)

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
        
        # Select appropriate SHAP explainer
        model_name = model_step.__class__.__name__
        if "XGB" in model_name or "RandomForest" in model_name or "Tree" in model_name:
            _explainer = shap.TreeExplainer(model_step, feature_perturbation="tree_path_dependent")
        elif "Linear" in model_name or "Logistic" in model_name:
            _explainer = shap.Explainer(model_step, maskers=None) if hasattr(shap, 'Explainer') else None
        else:
            _explainer = shap.Explainer(model_step)

        try:
            _feature_names = list(_preprocessor.get_feature_names_out())
        except Exception:
            _feature_names = []

        _shap_available = _explainer is not None
        if _shap_available:
            logger.info("✅ SHAP explainer initialized for %s.", model_name)

    except Exception as exc:
        logger.warning("SHAP setup failed (non-fatal): %s", exc)
        _shap_available = False

# Run at import time
_try_load()

def get_shap_causes(telemetry: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]] | None:
    if not _shap_available or _explainer is None or _preprocessor is None:
        return None

    try:
        device_type = str(telemetry.get("Device_Type", "Router")).strip()
        device_type = "Router" if device_type.lower() == "router" else "Switch"

        row = {
            "Device_Type":            device_type,
            "CPU_Usage":              float(telemetry.get("CPU_Usage", 0)),
            "Memory_Usage":           float(telemetry.get("Memory_Usage", 0)),
            "Temperature":            float(telemetry.get("Temperature", 0)),
            "Uptime":                 float(telemetry.get("Uptime", 0)),
            "Interface_Errors":       int(telemetry.get("Interface_Errors", 0)),
            "Packet_Loss":            float(telemetry.get("Packet_Loss", 0)),
            "Bandwidth_Usage":        float(telemetry.get("Bandwidth_Usage", 0)),
            "Log_Errors":             int(telemetry.get("Log_Errors", 0)),
            "CPU_5step_avg":          float(telemetry.get("CPU_5step_avg", telemetry.get("CPU_Usage", 0))),
            "Memory_5step_avg":       float(telemetry.get("Memory_5step_avg", telemetry.get("Memory_Usage", 0))),
            "Temperature_5step_avg":  float(telemetry.get("Temperature_5step_avg", telemetry.get("Temperature", 0))),
            "Error_5step_avg":        float(telemetry.get("Error_5step_avg", telemetry.get("Interface_Errors", 0))),
            "PacketLoss_5step_avg":   float(telemetry.get("PacketLoss_5step_avg", telemetry.get("Packet_Loss", 0))),
            "CPU_Trend":              float(telemetry.get("CPU_Trend", 0)),
            "Memory_Trend":           float(telemetry.get("Memory_Trend", 0)),
            "Temperature_Trend":      float(telemetry.get("Temperature_Trend", 0)),
            "Error_Trend":            float(telemetry.get("Error_Trend", 0)),
            "PacketLoss_Trend":       float(telemetry.get("PacketLoss_Trend", 0)),
            "CPU_Spike":              int(telemetry.get("CPU_Spike", 0)),
            "Temperature_Spike":      int(telemetry.get("Temperature_Spike", 0)),
            "Error_Spike":            int(telemetry.get("Error_Spike", 0))
        }
        
        df = pd.DataFrame([row])
        X_transformed = _preprocessor.transform(df)

        feat_names = _feature_names if _feature_names else [f"f{i}" for i in range(X_transformed.shape[1])]

        shap_vals = _explainer(X_transformed) if hasattr(_explainer, '__call__') else _explainer.shap_values(X_transformed)

        if hasattr(shap_vals, 'values'):
            shap_row = np.array(shap_vals.values[0])
            if shap_row.ndim > 1:
                shap_row = shap_row[:, 1] if shap_row.shape[1] > 1 else shap_row[:, 0]
        elif isinstance(shap_vals, list):
            shap_row = np.array(shap_vals[1][0])
        elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
            shap_row = shap_vals[0, :, 1]
        elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 2:
            shap_row = shap_vals[0]
        else:
            shap_row = np.array(shap_vals).flatten()

        contributions = []
        for i, sv in enumerate(shap_row):
            raw_name = feat_names[i] if i < len(feat_names) else f"feature_{i}"
            clean_name = raw_name.split("__")[-1] if "__" in raw_name else raw_name
            label = _LABEL_MAP.get(clean_name, clean_name.replace("_", " ").title())

            contributions.append({
                "feature": label,
                "contribution": round(float(sv), 4),
                "direction": "increases_risk" if sv > 0 else "decreases_risk",
                "impact": "HIGH" if abs(sv) > 0.15 else "MEDIUM" if abs(sv) > 0.05 else "LOW"
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions[:top_n]

    except Exception as exc:
        logger.warning("SHAP inference failed (non-fatal): %s", exc)
        return None
