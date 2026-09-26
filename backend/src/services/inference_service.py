"""
src/services/inference_service.py
----------------------------------
ML Inference Service for single-device predictive failure analysis.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import health_engine
import shap_explainer
from intelligence import diagnostic_engine
from monitoring import syslog_collector
from services.anomaly_service import AnomalyService
from feature_engineering import ANOMALY_FEATURE_COLUMNS


class InferenceService:
    def __init__(self, failure_model_path="models/failure_model.pkl"):
        self.failure_model_path = os.path.join(WORKSPACE_ROOT, failure_model_path) if not os.path.isabs(failure_model_path) else failure_model_path
        self.failure_model = None
        self.anomaly_service = AnomalyService()
        self.load_model()

    def load_model(self):
        if os.path.exists(self.failure_model_path):
            try:
                self.failure_model = joblib.load(self.failure_model_path)
            except Exception as e:
                print(f"[InferenceService] Warning loading failure model: {e}")

    def run_inference(self, telemetry: dict, device_id: str = "manual") -> dict:
        device_type = str(telemetry.get("Device_Type", "Router")).strip()
        device_type = "Router" if device_type.lower() == "router" else "Switch"

        cpu_usage        = float(telemetry.get("CPU_Usage", 0.0))
        memory_usage     = float(telemetry.get("Memory_Usage", 0.0))
        temperature      = float(telemetry.get("Temperature", 0.0))
        uptime           = float(telemetry.get("Uptime", 0.0))
        interface_errors = int(telemetry.get("Interface_Errors", 0))
        packet_loss      = float(telemetry.get("Packet_Loss", 0.0))
        bandwidth_usage  = float(telemetry.get("Bandwidth_Usage", 0.0))
        log_errors       = int(telemetry.get("Log_Errors", 0))

        cpu_trend        = float(telemetry.get("CPU_Trend", 0.0))
        memory_trend     = float(telemetry.get("Memory_Trend", 0.0))
        temp_trend       = float(telemetry.get("Temperature_Trend", 0.0))
        error_trend      = float(telemetry.get("Error_Trend", 0.0))

        cpu_spike        = int(telemetry.get("CPU_Spike", 1 if (cpu_usage > 85 and cpu_trend > 15) else 0))
        temp_spike       = int(telemetry.get("Temperature_Spike", 1 if (temperature > 75 and temp_trend > 5) else 0))
        error_spike      = int(telemetry.get("Error_Spike", 1 if (interface_errors > 20 and error_trend > 10) else 0))

        cpu_5step_avg    = float(telemetry.get("CPU_5step_avg", cpu_usage))
        mem_5step_avg    = float(telemetry.get("Memory_5step_avg", memory_usage))
        temp_5step_avg   = float(telemetry.get("Temperature_5step_avg", temperature))
        err_5step_avg    = float(telemetry.get("Error_5step_avg", float(interface_errors)))
        loss_5step_avg   = float(telemetry.get("PacketLoss_5step_avg", packet_loss))
        loss_trend       = float(telemetry.get("PacketLoss_Trend", 0.0))
        syslog_crit      = int(telemetry.get("Syslog_Critical_Count", 0))

        clean_telemetry = {
            "Device_Type":            device_type,
            "CPU_Usage":              cpu_usage,
            "Memory_Usage":           memory_usage,
            "Temperature":            temperature,
            "Uptime":                 uptime,
            "Interface_Errors":       interface_errors,
            "Packet_Loss":            packet_loss,
            "Bandwidth_Usage":        bandwidth_usage,
            "Log_Errors":             log_errors,
            "Syslog_Critical_Count":  syslog_crit,
            "CPU_5step_avg":          cpu_5step_avg,
            "CPU_Trend":              cpu_trend,
            "CPU_Spike":              cpu_spike,
            "Memory_5step_avg":       mem_5step_avg,
            "Memory_Trend":           memory_trend,
            "Temperature_5step_avg":  temp_5step_avg,
            "Temperature_Trend":      temp_trend,
            "Temperature_Spike":      temp_spike,
            "Error_5step_avg":        err_5step_avg,
            "Error_Trend":            error_trend,
            "Error_Spike":            error_spike,
            "PacketLoss_5step_avg":   loss_5step_avg,
            "PacketLoss_Trend":       loss_trend
        }

        feat_df = pd.DataFrame([clean_telemetry])

        prediction_available = False
        model_status = "AVAILABLE"
        proba = None
        heuristic_risk = min(100.0, float(
            (cpu_usage / 200.0) * 100 +
            (temperature / 180.0) * 100 +
            (interface_errors / 50.0) * 100
        ))

        if self.failure_model is not None:
            try:
                proba = float(self.failure_model.predict_proba(feat_df)[0, 1])
                prediction_available = True
            except Exception as e:
                print(f"[InferenceService] Prediction error: {e}")
                prediction_available = False
                model_status = "UNAVAILABLE"
        else:
            prediction_available = False
            model_status = "UNAVAILABLE"

        # Anomaly Detection Service call
        anomaly_res = self.anomaly_service.predict(clean_telemetry)
        anomaly_score = anomaly_res["anomaly_score"]
        is_anomaly = anomaly_res["is_anomaly"]

        effective_prob = proba if proba is not None else (heuristic_risk / 100.0)
        risk_level = health_engine.get_risk_level(effective_prob)
        health_score = health_engine.compute_health_score(clean_telemetry, probability=effective_prob, anomaly_score=anomaly_score)

        diag_info = diagnostic_engine.diagnostic_engine.diagnose(clean_telemetry, failure_probability=effective_prob)
        log_info  = syslog_collector.syslog_collector.ingest_and_summarize(clean_telemetry, effective_prob)
        shap_causes = shap_explainer.explain_prediction(clean_telemetry)

        report = health_engine.build_health_report(
            clean_telemetry,
            probability=effective_prob,
            shap_causes=shap_causes,
            anomaly_score=anomaly_score
        )

        return {
            "prediction_available":     prediction_available,
            "model_status":             model_status,
            "failure_probability":     round(proba, 4) if proba is not None else None,
            "failure_probability_pct": round(proba * 100, 1) if proba is not None else None,
            "heuristic_risk_score":     round(heuristic_risk, 1),
            "risk":                    risk_level,
            "risk_window":             health_engine.risk_window_from_probability(effective_prob),
            "predicted_failure":       diag_info["failure_type"],
            "diagnostic_confidence":   diag_info["diagnostic_confidence"],
            "anomaly_score":           anomaly_score,
            "is_anomaly":              is_anomaly,
            "health_score":            health_score,
            "recommended_actions":     diag_info["recommended_actions"],
            "top_contributing_causes": shap_causes,
            "health_report":           report,
            "syslog_summary":          log_info
        }
