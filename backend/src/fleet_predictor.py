"""
src/fleet_predictor.py
-----------------------
NetGuard NOC — Fleet-Wide Predictive Maintenance & Failure Intelligence Engine.
Analyzes the entire network fleet using vectorized batch inference.
Guarantees EXACTLY one prediction per unique device ID, sorted descending by failure probability.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from feature_engineering import (
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    compute_rolling_features,
    prepare_feature_matrix
)
from intelligence.diagnostic_engine import DiagnosticEngine
from health_engine import compute_health_score, compute_fleet_health_score, get_risk_level


class FleetPredictor:
    def __init__(self,
                 failure_model_path: str = "models/failure_model.pkl",
                 diagnostic_model_path: str = "models/diagnostic_model.pkl",
                 anomaly_model_path: str = "models/anomaly_model.pkl"):
        self.workspace_root = os.path.dirname(BASE_DIR)
        
        # Absolute path resolution
        self.failure_model_path = os.path.join(self.workspace_root, failure_model_path) if not os.path.isabs(failure_model_path) else failure_model_path
        self.diagnostic_model_path = os.path.join(self.workspace_root, diagnostic_model_path) if not os.path.isabs(diagnostic_model_path) else diagnostic_model_path
        self.anomaly_model_path = os.path.join(self.workspace_root, anomaly_model_path) if not os.path.isabs(anomaly_model_path) else anomaly_model_path

        self.failure_model = None
        self.anomaly_model = None
        self.diagnostic_engine = DiagnosticEngine(self.diagnostic_model_path)

        self.load_models()

    def load_models(self):
        """Loads trained failure prediction model and anomaly detector once into memory."""
        if os.path.exists(self.failure_model_path):
            try:
                self.failure_model = joblib.load(self.failure_model_path)
            except Exception as e:
                print(f"[FleetPredictor] Warning loading failure model: {e}")

        if os.path.exists(self.anomaly_model_path):
            try:
                self.anomaly_model = joblib.load(self.anomaly_model_path)
            except Exception as e:
                print(f"[FleetPredictor] Warning loading anomaly model: {e}")

    def load_fleet_data(self, csv_path: str = None) -> pd.DataFrame:
        """Loads and prepares telemetry dataset for all devices."""
        if csv_path is None:
            csv_path = os.path.join(self.workspace_root, "data", "network_devices_timeseries.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(self.workspace_root, "data", "netguard_noc_dataset_v1", "network_devices_timeseries.csv")

        if not os.path.exists(csv_path):
            csv_path = os.path.join(self.workspace_root, "data", "network_devices.csv")

        df = pd.read_csv(csv_path)
        df = compute_rolling_features(df)
        return df

    def predict_all(self, df: pd.DataFrame | None = None) -> List[Dict[str, Any]]:
        """
        Runs batch fleet-wide inference across every device in the network.
        Guarantees EXACTLY ONE prediction per unique Device_ID (using its latest observation).
        Returns predictions sorted by failure probability descending.
        """
        if df is None:
            df = self.load_fleet_data()

        if 'Timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Group by Device_ID and extract the latest observation per device
        latest_df = df.sort_values(['Device_ID', 'Timestamp']).groupby('Device_ID').last().reset_index()

        # Batch feature matrix preparation
        X_batch = prepare_feature_matrix(latest_df, FEATURE_COLUMNS)

        # Batch Model Inference
        is_fallback = False
        if self.failure_model is not None:
            try:
                probs = self.failure_model.predict_proba(X_batch)[:, 1]
            except Exception as e:
                print(f"[FleetPredictor] Error during batch prediction: {e}")
                is_fallback = True
                probs = np.clip(
                    (latest_df['CPU_Usage'].fillna(0) / 200.0) +
                    (latest_df['Temperature'].fillna(0) / 180.0) +
                    (latest_df['Interface_Errors'].fillna(0) / 50.0),
                    0.0, 1.0
                ).values
        else:
            is_fallback = True
            probs = np.clip(
                (latest_df['CPU_Usage'].fillna(0) / 200.0) +
                (latest_df['Temperature'].fillna(0) / 180.0) +
                (latest_df['Interface_Errors'].fillna(0) / 50.0),
                0.0, 1.0
            ).values

        # Batch Anomaly Detection
        anomaly_scores = []
        if self.anomaly_model is not None:
            try:
                anomaly_features = ['CPU_Usage', 'Memory_Usage', 'Temperature', 'Interface_Errors', 'Packet_Loss', 'Bandwidth_Usage']
                num_matrix = latest_df[[c for c in anomaly_features if c in latest_df.columns]].fillna(0.0)
                raw_anomaly = self.anomaly_model.score_samples(num_matrix)
                anomaly_scores = [round(float(min(max((-s - 0.3) * 200, 0), 100)), 1) for s in raw_anomaly]
            except Exception:
                anomaly_scores = [0.0] * len(latest_df)
        else:
            anomaly_scores = [0.0] * len(latest_df)

        predictions = []
        for idx, row in latest_df.iterrows():
            prob = float(probs[idx])
            anomaly_score = float(anomaly_scores[idx]) if idx < len(anomaly_scores) else 0.0

            telemetry_dict = row.to_dict()
            risk_level = get_risk_level(prob)
            health_score = compute_health_score(telemetry_dict, probability=prob, anomaly_score=anomaly_score)

            diag_info = self.diagnostic_engine.diagnose(telemetry_dict, failure_probability=prob)

            ts_val = row.get('Timestamp', '')
            ts_str = ts_val.isoformat() if hasattr(ts_val, 'isoformat') else str(ts_val)

            pred = {
                "device_id": str(row['Device_ID']),
                "hostname": str(row.get('Hostname', f"DEV-{row['Device_ID']}")),
                "ip_address": str(row.get('IP_Address', '10.10.1.1')),
                "device_type": str(row.get('Device_Type', 'Router')),
                "vendor": str(row.get('Vendor', 'Cisco')),
                "model": str(row.get('Model', 'ISR-4331')),
                "location": str(row.get('Location', 'Chennai DC-1')),
                "rack": str(row.get('Rack', 'Rack A01')),
                "firmware": str(row.get('Firmware', '17.6.4')),
                "timestamp": ts_str,
                "failure_probability": round(prob, 4),
                "failure_probability_pct": round(prob * 100, 1),
                "risk": risk_level,
                "predicted_failure": diag_info["failure_type"],
                "diagnostic_confidence": diag_info["diagnostic_confidence"],
                "anomaly_score": anomaly_score,
                "health_score": health_score,
                "current_failed": int(row.get('Failed', 0)),
                "recommended_actions": diag_info["recommended_actions"],
                "is_fallback": is_fallback,
                "telemetry": {
                    "cpu": round(float(row.get('CPU_Usage', 0)), 1),
                    "memory": round(float(row.get('Memory_Usage', 0)), 1),
                    "temperature": round(float(row.get('Temperature', 0)), 1),
                    "errors": round(float(row.get('Interface_Errors', 0)), 1),
                    "packet_loss": round(float(row.get('Packet_Loss', 0)), 2),
                    "bandwidth": round(float(row.get('Bandwidth_Usage', 0)), 1),
                    "log_errors": int(row.get('Log_Errors', 0))
                }
            }
            predictions.append(pred)

        # Rank fleet descending by failure probability
        predictions.sort(key=lambda x: x["failure_probability"], reverse=True)
        return predictions

    def get_fleet_summary(self, predictions: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """Calculates fleet-wide health scores and risk summary metrics."""
        if predictions is None:
            predictions = self.predict_all()

        total = len(predictions)
        low = sum(1 for p in predictions if p["risk"] == "LOW")
        medium = sum(1 for p in predictions if p["risk"] == "MEDIUM")
        high = sum(1 for p in predictions if p["risk"] == "HIGH")
        critical = sum(1 for p in predictions if p["risk"] == "CRITICAL")

        health_scores = [p["health_score"] for p in predictions]
        probabilities = [p["failure_probability"] for p in predictions]

        network_health = compute_fleet_health_score(health_scores, probabilities)

        mode_counts = {}
        for p in predictions:
            if p["risk"] in ["HIGH", "CRITICAL"]:
                m = p["predicted_failure"]
                mode_counts[m] = mode_counts.get(m, 0) + 1

        return {
            "total_devices": total,
            "risk_summary": {
                "low": low,
                "medium": medium,
                "high": high,
                "critical": critical
            },
            "network_health_score": network_health,
            "average_failure_probability": round(float(np.mean(probabilities)) * 100, 1) if probabilities else 0.0,
            "predicted_failures_next_12h": critical + high,
            "failure_mode_breakdown": mode_counts,
            "timestamp": pd.Timestamp.now().isoformat()
        }


def main():
    print("=" * 60)
    print("NETGUARD NOC — FLEET INTELLIGENCE")
    print("=" * 60)

    predictor = FleetPredictor()

    predictions = predictor.predict_all()
    summary = predictor.get_fleet_summary(predictions)

    print(f"\nDevices discovered:          {summary['total_devices']}")
    print(f"Devices analyzed:            {summary['total_devices']}")
    print(f"Predictions generated:       {len(predictions)}")

    print(f"\nNetwork Health:              {summary['network_health_score']} / 100")

    print(f"\nLOW:                         {summary['risk_summary']['low']}")
    print(f"MEDIUM:                      {summary['risk_summary']['medium']}")
    print(f"HIGH:                        {summary['risk_summary']['high']}")
    print(f"CRITICAL:                    {summary['risk_summary']['critical']}")

    print("\nTOP PREDICTED RISKS")
    print("-" * 65)
    for idx, p in enumerate(predictions[:10], 1):
        print(f"{idx:<2}. {p['device_id']:<10} {p['failure_probability_pct']:>5.1f}%   {p['risk']:<9}  {p['predicted_failure']:<12} Health: {p['health_score']}")

    print("\nFAILURE MODE DISTRIBUTION")
    print("-" * 40)
    for mode, count in summary['failure_mode_breakdown'].items():
        print(f"{mode:<15} {count}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
