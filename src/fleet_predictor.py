"""
src/fleet_predictor.py
-----------------------
NetGuard NOC — Fleet-Wide Predictive Maintenance & Failure Intelligence Engine.
Analyzes the entire network fleet, continuously estimates failure risk for every device,
identifies devices most likely to fail in the next 12 hours, determines probable failure modes,
detects telemetry anomalies, and prioritizes incidents for NOC operators.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from feature_engineering import FEATURE_COLUMNS, compute_rolling_features
from intelligence.diagnostic_engine import DiagnosticEngine
from health_engine import compute_health_score, compute_fleet_health_score, get_risk_level


class FleetPredictor:
    def __init__(self,
                 failure_model_path: str = "models/failure_model.pkl",
                 diagnostic_model_path: str = "models/diagnostic_model.pkl",
                 anomaly_model_path: str = "models/anomaly_model.pkl"):
        self.failure_model_path = failure_model_path
        self.diagnostic_model_path = diagnostic_model_path
        self.anomaly_model_path = anomaly_model_path

        self.failure_model = None
        self.anomaly_model = None
        self.diagnostic_engine = DiagnosticEngine(diagnostic_model_path)

        self.load_models()

    def load_models(self):
        """Loads trained failure prediction model and anomaly detector."""
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

    def load_fleet_data(self, csv_path: str = "data/netguard_noc_dataset_v1/network_devices_timeseries.csv") -> pd.DataFrame:
        """Loads and prepares telemetry dataset for all devices."""
        if not os.path.exists(csv_path):
            csv_path = "data/network_devices_timeseries.csv"

        df = pd.read_csv(csv_path)
        df = compute_rolling_features(df)
        return df

    def predict_all(self, df: pd.DataFrame | None = None) -> List[Dict[str, Any]]:
        """
        Runs fleet-wide inference across every device in the network.
        Guarantees EXACTLY one prediction per unique Device_ID (using its latest historical observation).
        """
        if df is None:
            df = self.load_fleet_data()

        # Sort by Device_ID and Timestamp, then extract latest telemetry row per device
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        latest_df = df.sort_values(['Device_ID', 'Timestamp']).groupby('Device_ID').last().reset_index()

        feat_cols = [
            "Device_Type", "CPU_Usage", "Memory_Usage", "Temperature", "Uptime",
            "Interface_Errors", "Packet_Loss", "Bandwidth_Usage", "Log_Errors",
            "Syslog_Critical_Count", "CPU_5step_avg", "Memory_5step_avg", "Temperature_5step_avg",
            "Error_5step_avg", "PacketLoss_5step_avg", "CPU_Trend", "Memory_Trend",
            "Temperature_Trend", "Error_Trend", "PacketLoss_Trend",
            "CPU_Spike", "Temperature_Spike", "Error_Spike"
        ]


        # Prepare batch feature matrix
        X_batch = latest_df[[c for c in feat_cols if c in latest_df.columns]].copy()
        for col in feat_cols:
            if col not in X_batch.columns:
                X_batch[col] = 0.0

        # Run vectorized binary model prediction
        if self.failure_model is not None:
            probs = self.failure_model.predict_proba(X_batch)[:, 1]
        else:
            # Fallback heuristic if model not loaded
            probs = np.clip(
                (latest_df['CPU_Usage'] / 200.0) + (latest_df['Temperature'] / 180.0) + (latest_df['Interface_Errors'] / 50.0),
                0.0, 1.0
            ).values

        # Run anomaly detection
        anomaly_scores = []
        if self.anomaly_model is not None:
            try:
                anomaly_features = ['CPU_Usage', 'Memory_Usage', 'Temperature', 'Interface_Errors', 'Packet_Loss', 'Bandwidth_Usage']
                num_matrix = latest_df[[c for c in anomaly_features if c in latest_df.columns]].fillna(0.0)
                raw_anomaly = self.anomaly_model.score_samples(num_matrix)
                # Map Isolation Forest score (higher negative = more anomalous) to [0, 100] index
                anomaly_scores = [round(float(min(max(( -s - 0.3) * 200, 0), 100)), 1) for s in raw_anomaly]
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

            # Diagnosis engine
            diag_info = self.diagnostic_engine.diagnose(telemetry_dict, failure_probability=prob)

            pred = {
                "device_id": str(row['Device_ID']),
                "hostname": str(row.get('Hostname', f"dev-{row['Device_ID']}")),
                "ip_address": str(row.get('IP_Address', '10.0.0.1')),
                "device_type": str(row.get('Device_Type', 'ROUTER')),
                "vendor": str(row.get('Vendor', 'Generic')),
                "model": str(row.get('Model', 'Unknown')),
                "location": str(row.get('Location', 'HQ')),
                "timestamp": str(row.get('Timestamp', '')),
                "failure_probability": round(prob, 4),
                "failure_probability_pct": round(prob * 100, 1),
                "risk": risk_level,
                "predicted_failure": diag_info["failure_type"],
                "diagnostic_confidence": diag_info["diagnostic_confidence"],
                "recommended_actions": diag_info["recommended_actions"],
                "anomaly_score": anomaly_score,
                "health_score": health_score,
                "current_failed": int(row.get('Failed', 0)),
                "telemetry": {
                    "cpu": round(float(row.get('CPU_Usage', 0)), 1),
                    "memory": round(float(row.get('Memory_Usage', 0)), 1),
                    "temperature": round(float(row.get('Temperature', 0)), 1),
                    "errors": round(float(row.get('Interface_Errors', 0)), 1),
                    "packet_loss": round(float(row.get('Packet_Loss', 0)), 2),
                    "bandwidth": round(float(row.get('Bandwidth_Usage', 0)), 1)
                }
            }
            predictions.append(pred)

        # Rank fleet descending by failure probability
        predictions.sort(key=lambda x: x["failure_probability"], reverse=True)
        return predictions

    def get_fleet_summary(self, predictions: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """Calculates fleet-wide health scores and statistics."""
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
    print("NETGUARD NOC — FLEET PREDICTION ENGINE")
    print("=" * 60)

    predictor = FleetPredictor()

    print("\nLoading models...")
    print(f"✓ Failure model loaded: {predictor.failure_model is not None}")
    print(f"✓ Diagnostic engine loaded: {predictor.diagnostic_engine.model is not None}")
    print(f"✓ Anomaly model loaded: {predictor.anomaly_model is not None}")

    print("\nRunning fleet prediction across all devices...")
    predictions = predictor.predict_all()
    summary = predictor.get_fleet_summary(predictions)

    print("\n" + "=" * 60)
    print("FLEET FORECAST SUMMARY (Next 12 Hours)")
    print("=" * 60)
    print(f"Total devices analyzed:  {summary['total_devices']}")
    print(f"Network Health Score:    {summary['network_health_score']} / 100")
    print(f"LOW Risk:                {summary['risk_summary']['low']}")
    print(f"MEDIUM Risk:             {summary['risk_summary']['medium']}")
    print(f"HIGH Risk:               {summary['risk_summary']['high']}")
    print(f"CRITICAL Risk:           {summary['risk_summary']['critical']}")

    print("\n" + "=" * 60)
    print("TOP PREDICTED HIGH-RISK DEVICES")
    print("=" * 60)
    top_devices = predictions[:10]
    print(f"{'#':<4} {'DEVICE ID':<12} {'HOSTNAME':<22} {'RISK':<10} {'PROB %':<8} {'DIAGNOSIS':<12} {'HEALTH':<8}")
    print("-" * 78)
    for idx, p in enumerate(top_devices, 1):
        print(f"{idx:<4} {p['device_id']:<12} {p['hostname']:<22} {p['risk']:<10} {p['failure_probability_pct']:<8.1f} {p['predicted_failure']:<12} {p['health_score']:<8.1f}")

    print("\nPrediction complete.\n")


if __name__ == "__main__":
    main()
