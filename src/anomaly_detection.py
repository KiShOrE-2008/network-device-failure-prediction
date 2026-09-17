"""
src/anomaly_detection.py
------------------------
Unsupervised Anomaly Detection using Isolation Forest for NetGuard NOC.
Detects unusual network device behavior even when metrics don't trigger
a supervised failure prediction.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    "CPU_Usage",
    "Memory_Usage",
    "Temperature",
    "Interface_Errors",
    "Packet_Loss",
    "Bandwidth_Usage",
    "Log_Errors",
    "CPU_Trend",
    "Memory_Trend",
    "Temperature_Trend",
    "Error_Trend",
    "PacketLoss_Trend"
]

DEFAULT_BASELINE_STATS = {
    "CPU_Usage": {"mean": 30.0, "std": 15.0},
    "Memory_Usage": {"mean": 40.0, "std": 15.0},
    "Temperature": {"mean": 40.0, "std": 10.0},
    "Interface_Errors": {"mean": 5.0, "std": 10.0},
    "Packet_Loss": {"mean": 0.5, "std": 1.0},
    "Bandwidth_Usage": {"mean": 40.0, "std": 20.0},
    "Log_Errors": {"mean": 2.0, "std": 3.0},
    "CPU_Trend": {"mean": 0.0, "std": 5.0},
    "Memory_Trend": {"mean": 0.0, "std": 4.0},
    "Temperature_Trend": {"mean": 0.0, "std": 2.5},
    "Error_Trend": {"mean": 0.0, "std": 5.0},
    "PacketLoss_Trend": {"mean": 0.0, "std": 0.5}
}

def train_anomaly_model(df: pd.DataFrame, model_path="models/anomaly_model.pkl"):
    """
    Trains an IsolationForest model on normal baseline telemetry data.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Train primarily on healthy / non-failed samples if available
    train_df = df[df["Failed"] == 0] if "Failed" in df.columns else df
    
    # Ensure all feature columns exist, filling missing with 0
    for col in FEATURE_COLS:
        if col not in train_df.columns:
            train_df[col] = 0.0
            
    X_train = train_df[FEATURE_COLS]
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("iso_forest", IsolationForest(
            n_estimators=150,
            contamination=0.08,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train)
    joblib.dump(pipeline, model_path)
    print(f"✅ Isolation Forest anomaly model trained and saved to {model_path}")
    return pipeline

def predict_anomaly(telemetry: dict, model=None, model_path="models/anomaly_model.pkl") -> dict:
    """
    Evaluates telemetry against the Isolation Forest anomaly detector.
    Returns anomaly score (0-100%), boolean flag, and anomalous feature highlights.
    """
    if model is None:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
            except Exception as e:
                print(f"⚠️ Could not load anomaly model: {e}")
                
    # Prepare single-row DataFrame
    input_row = {}
    for col in FEATURE_COLS:
        input_row[col] = float(telemetry.get(col, 0.0))
        
    df_input = pd.DataFrame([input_row])
    
    if model is not None:
        try:
            # score_samples returns opposite of anomaly score (lower = more anomalous)
            raw_score = model.score_samples(df_input)[0]
            # Map raw score (~ -0.8 to ~ -0.3) into 0-100% anomaly score
            # Lower score = higher anomaly percentage
            anomaly_pct = float(np.clip((0.15 - raw_score) / 0.55 * 100.0, 0.0, 100.0))
        except Exception:
            anomaly_pct = _heuristic_anomaly_score(input_row)
    else:
        anomaly_pct = _heuristic_anomaly_score(input_row)
        
    anomaly_pct = round(anomaly_pct, 1)
    is_anomaly = anomaly_pct > 65.0
    
    # Identify anomalous feature deviations
    anomalous_features = []
    for col in ["CPU_Usage", "Memory_Usage", "Temperature", "Interface_Errors", "Packet_Loss", "CPU_Trend", "Temperature_Trend"]:
        val = input_row[col]
        stats = DEFAULT_BASELINE_STATS.get(col, {"mean": 0, "std": 1})
        z_score = (val - stats["mean"]) / max(stats["std"], 0.001)
        if z_score > 2.2:
            anomalous_features.append({
                "feature": col.replace("_", " "),
                "raw_value": val,
                "deviation": f"+{z_score:.1f}σ above normal baseline"
            })
            
    return {
        "anomaly_score": anomaly_pct,
        "is_anomaly": is_anomaly,
        "anomalous_features": sorted(anomalous_features, key=lambda x: x["raw_value"], reverse=True)[:3]
    }

def _heuristic_anomaly_score(telemetry: dict) -> float:
    """Fallback score calculation if IsolationForest model file is unavailable."""
    score = 0.0
    if float(telemetry.get("CPU_Usage", 0)) > 85: score += 25
    if float(telemetry.get("CPU_Trend", 0)) > 20: score += 20
    if float(telemetry.get("Temperature_Usage", 0)) > 75: score += 25
    if float(telemetry.get("Temperature_Trend", 0)) > 10: score += 20
    if float(telemetry.get("Interface_Errors", 0)) > 50: score += 20
    return min(100.0, score)
