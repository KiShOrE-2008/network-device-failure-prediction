"""
tests/test_anomaly_detection.py
--------------------------------
Comprehensive Anomaly Detection Pipeline & Schema Validation Test Suite.
Verifies training, saving, reloading, single & batch predictions, and schema contract.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
import pytest
import joblib

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(BACKEND_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import ANOMALY_FEATURE_COLUMNS
from anomaly_detection import train_anomaly_model, predict_anomaly, predict_anomaly_batch


def test_anomaly_pipeline_train_save_reload_single_and_batch():
    """Verify train -> save -> reload -> single prediction -> batch prediction workflow."""
    # Create synthetic normal operational telemetry
    records = []
    for i in range(100):
        records.append({
            "Device_ID": f"DEV-{(i % 5) + 1:03d}",
            "Timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
            "CPU_Usage": np.random.uniform(10, 40),
            "Memory_Usage": np.random.uniform(20, 50),
            "Temperature": np.random.uniform(30, 50),
            "Interface_Errors": np.random.uniform(0, 5),
            "Packet_Loss": 0.0,
            "Bandwidth_Usage": np.random.uniform(10, 60),
            "Log_Errors": 0,
            "CPU_Trend": 0.0,
            "Memory_Trend": 0.0,
            "Temperature_Trend": 0.0,
            "Error_Trend": 0.0,
            "PacketLoss_Trend": 0.0,
            "CPU_Spike": 0,
            "Temperature_Spike": 0,
            "Error_Spike": 0,
            "Failed": 0
        })

    df = pd.DataFrame(records)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_anomaly_model.pkl")

        # 1. Train pipeline
        pipeline = train_anomaly_model(df, model_path=model_path)
        assert os.path.exists(model_path)

        # 2. Reload pipeline
        reloaded_pipeline = joblib.load(model_path)

        # 3. Single prediction
        sample_telemetry = {
            "CPU_Usage": 95.0,
            "Memory_Usage": 90.0,
            "Temperature": 85.0,
            "Interface_Errors": 100.0,
            "Packet_Loss": 5.0,
            "CPU_Trend": 30.0,
            "CPU_Spike": 1
        }
        res_single = predict_anomaly(sample_telemetry, model=reloaded_pipeline)

        assert "anomaly_score" in res_single
        assert "is_anomaly" in res_single
        assert res_single["is_anomaly"] is True
        assert res_single["anomaly_score"] > 50.0

        # 4. Batch prediction
        normal_telemetry = {
            "CPU_Usage": 25.0, "Memory_Usage": 30.0, "Temperature": 35.0,
            "Interface_Errors": 0.0, "Packet_Loss": 0.0, "Bandwidth_Usage": 20.0,
            "Log_Errors": 0, "CPU_Trend": 0.0, "Memory_Trend": 0.0,
            "Temperature_Trend": 0.0, "Error_Trend": 0.0, "PacketLoss_Trend": 0.0,
            "CPU_Spike": 0, "Temperature_Spike": 0, "Error_Spike": 0
        }
        batch_df = pd.DataFrame([sample_telemetry, normal_telemetry])
        batch_scores = predict_anomaly_batch(batch_df, model=reloaded_pipeline)

        assert len(batch_scores) == 2
        assert batch_scores[0] > batch_scores[1]



def test_anomaly_score_is_distinct_from_failure_probability():
    """Verify anomaly_score concept is distinct from ML failure probability."""
    telemetry = {
        "CPU_Usage": 92.0,
        "Memory_Usage": 80.0,
        "Temperature": 82.0,
        "CPU_Trend": 25.0,
        "CPU_Spike": 1
    }

    res = predict_anomaly(telemetry)
    anomaly_score = res["anomaly_score"]

    # Anomaly score represents 0-100 deviation score, NOT a probability string or binary classification
    assert isinstance(anomaly_score, float)
    assert 0.0 <= anomaly_score <= 100.0
    assert "failure_probability" not in res, "Anomaly detector must NOT return failure_probability!"
