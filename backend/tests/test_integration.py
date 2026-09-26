"""
tests/test_integration.py
--------------------------
End-to-End System Integration Test Suite for NetGuard NOC.
Validates the complete pipeline workflow:
Dataset -> Feature Engineering -> Target Engineering -> Model Training -> Save -> Reload ->
Fleet Inference -> Anomaly Detection -> Diagnostics -> Incident Engine -> API Flask App.
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(BACKEND_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import compute_rolling_features, prepare_feature_matrix
from target_engineering import compute_failure_next_12h_target
from train_model import train_all_models
from fleet_predictor import FleetPredictor
from anomaly_detection import predict_anomaly
from intelligence.diagnostic_engine import DiagnosticEngine
from history_store import HistoryStore
from services.incident_service import IncidentService
from web_app import app


def test_full_pipeline_integration(tmp_path):
    """
    Executes one complete end-to-end integration flow across all NetGuard NOC modules.
    """
    # 1. Dataset Generation & Preparation
    timestamps = pd.date_range('2026-01-01', periods=30, freq='1h')
    data = []
    for dev_id in ['DEV-0001', 'DEV-0002']:
        for ts in timestamps:
            data.append({
                'Device_ID': dev_id,
                'Hostname': f'host-{dev_id}',
                'IP_Address': '10.1.1.1',
                'Device_Type': 'Router',
                'Vendor': 'Cisco',
                'Model': 'ISR-4331',
                'Location': 'DC-1',
                'Rack': 'R01',
                'Firmware': '17.6.4',
                'Timestamp': ts,
                'CPU_Usage': np.random.uniform(20, 95),
                'Memory_Usage': np.random.uniform(30, 80),
                'Temperature': np.random.uniform(35, 85),
                'Interface_Errors': np.random.randint(0, 50),
                'Packet_Loss': np.random.uniform(0, 5),
                'Bandwidth_Usage': np.random.uniform(10, 90),
                'Uptime': 1000,
                'Log_Errors': 2,
                'Syslog_Critical_Count': 0,
                'Failed': 1 if ts == timestamps[15] else 0,
                'Failure_Type': 'THERMAL' if ts == timestamps[15] else 'NONE'
            })
    df_raw = pd.DataFrame(data)

    # 2. Feature Engineering & Target Generation
    df_feat = compute_rolling_features(df_raw)
    df_target = compute_failure_next_12h_target(df_feat, horizon=12)

    assert 'CPU_5step_avg' in df_target.columns
    assert 'Failure_Next_12h' in df_target.columns

    # 3. Model Training & Save
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    os.makedirs("models", exist_ok=True)
    
    # Train models
    train_all_models()

    assert os.path.exists("models/failure_model.pkl")
    assert os.path.exists("models/diagnostic_model.pkl")
    assert os.path.exists("models/anomaly_model.pkl")
    assert os.path.exists("models/model_metadata.json")

    # 4. Model Reload & Fleet Prediction
    predictor = FleetPredictor()
    predictions = predictor.predict_all()
    assert len(predictions) > 0

    top_pred = predictions[0]
    assert "failure_probability" in top_pred
    assert "anomaly_score" in top_pred
    assert "health_score" in top_pred

    # 5. Single Anomaly & Diagnostic Evaluation
    sample_telemetry = top_pred["telemetry"]
    anomaly_res = predict_anomaly(sample_telemetry)
    assert "anomaly_score" in anomaly_res

    diag_engine = DiagnosticEngine()
    diag_res = diag_engine.diagnose(sample_telemetry, failure_probability=top_pred.get("failure_probability") or 0.5)
    assert "failure_type" in diag_res

    # 6. Incident Engine Processing
    db_path = str(tmp_path / "test_history.db")
    history_store = HistoryStore(db_path=db_path)
    incident_service = IncidentService(db_store=history_store)

    incident_summary = incident_service.process_fleet_alerts(predictions)
    assert "total_active_alerts" in incident_summary

    # 7. Flask REST API Integration Endpoint Verification
    client = app.test_client()
    res = client.get('/api/v1/health')
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["status"] == "HEALTHY"

    res_fleet = client.get('/api/fleet/predictions')
    assert res_fleet.status_code == 200
    fleet_json = res_fleet.get_json()
    assert fleet_json["success"] is True
