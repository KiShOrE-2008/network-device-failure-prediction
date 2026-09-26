import os
import sys
import pytest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fleet_predictor import FleetPredictor
from health_engine import get_risk_level


def test_fleet_predictor_returns_exactly_one_prediction_per_device():
    """Verify N unique devices in fleet results in exactly N predictions."""
    predictor = FleetPredictor()
    predictions = predictor.predict_all()
    
    unique_devices_in_dataset = 500
    assert len(predictions) == unique_devices_in_dataset, f"Expected {unique_devices_in_dataset} predictions, got {len(predictions)}"


def test_all_device_ids_are_unique():
    """Verify no duplicate Device_ID exists in returned fleet predictions."""
    predictor = FleetPredictor()
    predictions = predictor.predict_all()
    
    device_ids = [p["device_id"] for p in predictions]
    assert len(device_ids) == len(set(device_ids)), "Duplicate device_ids found in predictions!"


def test_predictions_are_sorted():
    """Verify predictions are sorted descending by failure probability."""
    predictor = FleetPredictor()
    predictions = predictor.predict_all()
    
    probs = [p["failure_probability"] for p in predictions]
    assert probs == sorted(probs, reverse=True), "Predictions are not sorted descending by probability!"


def test_risk_boundaries():
    """
    Verify 4-tier risk boundaries:
    0.00 -> LOW
    0.2999 -> LOW
    0.30 -> MEDIUM
    0.6499 -> MEDIUM
    0.65 -> HIGH
    0.8499 -> HIGH
    0.85 -> CRITICAL
    1.00 -> CRITICAL
    """
    assert get_risk_level(0.00) == "LOW"
    assert get_risk_level(0.2999) == "LOW"
    assert get_risk_level(0.30) == "MEDIUM"
    assert get_risk_level(0.6499) == "MEDIUM"
    assert get_risk_level(0.65) == "HIGH"
    assert get_risk_level(0.8499) == "HIGH"
    assert get_risk_level(0.85) == "CRITICAL"
    assert get_risk_level(1.00) == "CRITICAL"
