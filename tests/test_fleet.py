import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from fleet_predictor import FleetPredictor


@pytest.fixture
def predictor():
    return FleetPredictor()


def test_all_devices_predicted(predictor):
    predictions = predictor.predict_all()
    assert len(predictions) == 500
    unique_ids = set(p["device_id"] for p in predictions)
    assert len(unique_ids) == 500


def test_risk_classification(predictor):
    predictions = predictor.predict_all()
    valid_risks = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    for p in predictions:
        assert p["risk"] in valid_risks
        prob = p["failure_probability"]
        if prob >= 0.85:
            assert p["risk"] == "CRITICAL"
        elif prob >= 0.65:
            assert p["risk"] == "HIGH"
        elif prob >= 0.30:
            assert p["risk"] == "MEDIUM"
        else:
            assert p["risk"] == "LOW"


def test_ranking_order(predictor):
    predictions = predictor.predict_all()
    probabilities = [p["failure_probability"] for p in predictions]
    # Verify sorted descending
    assert probabilities == sorted(probabilities, reverse=True)


def test_fleet_summary(predictor):
    summary = predictor.get_fleet_summary()
    assert summary["total_devices"] == 500
    assert "network_health_score" in summary
    assert "risk_summary" in summary
    assert summary["risk_summary"]["low"] + summary["risk_summary"]["medium"] + summary["risk_summary"]["high"] + summary["risk_summary"]["critical"] == 500
