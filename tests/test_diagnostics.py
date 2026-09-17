import pytest
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import health_engine
from intelligence import diagnostic_engine

def test_health_score_computation():
    healthy = {"CPU_Usage": 20.0, "Memory_Usage": 30.0, "Temperature": 35.0}
    degraded = {"CPU_Usage": 95.0, "Memory_Usage": 90.0, "Temperature": 85.0}
    
    score_healthy = health_engine.compute_health_score(healthy, probability=0.05)
    score_degraded = health_engine.compute_health_score(degraded, probability=0.90)
    
    assert score_healthy > score_degraded
    assert 0.0 <= score_healthy <= 100.0

def test_risk_level_4_tiers():
    assert health_engine.get_risk_level(0.15) == "LOW"
    assert health_engine.get_risk_level(0.45) == "MEDIUM"
    assert health_engine.get_risk_level(0.75) == "HIGH"
    assert health_engine.get_risk_level(0.90) == "CRITICAL"

def test_diagnostic_engine_narrative():
    telemetry = {"CPU_Usage": 90.0, "Temperature": 86.0}
    diag = diagnostic_engine.diagnose_failure_mode(telemetry, ml_predicted_type="THERMAL", failure_prob=0.85)
    assert diag["diagnosed_failure_type"] == "THERMAL"
    assert len(diag["recommended_actions"]) > 0
