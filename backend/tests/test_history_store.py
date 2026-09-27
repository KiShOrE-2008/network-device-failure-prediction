"""
tests/test_history_store.py
----------------------------
Test suite for SQLite History & Database Schema Contract.
Verifies nullable failure_probability storage, database migration, and query functions.
"""

import os
import sys
import tempfile
import sqlite3
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(BACKEND_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import history_store


def test_history_store_supports_nullable_probability():
    """Verify that predictions table accepts NULL probability when ML model is unavailable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_nullable.db")
        history_store.DB_PATH = db_path
        history_store.init_db()

        # 1. Prediction with ML model available
        telemetry_1 = {"CPU_Usage": 40.0}
        res_available = {
            "prediction_available": True,
            "failure_probability": 0.72,
            "risk": "HIGH",
            "health_score": 50.0
        }
        history_store.log_prediction("DEV-001", telemetry_1, res_available)

        # 2. Prediction with ML model unavailable (failure_probability = None)
        telemetry_2 = {"CPU_Usage": 95.0}
        res_unavailable = {
            "prediction_available": False,
            "failure_probability": None,
            "heuristic_risk_score": 85.0,
            "risk": "CRITICAL",
            "health_score": 20.0
        }
        history_store.log_prediction("DEV-002", telemetry_2, res_unavailable)

        # Query history
        hist_1 = history_store.get_history("DEV-001")
        assert len(hist_1) == 1
        assert hist_1[0]["probability"] == 0.72

        hist_2 = history_store.get_history("DEV-002")
        assert len(hist_2) == 1
        assert hist_2[0]["probability"] is None


def test_db_migration_preserves_schema():
    """Verify init_db and _migrate_db run idempotently without dropping tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_mig.db")
        history_store.DB_PATH = db_path

        history_store.init_db()
        history_store.init_db()  # Second call must be smooth and idempotent

        inv = history_store.get_device_inventory()
        assert isinstance(inv, list)
