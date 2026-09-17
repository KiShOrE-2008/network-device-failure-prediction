"""
src/history_store.py
--------------------
SQLite-backed prediction history & alert logger for NetGuard NOC.

Database file: data/predictions.db (auto-created on init_db())
"""

from __future__ import annotations
import os
import sqlite3
import json
import logging
from datetime import datetime, timezone
from typing import Any, List, Dict

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(WORKSPACE_ROOT, "data", "predictions.db")

_CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL,
    probability   REAL    NOT NULL,
    health_score  REAL,
    risk          TEXT,
    risk_window   TEXT,
    failure_type  TEXT,
    anomaly_score REAL,
    is_anomaly    INTEGER,
    telemetry     TEXT,
    result_json   TEXT
);
"""

_CREATE_ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    device_id    TEXT NOT NULL,
    severity     TEXT NOT NULL,
    title        TEXT NOT NULL,
    message      TEXT NOT NULL,
    status       TEXT DEFAULT 'ACTIVE'
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_predictions_device_time ON predictions (device_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_status_time ON alerts (status, timestamp DESC);
"""

def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Idempotently initialize SQLite tables and indexes."""
    try:
        with _connect() as conn:
            conn.execute(_CREATE_PREDICTIONS_TABLE)
            conn.execute(_CREATE_ALERTS_TABLE)
            conn.executescript(_CREATE_INDEXES)
            conn.commit()
        logger.info("✅ SQLite Database initialized at %s", DB_PATH)
    except Exception as exc:
        logger.error("Failed to initialize database: %s", exc)

def log_prediction(device_id: str, telemetry: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Logs prediction inference results to SQLite database."""
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        probability = float(result.get("probability", 0.0))
        health_score = result.get("health_score")
        risk = result.get("risk", "")
        risk_window = result.get("risk_window", "")
        failure_type = result.get("failure_type", "NONE")
        anomaly_score = result.get("anomaly_score", 0.0)
        is_anomaly = 1 if result.get("is_anomaly") else 0

        telemetry_json = json.dumps(telemetry)
        result_json = json.dumps(result)

        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (device_id, timestamp, probability, health_score, risk, risk_window,
                     failure_type, anomaly_score, is_anomaly, telemetry, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    device_id, timestamp, probability, health_score, risk, risk_window,
                    failure_type, anomaly_score, is_anomaly, telemetry_json, result_json
                )
            )
            
            # Log alert if risk is HIGH/CRITICAL or anomaly detected
            if risk in ["HIGH", "CRITICAL"] or is_anomaly:
                severity = "CRITICAL" if risk == "HIGH" else "WARNING"
                title = f"{severity} Alert on {device_id}"
                message = f"Failure Risk: {probability*100:.1f}%, Diagnosis: {failure_type}, Anomaly Score: {anomaly_score}%"
                conn.execute(
                    """
                    INSERT INTO alerts (timestamp, device_id, severity, title, message)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (timestamp, device_id, severity, title, message)
                )
            conn.commit()
    except Exception as exc:
        logger.warning("Log prediction failed (non-fatal): %s", exc)

def get_history(device_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Return historical prediction records for a specific device."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, device_id, timestamp, probability, health_score, risk,
                       risk_window, failure_type, anomaly_score, is_anomaly
                FROM predictions
                WHERE device_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (device_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("History fetch failed: %s", exc)
        return []

def get_recent_global(limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent predictions across all devices."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, device_id, timestamp, probability, health_score, risk,
                       risk_window, failure_type, anomaly_score, is_anomaly
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Global history fetch failed: %s", exc)
        return []

def get_active_alerts(limit: int = 30) -> List[Dict[str, Any]]:
    """Return active alerts for NOC feed."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, device_id, severity, title, message, status
                FROM alerts
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Alerts fetch failed: %s", exc)
        return []
