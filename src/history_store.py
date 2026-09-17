"""
src/history_store.py
--------------------
Relational SQLite Database & Alert Lifecycle Store for NetGuard NOC.
Consumes netguard_noc_dataset_v1 device inventory & ground truth failure events.
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
DB_PATH = os.path.join(WORKSPACE_ROOT, "data", "network_noc.db")

_CREATE_DEVICES_TABLE = """
CREATE TABLE IF NOT EXISTS devices (
    device_id    TEXT PRIMARY KEY,
    hostname     TEXT,
    ip_address   TEXT,
    device_type  TEXT,
    vendor       TEXT,
    model        TEXT,
    location     TEXT,
    rack         TEXT,
    firmware     TEXT,
    created_at   TEXT
);
"""

_CREATE_TELEMETRY_TABLE = """
CREATE TABLE IF NOT EXISTS telemetry (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    device_id         TEXT NOT NULL,
    cpu_usage         REAL,
    memory_usage      REAL,
    temperature       REAL,
    uptime            REAL,
    interface_errors  INTEGER,
    packet_loss       REAL,
    bandwidth_usage   REAL,
    log_errors        INTEGER,
    cpu_trend         REAL,
    temperature_trend REAL,
    error_trend       REAL,
    FOREIGN KEY (device_id) REFERENCES devices (device_id)
);
"""

_CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    probability   REAL NOT NULL,
    health_score  REAL,
    risk          TEXT,
    risk_window   TEXT,
    failure_type  TEXT,
    anomaly_score REAL,
    is_anomaly    INTEGER,
    telemetry     TEXT,
    result_json   TEXT,
    FOREIGN KEY (device_id) REFERENCES devices (device_id)
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
    status       TEXT DEFAULT 'ACTIVE'  -- ACTIVE | ACKNOWLEDGED | RESOLVED
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_telemetry_device_time ON telemetry (device_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_device_time ON predictions (device_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_status_time ON alerts (status, timestamp DESC);
"""

def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Idempotently initialize all 4 relational tables and indexes."""
    try:
        with _connect() as conn:
            conn.execute(_CREATE_DEVICES_TABLE)
            conn.execute(_CREATE_TELEMETRY_TABLE)
            conn.execute(_CREATE_PREDICTIONS_TABLE)
            conn.execute(_CREATE_ALERTS_TABLE)
            conn.executescript(_CREATE_INDEXES)
            conn.commit()
        logger.info("✅ NOC SQLite DB initialized at %s", DB_PATH)
    except Exception as exc:
        logger.error("Failed to initialize NOC database: %s", exc)

def seed_devices_from_dataset():
    """Populates devices inventory table from netguard_noc_dataset_v1 metadata."""
    csv_path = os.path.join(WORKSPACE_ROOT, "data", "netguard_noc_dataset_v1", "network_devices.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(WORKSPACE_ROOT, "data", "network_devices.csv")
    if not os.path.exists(csv_path):
        return

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        with _connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
            if count == 0:
                for _, row in df.iterrows():
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO devices 
                            (device_id, hostname, ip_address, device_type, vendor, model, location, rack, firmware, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(row.get("Device_ID")),
                            str(row.get("Hostname", f"DEV-{row.get('Device_ID')}")),
                            str(row.get("IP_Address", "10.10.1.1")),
                            str(row.get("Device_Type", "Router")),
                            str(row.get("Vendor", "Cisco")),
                            str(row.get("Model", "ISR-4331")),
                            str(row.get("Location", "Chennai DC-1")),
                            str(row.get("Rack", "Rack A01")),
                            str(row.get("Firmware", "17.6.4")),
                            datetime.now(timezone.utc).isoformat()
                        )
                    )
                conn.commit()
                print("✅ Seeded 500 device inventory records into SQLite database.")

        # Seed failure ground truth events
        events_path = os.path.join(WORKSPACE_ROOT, "data", "netguard_noc_dataset_v1", "failure_events.csv")
        if os.path.exists(events_path):
            df_evt = pd.read_csv(events_path)
            with _connect() as conn:
                alert_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
                if alert_count == 0:
                    for _, evt in df_evt.head(25).iterrows():
                        conn.execute(
                            """
                            INSERT INTO alerts (timestamp, device_id, severity, title, message, status)
                            VALUES (?, ?, ?, ?, ?, 'ACTIVE')
                            """,
                            (
                                str(evt.get("Failure_Start", datetime.now(timezone.utc).isoformat())),
                                str(evt.get("Device_ID")),
                                str(evt.get("Failure_Severity", "CRITICAL")),
                                f"{evt.get('Failure_Type')} Incident on {evt.get('Device_ID')}",
                                f"Historical failure event: {evt.get('Failure_Type')} reached peak severity.",
                            )
                        )
                    conn.commit()
                    print("✅ Seeded initial NOC alerts from failure_events.csv.")
    except Exception as exc:
        logger.warning("Device/event seeding failed: %s", exc)

def log_prediction(device_id: str, telemetry: Dict[str, Any], result: Dict[str, Any]) -> None:
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
                INSERT INTO telemetry
                    (timestamp, device_id, cpu_usage, memory_usage, temperature, uptime,
                     interface_errors, packet_loss, bandwidth_usage, log_errors,
                     cpu_trend, temperature_trend, error_trend)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp, device_id,
                    float(telemetry.get("CPU_Usage", 0)), float(telemetry.get("Memory_Usage", 0)),
                    float(telemetry.get("Temperature", 0)), float(telemetry.get("Uptime", 0)),
                    int(telemetry.get("Interface_Errors", 0)), float(telemetry.get("Packet_Loss", 0)),
                    float(telemetry.get("Bandwidth_Usage", 0)), int(telemetry.get("Log_Errors", 0)),
                    float(telemetry.get("CPU_Trend", 0)), float(telemetry.get("Temperature_Trend", 0)),
                    float(telemetry.get("Error_Trend", 0))
                )
            )

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
            
            if risk in ["HIGH", "CRITICAL"] or is_anomaly:
                severity = "CRITICAL" if risk in ["HIGH", "CRITICAL"] else "WARNING"
                title = f"{severity} Incident on {device_id}"
                message = f"Failure Probability: {probability*100:.1f}%, Diagnosis: {failure_type}, Anomaly Index: {anomaly_score}%"
                conn.execute(
                    """
                    INSERT INTO alerts (timestamp, device_id, severity, title, message, status)
                    VALUES (?, ?, ?, ?, ?, 'ACTIVE')
                    """,
                    (timestamp, device_id, severity, title, message)
                )
            conn.commit()
    except Exception as exc:
        logger.warning("Log prediction failed: %s", exc)

def acknowledge_alert(alert_id: int) -> bool:
    try:
        with _connect() as conn:
            cursor = conn.execute("UPDATE alerts SET status = 'ACKNOWLEDGED' WHERE id = ?", (alert_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as exc:
        logger.warning("Acknowledge alert failed: %s", exc)
        return False

def resolve_alert(alert_id: int) -> bool:
    try:
        with _connect() as conn:
            cursor = conn.execute("UPDATE alerts SET status = 'RESOLVED' WHERE id = ?", (alert_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as exc:
        logger.warning("Resolve alert failed: %s", exc)
        return False

def get_device_inventory() -> List[Dict[str, Any]]:
    try:
        with _connect() as conn:
            rows = conn.execute("SELECT * FROM devices ORDER BY device_id ASC").fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Fetch devices failed: %s", exc)
        return []

def get_history(device_id: str, limit: int = 50) -> List[Dict[str, Any]]:
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

def get_active_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, device_id, severity, title, message, status
                FROM alerts
                WHERE status IN ('ACTIVE', 'ACKNOWLEDGED')
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Alerts fetch failed: %s", exc)
        return []
