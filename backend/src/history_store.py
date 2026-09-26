"""
src/history_store.py
--------------------
Relational SQLite Database & Deduplicated Alert Incident Store for NetGuard NOC.
Manages devices, telemetry, predictions, alerts/incidents, and discovered_nodes tables.
Enforces strict incident deduplication and status transitions (ACTIVE -> ACKNOWLEDGED -> RESOLVED).
"""

from __future__ import annotations
import os
import sys
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
    status       TEXT DEFAULT 'OPERATIONAL',
    created_at   TEXT,
    updated_at   TEXT
);
"""

_CREATE_DISCOVERED_NODES_TABLE = """
CREATE TABLE IF NOT EXISTS discovered_nodes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id          TEXT,
    hostname           TEXT,
    ip_address         TEXT NOT NULL UNIQUE,
    device_type        TEXT,
    vendor             TEXT,
    model              TEXT,
    firmware           TEXT,
    status             TEXT DEFAULT 'REACHABLE',
    discovery_protocol TEXT,
    discovered_at      TEXT
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
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    severity      TEXT NOT NULL,
    title         TEXT NOT NULL,
    message       TEXT NOT NULL,
    status        TEXT DEFAULT 'ACTIVE', -- ACTIVE | ACKNOWLEDGED | RESOLVED
    failure_type  TEXT,
    probability   REAL,
    anomaly_score REAL,
    created_at    TEXT,
    updated_at    TEXT,
    FOREIGN KEY (device_id) REFERENCES devices (device_id)
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_devices_id ON devices (device_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_device_time ON telemetry (device_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_device_time ON predictions (device_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_device_status ON alerts (device_id, status);
CREATE INDEX IF NOT EXISTS idx_alerts_status_time ON alerts (status, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_discovered_ip ON discovered_nodes (ip_address);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_db(conn: sqlite3.Connection):
    """Safely migrates existing tables to include newly required columns."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(alerts)")
    alert_cols = [row[1] for row in cursor.fetchall()]

    if "failure_type" not in alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN failure_type TEXT")
    if "probability" not in alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN probability REAL")
    if "anomaly_score" not in alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN anomaly_score REAL")
    if "created_at" not in alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN created_at TEXT")
    if "updated_at" not in alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN updated_at TEXT")

    cursor.execute("PRAGMA table_info(devices)")
    dev_cols = [row[1] for row in cursor.fetchall()]
    if "status" not in dev_cols:
        conn.execute("ALTER TABLE devices ADD COLUMN status TEXT DEFAULT 'OPERATIONAL'")
    if "created_at" not in dev_cols:
        conn.execute("ALTER TABLE devices ADD COLUMN created_at TEXT")
    if "updated_at" not in dev_cols:
        conn.execute("ALTER TABLE devices ADD COLUMN updated_at TEXT")
    conn.commit()


def init_db() -> None:
    """Idempotently initialize all 5 relational tables and indexes."""
    try:
        with _connect() as conn:
            conn.execute(_CREATE_DEVICES_TABLE)
            conn.execute(_CREATE_DISCOVERED_NODES_TABLE)
            conn.execute(_CREATE_TELEMETRY_TABLE)
            conn.execute(_CREATE_PREDICTIONS_TABLE)
            conn.execute(_CREATE_ALERTS_TABLE)
            conn.executescript(_CREATE_INDEXES)
            _migrate_db(conn)
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
        now_str = datetime.now(timezone.utc).isoformat()
        with _connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
            if count == 0:
                for _, row in df.iterrows():
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO devices 
                            (device_id, hostname, ip_address, device_type, vendor, model, location, rack, firmware, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPERATIONAL', ?, ?)
                        """,
                        (
                            str(row.get("Device_ID")),
                            str(row.get("Hostname", f"DEV-{row.get('Device_ID')}")),
                            str(row.get("IP_Address", "10.10.1.1")),
                            str(row.get("Device_Type", "Router")),
                            str(row.get("Vendor", "Cisco")),
                            str(row.get("Model", "ISR-4331")),
                            str(row.get("Location", "DC-1")),
                            str(row.get("Rack", "R01")),
                            str(row.get("Firmware", "17.6.4")),
                            now_str, now_str
                        )
                    )
                conn.commit()
    except Exception as exc:
        logger.warning("Device seeding failed: %s", exc)


def register_discovered_devices(discovered: List[Dict[str, Any]]):
    """Registers discovered endpoints into devices & discovered_nodes tables."""
    try:
        now_str = datetime.now(timezone.utc).isoformat()
        with _connect() as conn:
            for dev in discovered:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO devices
                        (device_id, hostname, ip_address, device_type, vendor, model, location, rack, firmware, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPERATIONAL', ?, ?)
                    """,
                    (
                        dev.get("device_id"), dev.get("hostname"), dev.get("ip_address"),
                        dev.get("device_type"), dev.get("vendor"), dev.get("model"),
                        dev.get("location", "DC-1"), dev.get("rack", "R01"), dev.get("firmware", "1.0"),
                        now_str, now_str
                    )
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO discovered_nodes
                        (device_id, hostname, ip_address, device_type, vendor, model, firmware, status, discovery_protocol, discovered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dev.get("device_id"), dev.get("hostname"), dev.get("ip_address"),
                        dev.get("device_type"), dev.get("vendor"), dev.get("model"),
                        dev.get("firmware", "1.0"), dev.get("status", "REACHABLE"),
                        dev.get("discovery_protocol", "SIMULATED_SNMP"), now_str
                    )
                )
            conn.commit()
    except Exception as exc:
        logger.warning("Register discovered devices failed: %s", exc)


def log_prediction(device_id: str, telemetry: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Logs prediction observation and enforces incident alert deduplication.
    Repeated predictions for the same device DO NOT create duplicate alerts;
    they update the active/acknowledged incident in place.
    """
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        probability = float(result.get("failure_probability", result.get("probability", 0.0)))
        health_score = float(result.get("health_score", 100.0))
        risk = str(result.get("risk", "LOW"))
        risk_window = str(result.get("risk_window", "Next 12 hours"))
        failure_type = str(result.get("predicted_failure", result.get("failure_type", "NONE")))
        anomaly_score = float(result.get("anomaly_score", 0.0))
        is_anomaly = 1 if anomaly_score > 70 or result.get("is_anomaly") else 0

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

            # Deduplicated Alert Incident Lifecycle
            if risk in ["HIGH", "CRITICAL"] or is_anomaly:
                severity = "CRITICAL" if risk in ["HIGH", "CRITICAL"] else "WARNING"
                title = f"{severity} Incident on {device_id}"
                message = f"Risk: {risk} ({probability*100:.1f}%), Mode: {failure_type}, Anomaly Index: {anomaly_score}%"

                # Check if an ACTIVE or ACKNOWLEDGED incident exists for device_id
                existing = conn.execute(
                    "SELECT id, status FROM alerts WHERE device_id = ? AND status IN ('ACTIVE', 'ACKNOWLEDGED')",
                    (device_id,)
                ).fetchone()

                if existing:
                    # Update existing incident observation in place (do NOT duplicate alert row)
                    conn.execute(
                        """
                        UPDATE alerts 
                        SET timestamp = ?, severity = ?, message = ?, failure_type = ?, probability = ?, anomaly_score = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (timestamp, severity, message, failure_type, probability, anomaly_score, timestamp, existing['id'])
                    )
                else:
                    # Create new incident record
                    conn.execute(
                        """
                        INSERT INTO alerts 
                            (device_id, timestamp, severity, title, message, status, failure_type, probability, anomaly_score, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?)
                        """,
                        (device_id, timestamp, severity, title, message, failure_type, probability, anomaly_score, timestamp, timestamp)
                    )

            conn.commit()
    except Exception as exc:
        logger.warning("Log prediction failed: %s", exc)


def acknowledge_alert(alert_id: int) -> bool:
    try:
        now_str = datetime.now(timezone.utc).isoformat()
        with _connect() as conn:
            cursor = conn.execute("UPDATE alerts SET status = 'ACKNOWLEDGED', updated_at = ? WHERE id = ?", (now_str, alert_id))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as exc:
        logger.warning("Acknowledge alert failed: %s", exc)
        return False


def resolve_alert(alert_id: int) -> bool:
    try:
        now_str = datetime.now(timezone.utc).isoformat()
        with _connect() as conn:
            cursor = conn.execute("UPDATE alerts SET status = 'RESOLVED', updated_at = ? WHERE id = ?", (now_str, alert_id))
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


def get_active_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, device_id, severity, title, message, status, failure_type, probability, anomaly_score, created_at, updated_at
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


def get_discovered_nodes() -> List[Dict[str, Any]]:
    try:
        with _connect() as conn:
            rows = conn.execute("SELECT * FROM discovered_nodes ORDER BY discovered_at DESC").fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Fetch discovered nodes failed: %s", exc)
        return []


init_db()
seed_devices_from_dataset()
