"""
history_store.py
----------------
SQLite-backed prediction history for NetGuard NOC.

Database file: data/predictions.db (auto-created on first init_db() call)

Public API
----------
init_db()                                          → None  (idempotent)
log_prediction(device_id, telemetry, result)       → None
get_history(device_id, limit=50)                   → list[dict]
get_recent_global(limit=20)                        → list[dict]
"""

from __future__ import annotations
import os
import sqlite3
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(WORKSPACE_ROOT, "data", "predictions.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id    TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,
    probability  REAL    NOT NULL,
    health_score REAL,
    risk         TEXT,
    risk_window  TEXT,
    telemetry    TEXT,
    result_json  TEXT
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_predictions_device_time
    ON predictions (device_id, timestamp DESC);
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the predictions table and index if they don't exist (idempotent)."""
    try:
        with _connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.commit()
        logger.info("✅ History DB initialised at %s", DB_PATH)
    except Exception as exc:
        logger.error("Failed to initialise history DB: %s", exc)


def log_prediction(
    device_id: str,
    telemetry: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """
    Persist a single prediction to the database.
    `result` should be the full enriched dict returned by /api/predict.
    Silently swallows errors so the API response is never blocked.
    """
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        probability = float(result.get("probability", 0.0))
        health_score = result.get("health_score")
        risk = result.get("risk", "")
        risk_window = result.get("risk_window", "")
        telemetry_json = json.dumps(telemetry)
        result_json = json.dumps(result)

        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (device_id, timestamp, probability, health_score,
                     risk, risk_window, telemetry, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    device_id,
                    timestamp,
                    probability,
                    health_score,
                    risk,
                    risk_window,
                    telemetry_json,
                    result_json,
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("History log failed (non-fatal): %s", exc)


def get_history(
    device_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Return the `limit` most recent predictions for a specific device,
    ordered newest first.
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, device_id, timestamp, probability,
                       health_score, risk, risk_window
                FROM predictions
                WHERE device_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (device_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.warning("History fetch failed: %s", exc)
        return []


def get_recent_global(limit: int = 20) -> list[dict[str, Any]]:
    """Return the `limit` most recent predictions across ALL devices."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT id, device_id, timestamp, probability,
                       health_score, risk, risk_window
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.warning("Global history fetch failed: %s", exc)
        return []
