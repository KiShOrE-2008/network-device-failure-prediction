# NetGuard NOC — Database Schema & Incident Store Specification

**Database File:** `backend/data/network_noc.db` (SQLite 3)

---

## 1. Schema ERD & Tables

### 1.1 `devices`
Device inventory registry.
- `device_id` (TEXT PRIMARY KEY)
- `hostname` (TEXT)
- `ip_address` (TEXT)
- `device_type` (TEXT)
- `vendor` (TEXT)
- `model` (TEXT)
- `location` (TEXT)
- `rack` (TEXT)
- `firmware` (TEXT)
- `status` (TEXT DEFAULT 'OPERATIONAL')
- `created_at` (TEXT)
- `updated_at` (TEXT)

### 1.2 `telemetry`
Raw and engineered time-series telemetry observations.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `timestamp` (TEXT NOT NULL)
- `device_id` (TEXT NOT NULL, FK to `devices`)
- `cpu_usage`, `memory_usage`, `temperature`, `uptime`, `interface_errors`, `packet_loss`, `bandwidth_usage`, `log_errors`

### 1.3 `predictions`
Historical inference observations logged by `FleetPredictor`.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `device_id` (TEXT NOT NULL, FK to `devices`)
- `timestamp` (TEXT NOT NULL)
- `probability` (REAL NOT NULL)
- `health_score` (REAL)
- `risk` (TEXT)
- `risk_window` (TEXT)
- `failure_type` (TEXT)
- `anomaly_score` (REAL)
- `is_anomaly` (INTEGER)
- `telemetry` (TEXT JSON)
- `result_json` (TEXT JSON)

### 1.4 `alerts` (Incidents Store)
Deduplicated NOC incident alert lifecycle table.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `device_id` (TEXT NOT NULL, FK to `devices`)
- `timestamp` (TEXT NOT NULL)
- `severity` (TEXT NOT NULL) -- `CRITICAL` | `WARNING`
- `title` (TEXT NOT NULL)
- `message` (TEXT NOT NULL)
- `status` (TEXT DEFAULT 'ACTIVE') -- `ACTIVE` | `ACKNOWLEDGED` | `RESOLVED`
- `failure_type` (TEXT)
- `probability` (REAL)
- `anomaly_score` (REAL)
- `created_at` (TEXT)
- `updated_at` (TEXT)

### 1.5 `discovered_nodes`
Discovered network endpoints table.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `device_id` (TEXT)
- `hostname` (TEXT)
- `ip_address` (TEXT UNIQUE)
- `device_type`, `vendor`, `model`, `firmware`, `status`, `discovery_protocol`, `discovered_at`

---

## 2. Incident Deduplication Logic

When predictions are logged:
1. If device risk is `HIGH` or `CRITICAL` (or `is_anomaly == 1`):
2. The store queries: `SELECT id FROM alerts WHERE device_id = ? AND status IN ('ACTIVE', 'ACKNOWLEDGED')`.
3. **Existing Incident Found:** Updates existing incident row in place (`timestamp`, `severity`, `message`, `probability`, `updated_at`). **No duplicate alert row is created.**
4. **No Active Incident:** Inserts a new row with `status = 'ACTIVE'`.
5. **Resolved Transition:** If an incident was `RESOLVED`, a subsequent critical prediction creates a new `ACTIVE` incident.
